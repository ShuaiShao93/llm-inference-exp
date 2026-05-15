---
name: profile-llm
description: Profile TRT-LLM and/or vLLM GPU kernels with nsys (kernel breakdown) and ncu (per-kernel metrics) and compare results to explain latency differences. Use when asked to profile, compare kernel breakdown, dive into a specific kernel's bottleneck, or investigate why one framework is slower.
argument-hint: [--trtllm-cmd <cmd>] [--vllm-cmd <cmd>]
allowed-tools: [Bash, Read, Write, Edit]
---

# Profile LLM Inference (TRT-LLM vs vLLM)

Two-tool workflow: **nsys** (Nsight Systems) for system-wide kernel time breakdown and **ncu** (Nsight Compute) for per-kernel bottleneck analysis. Use nsys first to find where time is spent, then ncu to understand *why* a specific kernel is slow.

## Arguments

$ARGUMENTS

Parse optional args: `--trtllm-cmd` and `--vllm-cmd` (the benchmark commands to profile). If not provided, infer from context (model, precision, input_tokens currently being benchmarked).

## Key Caveats (Read First)

- **Both frameworks run GPU work in subprocesses**: `torch.profiler` in the main process won't capture GPU kernels. Use nsys with `--trace-fork-before-exec=true` to follow child processes.
- **TRT-LLM requires `PYTHONNOUSERSITE=1`**: pass it via `env PYTHONNOUSERSITE=1 ...` since nsys takes the executable directly.
- **Always warm up before profiling**: both scripts call `cudaProfilerStart()` after the warmup run, so nsys with `--capture-range=cudaProfilerApi` captures only the timed iterations.

---

## Step 1: Profile TRT-LLM

```bash
nsys profile --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output /tmp/profiles/trtllm_clean \
  env PYTHONNOUSERSITE=1 ~/miniconda3/envs/trtllm/bin/python scripts/trtllm_local.py \
    --model <model> --precision <prec> --kv_cache_precision fp8 \
    --input_tokens 100000 --num_runs 3
```

---

## Step 2: Profile vLLM

```bash
nsys profile --trace=cuda,nvtx \
  --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output /tmp/profiles/vllm_clean \
  /usr/bin/python3.12 scripts/vllm_local.py \
    --model <model> --precision <prec> --kv_cache_precision fp8 \
    --input_tokens 100000 --num_runs 3
```

---

## Step 3: Get Kernel Summaries

```bash
nsys stats /tmp/profiles/trtllm_clean.nsys-rep -r cuda_gpu_kern_sum 2>&1 | head -40
nsys stats /tmp/profiles/vllm_clean.nsys-rep   -r cuda_gpu_kern_sum 2>&1 | head -40
```

Key columns: `Time (%)`, `Instances`, `Avg (ns)`, kernel name.

---

## Step 4: Compute Per-Run Breakdown

Query the SQLite files to get clean per-run totals grouped by operation type:

```python
import sqlite3

def breakdown(db_path, runs, label, wall_ms):
    conn = sqlite3.connect(db_path)
    names = {r[0]: r[1] for r in conn.execute("SELECT id, value FROM StringIds")}
    rows = conn.execute("SELECT demangledName, end-start FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()

    def classify(n):
        n = n.lower()
        if any(x in n for x in ["fmha", "flash_attention", "batchprefill", "kernel_mha", "mmha", "merge_states"]):
            return "attention"
        if any(x in n for x in ["gemm", "cutlass", "gemv"]):
            return "gemm"
        if any(x in n for x in ["norm", "quantize", "cvt_fp", "fp16_to_fp4", "block_scale"]):
            return "norm/quant"
        if any(x in n for x in ["silu", "gelu", "act"]):
            return "activation"
        return "other"

    totals = {}
    for nameId, dur in rows:
        cat = classify(names.get(nameId, ""))
        totals[cat] = totals.get(cat, 0) + dur

    gpu_total_ms = sum(totals.values()) / 1e6
    print(f"\n{label}  |  GPU: {gpu_total_ms/runs:.1f} ms/run  |  wall: {wall_ms:.0f} ms/run")
    for cat in ["attention", "gemm", "norm/quant", "activation", "other"]:
        ms = totals.get(cat, 0) / 1e6 / runs
        print(f"  {cat:<14} {ms:>7.1f} ms/run  ({100*ms/wall_ms:.0f}% of wall)")
```

---

## Step 5: Interpret the Results

For each category, compare absolute time and percentage of wall:

- **Attention** is usually the dominant cost at long input lengths (O(N²)). Compare TRT-LLM's `fmha_v2_*` family vs vLLM's FlashInfer `BatchPrefillWithPagedKVCacheKernel` (or `flash_fwd_kernel` for MLA backends).
- **GEMM** covers projections (QKV / output / FFN) and MoE expert matmuls. Usually similar across frameworks for the same model.
- **Norm/quant** includes RMS norm + activation quantization. Differences here often reflect kernel fusion choices (e.g. fused silu+fp4-quant vs separate kernels).

**Classifier note**: the breakdown script classifies by substring. If a kernel name doesn't match expected keywords (e.g. Triton fused kernels with custom names), it falls into "other". Tune the keyword list per workload as needed.

---

# Part 2: Per-Kernel Bottleneck Analysis with ncu

Use ncu after nsys identifies a kernel of interest. ncu gives per-kernel metrics: SM compute throughput, memory throughput, L1/L2 cache utilization, achieved occupancy, and an automated bottleneck verdict ("latency-bound" vs "compute-bound" vs "memory-bound").

## Caveats (Read First)

- **Use ncu 2024+ for SM120**. The system default at `/usr/bin/ncu` may be too old. Check `/opt/nvidia/nsight-compute/<latest>/ncu` and prefer the newest available.
- **Profiling counters need permission.** On first use you'll hit `ERR_NVGPUCTRPERM`. Fix once with:
  ```bash
  echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo systemctl stop google-cloud-ops-agent.service     # only if it holds /dev/nvidia*
  sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
  sudo modprobe nvidia
  sudo systemctl start google-cloud-ops-agent.service
  ```
- **Write outputs to `~/ncu_profiles/`, not `/tmp`** (`/tmp/profiles` may be cleaned between sessions).
- **ncu has ~100× slowdown per profiled kernel** — limit captures with `--kernel-name regex:<name>` and `--launch-count 1-3`. Use `--launch-skip N` to bypass warmup/setup launches.
- **`--set full` collects all metrics**; `--set basic` is much faster but less detail. Use `full` once you've found the target kernel.

## Step A: Discover the Main Kernel and Its Launch Index

First find which kernel variants are launched and identify the *main prefill kernel* (not warmup/setup launches). Discovery pass with `--set basic`:

```bash
/opt/nvidia/nsight-compute/<ver>/ncu \
    --kernel-name regex:"fmha_v2|flash_fwd|BatchPrefill" \
    --launch-skip 100 --launch-count 50 \
    --section LaunchStats --section SpeedOfLight \
    -o ~/ncu_profiles/discover \
    --force-overwrite \
    <benchmark command>
```

Group captured launches by grid size to identify the prefill kernel (typically the largest grid):

```bash
/opt/nvidia/nsight-compute/<ver>/ncu --import ~/ncu_profiles/discover.ncu-rep --csv --section SpeedOfLight | \
  python3 -c "
import sys, csv
launches = {}
for row in csv.DictReader(sys.stdin):
    if row['Metric Name'] == 'Duration':
        v = float(row['Metric Value'])
        u = row['Metric Unit']; v = v*1000 if u=='ms' else v/1000 if u=='ns' else v
        launches.setdefault((row['Grid Size'], row['Block Size']), []).append(v)
for (g,b),d in sorted(launches.items(), key=lambda x:-sum(x[1])):
    print(f'{g:<25} {b:<15} n={len(d):>4} mean={sum(d)/len(d):>8.1f}us total={sum(d):>9.1f}us')
"
```

Pick the row with the largest mean duration and matching grid shape — that's the main kernel. Note its launch index (you'll need `--launch-skip` for the targeted profile).

## Step B: Profile a Single Representative Launch

```bash
/opt/nvidia/nsight-compute/<ver>/ncu \
    --kernel-name regex:"<exact kernel name>" \
    --launch-skip <N> --launch-count 1 --set full \
    -o ~/ncu_profiles/<label>_attn \
    --force-overwrite \
    <benchmark command>
```

Choose `--launch-skip` so the captured launch is in the timed-run phase (past warmup). For most prefill cases, `--launch-skip 100` is enough to skip past warmup variants.

## Step C: Extract Key Metrics

```bash
/opt/nvidia/nsight-compute/<ver>/ncu --import ~/ncu_profiles/<label>_attn.ncu-rep --page details 2>&1 | \
  grep -E "^[[:space:]]+(Duration|DRAM Throughput|Memory Throughput|L1/TEX Cache Throughput|L2 Cache Throughput|Compute \(SM\) Throughput|Block Size|Grid Size|Registers Per Thread|Theoretical Occupancy|Achieved Occupancy)"
```

## Step D: Interpret the Verdict

NVIDIA's automated SOL (Speed of Light) analysis classifies each kernel:

| Verdict | Meaning | What to look for |
|---|---|---|
| **Compute-bound** (SM throughput > 80%) | Healthy; saturating the tensor pipe | Look for ways to reduce FLOPs (precision, algorithm) |
| **Memory-bound** (Memory throughput > 80%, SM throughput < 60%) | Reading/writing too much DRAM | Smaller working set, better cache reuse |
| **Latency-bound** (both throughputs < 60%) | Stalled waiting (instruction issue, memory latency, dependency) | Many small launches → bigger work per kernel; warp stalls; low occupancy |

Other metrics that matter:
- **L2 throughput**: 30%+ means good cache reuse; 10%- means working set spills to DRAM
- **Achieved occupancy**: < 25% often limited by registers/shared memory (check Theoretical Occupancy notes)
- **DRAM throughput (GB/s)** absolute number compared to peak (~3 TB/s on RTX PRO 6000 / SM120)

## Step E: Compare Multiple Kernels

To compare attention kernels across frameworks (or attention backends within vLLM), profile each separately and tabulate the same metrics from Step C side by side. Key columns to compare: per-launch duration, total number of launches (from nsys), grid size, compute (SM) throughput, L2 throughput, DRAM GB/s, and the SOL verdict.

Durable patterns to look for:

- **Many small kernel launches → latency-bound**: even if each individual launch is fast, kernel-launch overhead dominates. Look for >>thousands of launches per run.
- **Few monolithic per-layer launches → compute- or memory-bound**: this is the healthy state. One large grid per layer saturates the SMs.
- **High L2 throughput (>30%) means the kernel is reusing cache**, reducing the impact of larger per-token KV footprint.
- **A well-tuned hardware-native kernel** can outperform an algorithmically-clever-but-less-tuned one (e.g. a hand-tuned GQA kernel beating a less-optimized MLA implementation, despite MLA's smaller KV).
