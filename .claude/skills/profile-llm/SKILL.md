---
name: profile-llm
description: Profile TRT-LLM and/or vLLM GPU kernels using nsys and compare the results to explain latency differences. Use when asked to profile, compare kernel breakdown, or investigate why one framework is slower.
argument-hint: [--trtllm-cmd <cmd>] [--vllm-cmd <cmd>]
allowed-tools: [Bash, Read, Write, Edit]
---

# Profile LLM Inference (TRT-LLM vs vLLM)

Profile GPU kernel execution for TRT-LLM and/or vLLM using NVIDIA Nsight Systems (nsys), then compare kernel breakdowns.

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

**Attention** — compare `fmha_v2` (TRT-LLM) vs `BatchPrefillWithPagedKVCache` (vLLM, FlashInfer):
- Both frameworks use paged KV cache. The gap is SM120-native kernel quality: `fmha_v2` (4851ms) vs FlashInfer `BatchPrefillWithPagedKVCacheKernel` (6981ms) = 1.44× at 100k tokens.

**GEMM** — compare framework-native SM120 kernels (TRT-LLM) vs FlashInferCutlass (vLLM):
- At 15k tokens GEMMs are roughly equal; at 100k the attention gap dominates.

**Norm/quant** — broadly similar; vLLM fuses silu+fp4-quant into one kernel (`silu_mul_cvt_fp16_to_fp4`) while TRT-LLM runs them separately.

**Classifier note**: the breakdown script classifies by substring. Triton kernels named `triton_red_fused_..._scaled_fp4_quant_zeros_*` contain "quant" but not "quantize", so they fall into "other" for vLLM. Add `"quant"` to the norm/quant keyword list if you need finer accuracy.

---

## Reference: Observed Results (RTX PRO 6000, SM120, Llama 3.2 3B, FP4+FP8KV)

### 15k tokens (5 timed runs, cudaProfilerApi-gated)

| | TRT-LLM | vLLM |
|---|---|---|
| Wall latency | ~193 ms | ~256 ms |
| Attention (GPU) | 86.6 ms/run (45% of wall) | 149.6 ms/run (58% of wall) |
| GEMM (GPU) | 58.5 ms/run (30%) | 59.8 ms/run (23%) |
| Norm/quant (GPU) | 22.6 ms/run (12%) | 21.9 ms/run (9%) |

**Root cause at 15k**: attention is 1.73× slower in vLLM due to paged KV cache (indirect memory access). GEMM is essentially equal.

### 100k tokens (3 timed runs, cudaProfilerApi-gated, both with paged KV)

| | TRT-LLM | vLLM |
|---|---|---|
| Wall latency | ~4851 ms | ~6981 ms |
| Attention (GPU) | ~84% of wall | ~90% of wall |
| GEMM (GPU) | ~9% of wall | ~6% of wall |
| Norm/quant (GPU) | ~4% of wall | ~2% of wall |

**Root cause at 100k**: attention is 1.44× slower in vLLM. GEMM is within 5%. The gap is SM120-native kernel quality, not paging overhead.

Top kernel names:
- TRT-LLM attention: `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm120_kernel_nl`
- vLLM attention: `BatchPrefillWithPagedKVCacheKernel` (FlashInfer)
