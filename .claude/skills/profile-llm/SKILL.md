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
- **TRT-LLM requires `PYTHONNOUSERSITE=1`**: pass it via `env PYTHONNOUSERSITE=1 ...` since nsys takes the executable directly (no shell env-var prefix syntax).
- **nsys captures the entire session** (model load + warmup + timed runs): the kernel breakdown includes initialization overhead. Use call counts and per-call averages to reason about steady-state behavior, not raw totals.
- **Use a small input (e.g. 15k tokens)** for profiling so traces are fast to collect and easy to read. Findings generalize to longer sequences.

---

## Step 1: Profile TRT-LLM

```bash
nsys profile --trace=cuda,nvtx --trace-fork-before-exec=true \
  --output /tmp/profiles/trtllm_nsys \
  env PYTHONNOUSERSITE=1 ~/miniconda3/envs/trtllm/bin/python scripts/trtllm_local.py \
    --model <model> --precision <prec> --kv_cache_precision fp8 \
    --input_tokens 15000 --num_runs 3
```

---

## Step 2: Profile vLLM

```bash
nsys profile --trace=cuda,nvtx --trace-fork-before-exec=true \
  --output /tmp/profiles/vllm_nsys \
  /usr/bin/python3.12 scripts/vllm_local.py \
    --model <model> --precision <prec> --kv_cache_precision fp8 \
    --input_tokens 15000 --num_runs 3
```

---

## Step 3: Get Kernel Summaries

```bash
nsys stats /tmp/profiles/trtllm_nsys.nsys-rep -r cuda_gpu_kern_sum 2>&1 | head -40
nsys stats /tmp/profiles/vllm_nsys.nsys-rep  -r cuda_gpu_kern_sum 2>&1 | head -40
```

Key columns: `Time (%)`, `Instances`, `Avg (ns)`, kernel name.

---

## Step 4: Interpret the Results

Group kernels into three categories and compare:

**Attention kernels** — search for `flash_attention`, `BatchPrefill`, `mha`, `mmha`
- Fragmented prefill (many small calls) = chunked prefill overhead
- Single large calls per layer = better arithmetic intensity

**GEMM kernels** — search for `cutlass`, `gemm`, `GemmFp4`, `GemmSm120`
- Framework-native tile-size variants (e.g. TRT-LLM `DeviceGemmFp4GemmSm120`) are tuned for the GPU
- Many identical small calls = high launch overhead relative to compute

**Memory ops** — search for `FillFunctor`, `memcpy`, `memset`
- KV cache page zeroing (`FillFunctor<signed char>`) is a vLLM-specific cost that scales with sequence length

---

## Reference: Observed Results (RTX PRO 6000, SM120, Llama 3.2 3B, FP4+FP8KV, 15k tokens)

| Framework | Wall latency | Attention | GEMM | Notes |
|---|---|---|---|---|
| TRT-LLM | ~192 ms | `fmha_v2` single call/layer, avg 2270µs | Native `DeviceGemmFp4GemmSm120`, 3 tile variants | Fewer, larger kernels |
| vLLM | ~261 ms | FlashInfer `BatchPrefill` chunked, avg 34µs | `FlashInferCutlass`, 11k small calls avg 31µs | Chunked prefill + KV fill overhead |

Main causes of vLLM being ~1.36× slower at 15k tokens:
1. Chunked prefill fragments attention across ~12 chunks/layer vs one call
2. Generic FlashInfer CUTLASS GEMM path vs TRT-LLM's SM120-native kernels
3. KV page zeroing (`FillFunctor<signed char>`, ~14ms/run at 15k tokens)
