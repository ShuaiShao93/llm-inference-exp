# Guidelines

## Docs

- This guideline doc should be concise and most of the bullets should be 1 sentence.
- When executing the skills, if anything is found to need updates, ask if we should update it.
- For any complex flow that takes long to figure out what to do, ask if we should add a skill and/or a script for it.
- For skills, avoid too specific steps/details that are spefiic to a particular model/env or that may be changed after a dependency upgrade.
- When updating CLAUDE.md or skills, prefer durable lessons (architectural facts, hardware constraints, general patterns) over version-specific details (exact kernel names, percentages tied to a benchmark run, PR/issue numbers, current bug counts). Specific numbers belong in benchmark output or commit messages, not in long-lived docs.

## Code

- Do not commit unless you are asked to.

## Benchmarking

- scripts/ directory has scripts for running benchmark
- trtllm must be run with `PYTHONNOUSERSITE=1 ~/miniconda3/envs/trtllm/bin/python` (prevents user-local packages from shadowing the conda torch); vllm runs from the system env (`/usr/bin/python3.12`)
- To profile GPU kernels, use the `profile-llm` skill — covers both **nsys** (system-wide kernel breakdown) and **ncu** (per-kernel bottleneck analysis: compute/memory/latency-bound verdict, L2 throughput, occupancy). torch.profiler doesn't capture GPU kernels from subprocesses.
- Always pick the most optimal default arguments unless you are asked to test a specific config.
- Never run two benchmark processes at the same time. They share the GPU; concurrent runs corrupt results and can OOM.
- If a pre-quantized checkpoint at the desired precision is not available on HuggingFace, use `scripts/quantize_trtllm.py` to create one from a BF16 model. Store quantized models under `~/model_ckpt/` (NOT `/tmp` — it can be cleared on reboot).
- For HF custom-code models (e.g. DeepSeek-V2-Lite), modelopt quantization doesn't carry over `configuration_*.py` / `modeling_*.py` / `tokenization_*.py` files. Copy them from the source model dir into the quantized output dir before loading.

## Daily Maintenance

- **Check for newer vLLM and TRT-LLM versions once per session if it's been >24h since last check**. Use `pip index versions vllm` (system Python) and `pip index versions tensorrt-llm --pre` (in the trtllm conda env). Upgrade only after confirming with the user.
- Blackwell GPUs require the NVIDIA **open kernel module** variant (the `-open` package), not the proprietary one. If `nvidia-smi` fails with "requires use of the NVIDIA open kernel modules" after a driver/kernel update, install the `-open` variant of the driver.

## Optimization Lessons (RTX PRO 6000, SM120, prefill-heavy / 1-token output)

### Precision
- **FP4 > FP8 on Blackwell**: Blackwell tensor cores have native FP4 support; FP4 weights/activations are faster than FP8 for GEMM-heavy workloads.
- **FP8 is the floor for KV cache on SM120**: nvfp4 KV cache is exposed in both frameworks' APIs but no FP4-input FMHA kernel ships for SM120 (datacenter SM100 is the only architecture with FP4-KV FMHA kernels). Verify on each upgrade.
- **FP4 input only applies to GEMM activations** (W4A4 NVFP4). Q/K/V into the FMHA kernel are always FP8 (e4m3) — softmax precision is too sensitive for FP4 in the attention compute path.

### Chunked prefill / paged FMHA (TRT-LLM)
- **Chunked prefill is disabled in both scripts**: chunked prefill requires `use_paged_context_fmha=True` (paged KV access in the attention kernel), which is slower than the default contiguous gather path. Only re-enable for multi-request batching or prefix caching.
- **Contiguous KV cache would be faster in TRT-LLM** (via `enable_block_reuse=False` in KvCacheConfig), but we can't use it because `enable_block_reuse=True` is required for paged KV cache reuse across requests.

