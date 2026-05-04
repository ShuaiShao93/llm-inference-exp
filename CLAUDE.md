# Guidelines

## Docs

- This guideline doc should be concise and most of the bullets should be 1 sentence.
- When executing the skills, if anything is found to need updates, ask if we should update it.
- For any complex flow that takes long to figure out what to do, ask if we should add a skill and/or a script for it.
- For skills, avoid too specific steps/details that are spefiic to a particular model/env or that may be changed after a dependency upgrade.

## Code

- Do not commit unless you are asked to.

## Benchmarking

- scripts/ directory has scripts for running benchmark
- trtllm must be run with `PYTHONNOUSERSITE=1 ~/miniconda3/envs/trtllm/bin/python` (prevents user-local packages from shadowing the conda torch); vllm runs from the system env (`/usr/bin/python3.12`)
- To profile GPU kernels, use the `profile-llm` skill (nsys required; torch.profiler won't capture GPU kernels from subprocesses)
- Always pick the most optimal default arguments unless you are asked to test a specific config.
- Never run two benchmark processes at the same time. They share the GPU; concurrent runs corrupt results and can OOM.
- If a pre-quantized checkpoint at the desired precision is not available on HuggingFace, use `scripts/quantize_trtllm.py` to create one from a BF16 model before benchmarking.

## Optimization Lessons (RTX PRO 6000, SM120, Llama 3.2 3B, prefill-heavy / 1-token output)

- **FP4 > FP8**: ~17% faster in TRT-LLM and ~6% faster in vLLM at 100k input tokens.
- **FP8 KV cache**: always use `--kv_cache_precision fp8`; reduces memory footprint and improves throughput with negligible accuracy impact.
- **Chunked prefill is disabled in both scripts**: chunked prefill requires `use_paged_context_fmha=True` (paged KV access in the attention kernel), which adds ~8% overhead in TRT-LLM vs the default contiguous gather path. Disabled by default; only re-enable if testing multi-request batching or prefix caching.
- **TRT-LLM vs vLLM gap is entirely attention**: at both 15k and 100k tokens GEMM is within 5%; TRT-LLM's `fmha_v2` (contiguous gather) is 1.56× faster than vLLM's FlashInfer `BatchPrefillWithPagedKVCacheKernel` (paged) at 100k tokens — the gap is SM120-native kernel quality plus contiguous vs paged memory access.
