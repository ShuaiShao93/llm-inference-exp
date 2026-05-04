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
- **Disable chunked prefill in vLLM for long inputs with short output**: `--disable_chunked_prefill` saves ~3% at 100k tokens; negligible at ≤15k tokens. Only beneficial when there is no decode-phase batching to hide (i.e. batch size 1, output ≈ 1 token).
- **TRT-LLM vs vLLM**: TRT-LLM is ~1.3–1.4× faster at 100k input tokens, primarily because it uses SM120-native FP4 GEMM kernels (`DeviceGemmFp4GemmSm120`) and runs attention as a single kernel per layer rather than chunked FlashInfer calls.
