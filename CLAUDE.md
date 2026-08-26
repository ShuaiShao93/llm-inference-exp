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
- **Quantization tool selection**: prefer a pre-quantized checkpoint from HuggingFace when available (search for `<model> NVFP4`, `<model> FP8`). When you need to quantize yourself: (1) `scripts/quantize_trtllm.py` uses `nvidia-modelopt` which lags transformers updates by months — newer models (e.g. Gemma 4) may not work; (2) for those, install `llm-compressor` (vLLM project) which tracks transformers releases more closely, but currently produces W4A16 not W4A4 NVFP4. Don't upgrade transformers/modelopt in the trtllm conda env to fix this — it breaks TRT-LLM's pinned deps.
- **vLLM 8-bit tiers quantize online; sub-8-bit does not.** Pass a vLLM online scheme to `--precision` (e.g. `fp8_per_channel`, which matches llmcompressor's `FP8_DYNAMIC` recipe) and one BF16 base serves every 8-bit tier on every GPU — no per-precision checkpoint hunt. Below 8 bits, vLLM's online schemes set only the MoE spec, so on a dense model every Linear layer silently stays BF16 and the run *succeeds* reporting BF16 latency under an FP4/INT8 label; `vllm_local.py` refuses this rather than measuring it. Use a pre-quantized checkpoint for sub-8-bit. Note a config can carry MoE keys with experts disabled (Gemma 4 E2B/E4B: `enable_moe_block: False`) — check the values, not the keys.
- **Quantization format ≠ portability across frameworks**: vLLM accepts both `modelopt`-format and `compressed-tensors`-format NVFP4. TRT-LLM only accepts modelopt format (with `hf_quant_config.json`); it rejects `compressed-tensors` 4-bit (`Unsupported quant_bits: 4. Supported: 8`) and most non-modelopt FP8 schemes. For a fair vLLM-vs-TRT-LLM comparison, quantize the BF16 base with `scripts/quantize_trtllm.py`. For multimodal models (e.g. `Mistral3ForConditionalGeneration`), if the vendor only ships pre-quantized weights (no BF16 base), TRT-LLM is generally not benchmarkable without significant extraction work.

## Daily Maintenance

- **Check for newer vLLM and TRT-LLM versions once per session if it's been >24h since last check, and ALWAYS before any benchmark run or `vllm-backend-matrix` invocation**. Use `pip index versions vllm` (system Python) and `pip index versions tensorrt-llm --pre` (in the trtllm conda env). Also diff the installed versions against the pinned-version blocks already in `backend_compatibility.md` — if any existing row was measured on a newer stack than what's currently installed, prompt the user to upgrade *first* so new rows are comparable to existing ones. Upgrade only after confirming with the user.
- **Only three things are upgrade candidates: `vllm`, the NVIDIA driver, and the CUDA toolkit.** `flashinfer-python`, `torch` and `triton` are pinned *exactly* by vLLM (0.27.1 declares `flashinfer-python==0.6.16.post3`), so install `vllm==<latest>` and take whatever it resolves — never bump them independently. **A newer `flashinfer-python` on PyPI is therefore not an upgrade signal**; it is normally ahead of every released vLLM, and overriding the pin fabricates kernel errors that read like genuine backend incompatibilities. It becomes adoptable when a vLLM release pins it, not when it appears. `transformers` is the exception that needs a manual ceiling — vLLM declares only a floor (`>=5.5.3`), so a fresh install can pick up a release that breaks config validation. Full rationale in the `vllm-backend-matrix` skill under "Which versions to pin".
- **After any vllm / flashinfer-python / triton upgrade**, diff the new versions against the version block at the top of each GPU section in `backend_compatibility.md` and run the `vllm-backend-matrix` skill on any GPU whose pinned versions are now stale. Even patch releases regularly change attention autotune defaults.
- Blackwell GPUs require the NVIDIA **open kernel module** variant (the `-open` package), not the proprietary one. If `nvidia-smi` fails with "requires use of the NVIDIA open kernel modules" after a driver/kernel update, install the `-open` variant of the driver.

## Optimization Lessons (Blackwell, prefill-heavy / 1-token output)

### Precision
- **FP4 > FP8 on Blackwell**: Blackwell tensor cores have native FP4 support; FP4 weights/activations are faster than FP8 for GEMM-heavy workloads.
- **KV-cache precision depends on SM and head_dim**: NVFP4 KV is exposed in both frameworks' APIs, but the FMHA cubin set is uneven — consumer Blackwell (SM120) has no FP4-input FMHA at all, and datacenter Blackwell (SM100/B200) ships FP4-KV cubins only for some head_dim/shape combinations (e.g. head_dim=512 has prefill cubins but not decode). When in doubt fall back to FP8 KV. Re-check after each FlashInfer/TRT-LLM upgrade — check the cubin manifest under `flashinfer_cubin/cubins/.../fmha/trtllm-gen/`. Check actual hardware with `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` before assuming.
- **FP4 input only applies to GEMM activations** (W4A4 NVFP4). Q/K/V into the FMHA kernel are always FP8 (e4m3) — softmax precision is too sensitive for FP4 in the attention compute path.

### Attention backends
- When a vLLM attention backend rejects a model or underperforms, consult the per-backend compatibility matrix at `vllm-project/vllm/docs/design/attention_backends.md` (supported head sizes, Q/KV dtypes, GPU CC ranges) before switching backends or filing a bug.
- **For empirical "which backend on which GPU" decisions**, consult `backend_compatibility.md` at the repo root — measured latency at 100k input per (model × backend) for each GPU we've benchmarked. Regenerate via the `vllm-backend-matrix` skill when a row is stale (new vLLM version, new model, new GPU) or when a measurement contradicts the current table.

### Chunked prefill / paged FMHA (TRT-LLM)
- **Chunked prefill is disabled in both scripts**: chunked prefill requires `use_paged_context_fmha=True` (paged KV access in the attention kernel), which is slower than the default contiguous gather path. Only re-enable for multi-request batching or prefix caching.
- **Contiguous KV cache would be faster in TRT-LLM** (via `enable_block_reuse=False` in KvCacheConfig), but we can't use it because `enable_block_reuse=True` is required for paged KV cache reuse across requests.

