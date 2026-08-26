# Agent index

Skills live in `.claude/skills/<name>/SKILL.md`. **Add a row here when you add a skill.**

## Skills

| Skill | Use it when |
|---|---|
| `vllm-backend-matrix` | Deciding which attention backend / precision to run for a model on a given GPU, or refreshing `backend_compatibility.md` because a row is stale. Owns that file. |
| `benchmark-gemma4` | Benchmarking Gemma 4, or hitting a `head size not supported` / `kv_cache_sf must be provided` error. Covers the split `head_dim` and the FP4 + FP8-KV patches. |
| `profile-llm` | Getting a kernel breakdown (nsys) or a per-kernel bottleneck verdict (ncu), or explaining why one framework or backend is slower. |
| `optimize-llm` | Quantifying compute/DRAM utilization end-to-end and at kernel level, and turning the gap into ranked optimization targets. |
| `lora-cost` | Anything about what a LoRA adapter costs at inference — the fixed-vs-per-token split, punica traffic, `max_lora_rank` padding, choosing or building an adapter. |
| `setup-trtllm` | Standing up TensorRT-LLM on a fresh Ubuntu box with an NVIDIA GPU. |
| `upgrade-cuda` | Bumping the NVIDIA driver or CUDA toolkit. Covers the reboot requirement and PATH-shadowing traps. |

## Docs

| File | What it is |
|---|---|
| `CLAUDE.md` | Standing rules: benchmarking discipline, quantization tool selection, daily version-check duty, hardware constraints. Read first. |
| `backend_compatibility.md` | **Report only** — measured latency per (GPU × model × backend), plus infra cost per request. Method and lessons live in the skills. |
| `scripts/` | Benchmark harnesses (`vllm_local.py`, `trtllm_local.py`), quantization, and the LoRA/profile analysis tools. |
