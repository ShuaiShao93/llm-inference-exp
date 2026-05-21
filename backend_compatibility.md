# vLLM Attention Backend Compatibility Matrix

Empirical compatibility and latency data for vLLM attention backends at **100k input / 1 output** (prefill-dominated). Refreshed by the `vllm-backend-matrix` skill.

## How to read

- **Rows**: models. **Columns**: backends.
- **Cells**: mean latency (ms) over 5 runs, or `❌ <reason>` when the backend rejects the configuration / OOMs / errors out.
- **Default precision** for each GPU is listed under the table header. Cells using a non-default precision/KV-dtype carry a superscript footnote.
- Bold = best backend for that model.
- Lower latency is not always strictly better — but at 100k context prefill is overwhelmingly the GPU-bound work, so wall-clock difference closely tracks kernel quality.

## When to regenerate this file

Run the `vllm-backend-matrix` skill when any of these change:
- **Pinned package version** moves (each GPU section lists `vllm`, `flashinfer-python`, `flashinfer-cubin`, `triton`). The skill auto-checks current versus pinned and flags drift. Even patch releases of flashinfer / triton routinely change autotune defaults.
- New model architecture added to comparison set
- New GPU added (any compute capability not yet in the file)
- A model's quantization checkpoint is updated
- A vLLM PR lands that touches the backend you care about (e.g. attention kernel tuning, new backend, dtype gate change)

---

## NVIDIA H100 80GB HBM3 (SM90, Hopper)

Default precision: **FP8 W8A8 weights + FP8 KV cache**. Last measured **2026-05-21**.

| Package | Version |
|---|---|
| `vllm` | 0.21.0 |
| `flashinfer-python` | 0.6.8.post1 |
| `flashinfer-cubin` | 0.6.8.post1 |
| `triton` | 3.6.0 |
| `flash-attn` | vendored in vllm (tracks vllm version) |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | **2973¹** | ❌ head_size=512 unsupported | 8212² | ❌ KV-sharing not supported |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | **3506** | 9744 | 27298 | ❌ OOM at 100k (BF16 KV) |
| Ministral 3-3B Instruct (`unsloth/Ministral-3-3B-Instruct-2512-FP8`) | **3997** | 11840 | 31443 | ❌ OOM expected (BF16 KV) |

**Footnotes**:
- ¹ FA's cute (FA4) path on Gemma 4 (`head_dim=512`) asserts `q.dtype in [fp16, bf16]` when KV cache is FP8 → falls back to **BF16 KV cache** for this cell. All other cells use FP8 KV.
- ² TRITON_ATTN result reflects the tuning from [vllm#43257](https://github.com/vllm-project/vllm/pull/43257) (`num_warps=8, num_stages=2, TILE_SIZE=64` gated on `head_dim≥512`). Pre-PR baseline was 12259 ms.

**Notes**:
- FLASH_ATTN is the default to reach for on Hopper FP8 workloads — 2-9× faster than the alternatives across every working configuration.
- FLASHINFER is the fallback when FA rejects the shape (head_dim, mask combos, …). On Hopper today it underperforms FA significantly at long context.
- TRITON_ATTN is the last-resort fallback. Useful when both FA and FlashInfer reject the model (Gemma 4 + FP8 KV is the canonical case).
- FLEX_ATTENTION is not viable at 100k context — heavy metadata (block-mask building) OOMs even when the kernel itself would work.
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM (`SHORT_CONV`, `LINEAR`, `GDN_ATTN`) backends are not applicable to standard dense LLMs on NVIDIA datacenter GPUs.

---

## (Add tables for other GPUs here as they are measured.)

Empty for now. Run `vllm-backend-matrix` on the target GPU to populate.
