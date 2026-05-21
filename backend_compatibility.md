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

## NVIDIA L40S 46GB (SM89, Ada)

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
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | ❌ head_size unsupported¹ | ❌ head_size=512 unsupported | **9114** | ❌ KV-sharing not supported² |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | **11179³** | 12617 | 28094³ | ❌ OOM at 100k (BF16 KV) |
| Ministral 3-3B Instruct (`unsloth/Ministral-3-3B-Instruct-2512-FP8`) | ❌ OOM | ❌ OOM | ❌ OOM | ❌ OOM |

**Footnotes**:
- ¹ On Ada, the vendored FA falls back to **FlashAttention v2** (the cute / FA4 path is Hopper+ only). FA2 doesn't support `head_dim=512`, so both the default FP8 KV cell and the BF16-KV fallback fail — unlike H100, where the FA-cute fallback handles Gemma 4 via BF16 KV.
- ² FLEX_ATTENTION rejects FP8 KV; falling back to BF16 KV trips Gemma 4's sliding-window/global KV-sharing path which FlexAttention doesn't support.
- ³ FA's cute kernel-path Q-dtype assert isn't reachable here (FA2 instead of FA4), but FA2 itself rejects FP8 KV for these models → falls back to **BF16 KV cache** for this cell. FLASHINFER's cell uses default FP8 KV.

**Notes**:
- **TRITON_ATTN is the only working backend for Gemma 4** on Ada (head_dim=512 has no FA2 / FlashInfer support).
- For Llama 3.2 3B, FLASH_ATTN (with BF16-KV fallback) is the fastest, with FLASHINFER close behind at default FP8/FP8. TRITON_ATTN is ~2.5× slower.
- **46 GB is tight at 100k context**: Ministral 3-3B OOMs across all four backends — the model is small but at vLLM's default `gpu_memory_utilization` plus a 100k KV cache the engine doesn't fit. Either drop `--max_model_len`, lower `--gpu_memory_utilization`, or use a smaller context for this model on L40S.
- FLEX_ATTENTION is not viable at 100k on Ada for the same reasons as Hopper (metadata OOM on BF16-KV fallback) plus the FP8 KV rejection.

---

## NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, Consumer Blackwell)

Default precision: **FP4 W4A4 weights + FP8 KV cache**. Last measured **2026-05-21**.

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
| Gemma 4 E4B (`cosmicproc/gemma-4-E4B-it-NVFP4`) | ❌ head_size² | ❌ head_size not supported | **5010** | ❌ KV sharing not supported |
| Llama 3.2 3B Instruct (local NVFP4, modelopt from `unsloth/Llama-3.2-3B-Instruct`) | **6378¹** | 6707 | 15433 | ❌ kv_cache_dtype not supported |
| Ministral 3-3B Instruct (`Firworks/Ministral-3-3B-Instruct-2512-nvfp4`) | **7781¹** | 8145 | 17819 | ❌ kv_cache_dtype not supported |

**Footnotes**:
- ¹ FA's cute kernel asserts on Q dtype when KV cache is FP8 on SM120 → falls back to **BF16 KV cache** for this cell (`fp4 + auto KV`). All other working cells use FP8 KV. Despite using 2× the KV memory, FA still wins at 100k latency because its SM120 path is the most tuned. Watch for vllm/flashinfer updates that fix this assert.
- ² FA doesn't work for Gemma 4 at *any* KV dtype on SM120 because Gemma 4's full-attention layers have `global_head_dim=512` (sliding-attention layers use 256). FA's SM120 build requires FA4 to support `head_size > 256`, which isn't available here. With FP8 KV the `kv_cache_dtype` check fires first; with BF16 KV the head_size check fires.

**Notes**:
- **FLASH_ATTN** is the default for Llama-family and Ministral-style dense GQA models — fastest where it works. Rejects Gemma 4 because of `global_head_dim=512` (needs FA4, not available on SM120).
- **FLASHINFER** is the FP8-KV-respecting fallback for Llama/Ministral; ~5% slower than FA at 100k. Rejects Gemma 4 entirely (head_size).
- **TRITON_ATTN** is the only backend that runs Gemma 4 here, and on Gemma 4 it's *faster* than FA/FlashInfer on the GQA models (sliding window helps). On standard GQA models it's 2-3× slower than FA.
- **FLEX_ATTENTION** is unusable on SM120 at any of these models — kv_cache_dtype gate refuses FP8 KV; on Gemma 4 it additionally rejects KV-sharing.
- Ministral 3-3B needs `--tokenizer_mode mistral` (Mistral's `tekken.json` only). The bench script passes this through when `--tokenizer_mode mistral` is set.
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.
