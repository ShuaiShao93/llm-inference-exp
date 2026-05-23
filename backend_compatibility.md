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

---

## NVIDIA B200 180GB HBM3e (SM100, Datacenter Blackwell)

Default precision: **FP4 W4A4 weights + FP8 KV cache**. Last measured **2026-05-22**.

| Package | Version |
|---|---|
| `vllm` | 0.21.0 + 2 local patches (see §) |
| `flashinfer-python` | 0.6.11.post3 |
| `flashinfer-cubin` | 0.6.11.post3 |
| `triton` | 3.6.0 |
| `flash-attn` | vendored in vllm (tracks vllm version) |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**Version-drift note:** `flashinfer-python` / `flashinfer-cubin` are **newer** than the 0.6.8.post1 used in the H100, L40S, and SM120 sections above. The upgrade was required for `head_dim=512` cubin coverage ([flashinfer#2959](https://github.com/flashinfer-ai/flashinfer/pull/2959)). Cross-GPU comparisons against the older sections are not strictly apples-to-apples until those sections are refreshed against the newer flashinfer release on their hardware.

**vLLM local patches in effect for these numbers** (both in `vllm/v1/attention/backends/flashinfer.py`):
1. [vllm#38822](https://github.com/vllm-project/vllm/pull/38822) — `head_dim=512` added to `FlashInferBackend.get_supported_head_sizes()`. Unblocks Gemma 4 full-attention layers from reaching the FlashInfer call.
2. uint8 → `torch.float8_e4m3fn` view bridge in `FlashInferImpl.forward` right after `kv_cache_permute = fixed`. vLLM stores FP8 KV with uint8 backing; since flashinfer#2954, the trtllm-gen kernels treat uint8 unambiguously as NVFP4 and raise `kv_cache_sf must be provided for NVFP4 KV cache.` Other vLLM v1 backends already do this view (`triton_attn.py:570-571`, `rocm_attn.py:416-417`, `rocm_aiter_unified_attn.py:204-205`); FlashInfer was the only one missing it. Without this patch, every FLASHINFER cell here (and on any GPU with flashinfer ≥ 0.6.11 and FP8 KV) would fail.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E4B (`cosmicproc/gemma-4-E4B-it-NVFP4`) | ❌ kv_cache_dtype not supported² | **913³** | 9781 | ❌ kv_cache_dtype not supported |
| Llama 3.2 3B Instruct (`inference-optimization/Llama-3.2-3B-Instruct-NVFP4`) | 1643¹ | **1275** | 22705 | ❌ kv_cache_dtype not supported |
| Ministral 3-3B Instruct (`Firworks/Ministral-3-3B-Instruct-2512-nvfp4`) | 1944¹ | **1500** | 26091 | ❌ kv_cache_dtype not supported |

**Footnotes**:
- ¹ FA's cute (FA4) kernel asserts on Q dtype when KV cache is FP8 on Blackwell → falls back to **BF16 KV cache** (`fp4 + auto KV`). All other working cells use FP8 KV. Same SM120 footnote ¹ pattern — the assert hasn't been fixed in vLLM 0.21.0.
- ² FA doesn't work for Gemma 4 at *any* KV dtype on B200 either: `head_dim=512` plus FP8 KV trips the kv_cache_dtype check; FA4 supports head_size up to 512 on SM100 but the FP8-KV gate fires first. With BF16 KV fallback the head-size check would clear, but FA's KV-sharing handling for Gemma 4's sliding/global mix doesn't apply here — same architectural mismatch as on SM120.
- ³ FLASHINFER on Gemma 4 only works with **both** local vLLM patches above. Without #38822 it fails at backend selection (head_size 512); without the uint8-view bridge it fails at first kernel dispatch with `kv_cache_sf must be provided for NVFP4 KV cache.` The kernel actually used is `fmhaSm100fKernel_QkvE4m3OBfloat16H512HVPerCta256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` (split-CTA, per-CTA-V=256 — see the `benchmark-gemma4` skill for why head_dim=512 attention is ~2.4× the cost of head_dim=256).

**Notes**:
- **FLASHINFER is the default on B200** for all three models tested — beats FLASH_ATTN by 20-30% on Llama/Ministral, and is the only viable option for Gemma 4 (after patches). This is the opposite of SM120, where FA wins on GQA models; the inversion comes from FlashInfer 0.6.11's SM100 trtllm-gen tuning being further along than FA4's SM100 path for these shapes.
- **FLASH_ATTN** still works on Llama/Ministral but pays the BF16-KV memory tax. Rejects Gemma 4 outright. Same Q-dtype assert as SM120 — watch for vllm 0.22 / flashinfer upgrades.
- **TRITON_ATTN** is 7-17× slower than FLASHINFER on B200 — only useful as a correctness baseline. Works on every model including Gemma 4.
- **FLEX_ATTENTION** is unusable for the same kv_cache_dtype reason as SM120.
- B200 has plenty of HBM (180 GB), so OOM is not a constraint at 100k context for these models, unlike L40S (46 GB).
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.

---

## NVIDIA A100 SXM4 40GB (SM80, Ampere)

Default precision: **INT8 W8A8 weights + BF16 KV cache**. Last measured **2026-05-23**.

| Package | Version |
|---|---|
| `vllm` | 0.21.0 |
| `flashinfer-python` | 0.6.11.post3 |
| `flashinfer-cubin` | 0.6.11.post3 |
| `triton` | 3.6.0 |
| `flash-attn` | vendored in vllm (tracks vllm version) |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**Version-drift note:** `flashinfer-python` / `flashinfer-cubin` 0.6.11.post3 here matches the B200 section; H100, L40S, and SM120 sections were measured on the older 0.6.8.post1 and should be refreshed before cross-GPU comparisons against this row.

**Precision-drift note:** Ampere has **no native FP8 or FP4 tensor cores** — those paths would fall back to BF16 dequant and defeat the point. INT8 tensor cores have been available since Turing (SM 7.5), so W8A8 INT8 (CompressedTensors `int-quantized`) is the natural quantized baseline. `kv_cache_dtype=auto` resolves to **BF16** here (the model's compute dtype); vLLM's FP8 KV path requires an FP8-capable SM (≥ 8.9).

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E4B (`nunusadmqk/gemma-4-E4B-it-W8A8-INT8-v10-datafree`) | ❌ head_size unsupported¹ | ❌ head_size=512 unsupported | **21101** | ❌ KV-sharing not supported |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8`) | **11666** | ❌ OOM² | 41059 | ❌ OOM² |
| Ministral 3-3B Instruct (`Ministral-3-3B-Instruct-2512-w8a8`, locally quantized³) | ❌ OOM⁴ | ❌ OOM⁴ | ❌ OOM⁴ | ❌ OOM⁴ |

**Footnotes**:
- ¹ On Ampere, the vendored FA uses **FlashAttention v2** (FA-cute / FA4 are Hopper+). FA2 rejects `head_dim>256`, so Gemma 4's global-attention layers (`head_dim=512`) trip the head-size check. Same pattern as the L40S section. FLASHINFER rejects for the same reason on this SM.
- ² FLASHINFER and FLEX_ATTENTION OOM for Llama 3.2 3B on the 40 GB A100 with flashinfer **0.6.11.post3**. The earlier measurement on flashinfer 0.6.8.post1 had FLASHINFER passing at ~10.7 s; the 0.6.11 upgrade added enough workspace bulk to push FLASHINFER over the 40 GB budget at 100k context (FA still fits because it doesn't preallocate the same prefill workspace). On 80 GB A100 or with `--gpu_memory_utilization` lowered, FLASHINFER would likely run again.
- ³ Ministral 3-3B's published checkpoints are FP8/NVFP4; the Mistral-recommended W8A8 INT8 quantization for this revision had to be produced locally with `auto-round` against the BF16 base `unsloth/Ministral-3-3B-Instruct-2512` (RTN mode — calibration doesn't affect latency-only benchmarks).
- ⁴ Ministral 3-3B is a `Mistral3ForConditionalGeneration` **multimodal** model — its vision tower stays BF16 (only the language-model linear layers are quantized to INT8). The unquantized vision-tower bulk plus 100k-context KV cache for the text model overflows 40 GB on every backend. On 80 GB A100 this row should look more like the H100/B200 rows.

**Notes**:
- **FLASH_ATTN is the default on A100** for W8A8 INT8 long-context prefill where the model fits — Llama 3.2 3B is the only row that completed at default config. FA2 keeps a smaller workspace than FLASHINFER 0.6.11, which is what saves it on the 40 GB card.
- **FLASHINFER** would be the natural choice on a larger A100 (80 GB) — the kernel itself runs fine; only the workspace allocation OOMs on 40 GB after the 0.6.11 upgrade.
- **TRITON_ATTN** is the universal fallback — works on every model that fits, including Gemma 4 (the only working backend for `head_dim=512` on Ampere). ~3.5× slower than FA on Llama 3.2 3B at long context.
- **FLEX_ATTENTION** is unusable at 100k on a 40 GB Ampere GPU: rejects Gemma 4's sliding/global KV-sharing, OOMs on the dense models. Same pattern as the H100 (80 GB) section where FLEX OOMs even with more headroom.
- **40 GB is the binding constraint** for this row — three of four backends OOM on the dense Llama 3.2 3B, and Ministral OOMs across the board. The matrix would be much more readable on an 80 GB A100; rows here primarily document what vLLM rejects vs. what just doesn't fit.
- **INT8 W8A8 vs other quant schemes**: vLLM picks `CutlassInt8ScaledMMLinearKernel` for the matmul; attention runs in BF16 against the BF16 KV cache. INT8 activations are quantized per-token dynamically (the `int-quantized` strategy with `act_group_size=-1`).
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.
