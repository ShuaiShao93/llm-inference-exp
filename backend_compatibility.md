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
- **Pinned package version** moves (each GPU section lists `vllm`, `flashinfer-python`, `flashinfer-cubin`, `triton`, `cuda-driver`, `cuda-toolkit`). The skill auto-checks current versus pinned and flags drift. Even patch releases of flashinfer / triton routinely change attention autotune defaults; CUDA driver/toolkit bumps can change JIT-compiled kernel codegen and trigger FlashInfer cubin redownloads.
- New model architecture added to comparison set
- New GPU added (any compute capability not yet in the file)
- A model's quantization checkpoint is updated
- A LoRA adapter for one of the test models is updated (the matrix is benchmarked with LoRA loaded — see table below)
- A vLLM PR lands that touches the backend you care about (e.g. attention kernel tuning, new backend, dtype gate change)

## LoRA adapters used for every benchmark

Each model in the matrix is benchmarked with a LoRA adapter loaded, so the cells exercise the backend's LoRA dispatch path (closer to production deployment than a no-LoRA baseline). All adapters are pinned at **rank 16, alpha 16**, targeting the 7 standard projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, `down_proj`) so the LoRA compute shape is identical across models and only the base architecture varies.

| Base model | LoRA adapter | r / α | Source |
|---|---|---|---|
| `prithivMLmods/gemma-4-E4B-it-FP8` (Gemma 4 E4B) | `Semaj90/gemma4-e4b-legal-grpo` | 16 / 16 | HF (real). Already excludes `vision_tower.*`, `audio_tower.*`, `multi_modal_projector.*`. |
| `prithivMLmods/gemma-4-E2B-it-FP8` (Gemma 4 E2B) | `~/model_ckpt/synthetic-loras/gemma-4-e2b-r16-stripped` | 16 / 16 | HF `imranulhaquenoor/gemma4-e2b-urdu-tutor-lora` (r=16/α=16) post-processed by `scripts/strip_tower_lora.py` to drop `vision_tower.*` / `audio_tower.*` / `multi_modal_projector.*` weights (the source LoRA was trained on the full multimodal model). |
| `RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic` (Llama 3.2 3B) | `~/model_ckpt/synthetic-loras/llama-3.2-3b-r16` | 16 / 16 | Synthetic (built by `scripts/build_synthetic_lora.py`; random weights). |
| `unsloth/Ministral-3-3B-Instruct-2512-FP8` (Ministral 3-3B) | `~/model_ckpt/synthetic-loras/ministral-3-3b-r16` | 16 / 16 | Synthetic (built by `scripts/build_synthetic_lora.py`; random weights, towers excluded). |

The Gemma 4 rows use real HF adapters because PEFT can't currently wrap Gemma 4's `Gemma4ClippableLinear` (raises *"Target module ... is not supported"*) — we can't generate one synthetically. The E4B row's adapter happens to be tower-clean already; the E2B row's adapter needed a post-process pass (`scripts/strip_tower_lora.py`) to drop multimodal-tower weights that vLLM rejects.

To regenerate the synthetic adapters (e.g. when a base model is replaced), run:

```bash
/usr/bin/python3.12 scripts/build_synthetic_lora.py \
  --base RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic \
  --out ~/model_ckpt/synthetic-loras/llama-3.2-3b-r16
/usr/bin/python3.12 scripts/build_synthetic_lora.py \
  --base unsloth/Ministral-3-3B-Instruct-2512-FP8 \
  --out ~/model_ckpt/synthetic-loras/ministral-3-3b-r16
```

Synthetic-weight performance is identical to real-weight performance (LoRA dispatch only cares about r/α/target_modules, not the weight values), so this lets us match `r/α` across models without hunting for an HF adapter at every desired rank.

---

## NVIDIA H100 80GB HBM3 (SM90, Hopper)

Default precision: **FP8 W8A8 weights + FP8 KV cache**. Last measured **2026-06-12**.

| Package | Version |
|---|---|
| `vllm` | 0.22.1 |
| `flashinfer-python` | 0.6.12 |
| `flashinfer-cubin` | 0.6.12 |
| `triton` | 3.7.0 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `cuda-driver` | 610.43.02 |
| `cuda-toolkit` | 13.3 |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`prithivMLmods/gemma-4-E2B-it-FP8`) | ❌ CUBLAS crash on BF16-KV fallback³ | ❌ FlashInfer kernel template missing on SM90⁴ | **12388**² | ❌ FP8 KV cache unsupported |
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | **3814¹** | ❌ FlashInfer kernel template missing on SM90⁴ | 13540² | ❌ FP8 KV cache unsupported |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | **3964** | 10166 | 26656 | ❌ FP8 KV cache unsupported |
| Ministral 3-3B Instruct (`unsloth/Ministral-3-3B-Instruct-2512-FP8`) | **4443** | 12194 | 29958 | ❌ FP8 KV cache unsupported |

**Footnotes**:
- ¹ FA on Gemma 4 (`head_dim=512`) with FP8 KV now hits `AssertionError: FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe` (vllm 0.22.1 made this gate explicit, was a q.dtype assert before). Falls back to **BF16 KV cache**. All other cells use FP8 KV.
- ² TRITON_ATTN on Gemma 4 head_dim=512: the [vllm#43257](https://github.com/vllm-project/vllm/pull/43257) tuning (`num_warps=8, num_stages=2, TILE_SIZE=64` gated on `head_dim≥512`) **is not present in vLLM 0.22.1's bundled `triton_unified_attention.py`** — the file was significantly rewritten between releases and the tune did not survive. E4B TRITON_ATTN regressed from 9003 ms (0.21.0 + PR patch) to 13540 ms (0.22.1 stock). Re-upstreaming the tune onto the 0.22.x file is a TODO.
- ³ E2B + LoRA + FA at 100k still crashes on the BF16-KV fallback path: `cublasGemmEx → CUBLAS_STATUS_EXECUTION_FAILED → cudaErrorIllegalAddress`. The bug fix in 0.22.1 only repaired the TRITON_ATTN path (E2B × TRITON_ATTN now succeeds at 12388 ms vs CUBLAS crash on 0.21.0). FA path remains broken — likely the same narrow-KV (num_kv_heads=1) + LoRA edge case but in a different GEMM call. Isolation:  E2B no-LoRA at 100k works ✓ ;  E2B + LoRA at 1k works ✓ ;  E2B + LoRA + FA at 100k fails ✗.
- ⁴ vllm 0.22.1 picked up [vllm#38822](https://github.com/vllm-project/vllm/pull/38822) (FlashInfer head_dim=512 whitelist) so FLASHINFER no longer rejects Gemma 4 at backend selection. Dispatch now reaches the FlashInfer kernel, which then fails with *"Invalid configuration"* because no SM90 cubin exists for head_dim=512 (the prebuilt cubins target Blackwell SM100+ only). End result is the same — FLASHINFER unusable for Gemma 4 on Hopper — but the failure mode moved one layer deeper.

**Notes**:
- All cells are measured with a LoRA adapter loaded (r=16, α=16, 7 standard projection modules). See the "LoRA adapters used for every benchmark" table near the top of this file.
- FLASH_ATTN is the default to reach for on Hopper FP8 workloads — 2.5-6.7× faster than the alternatives across every working configuration here.
- FLASHINFER is the fallback when FA rejects the shape (head_dim, mask combos, …). On Hopper today it underperforms FA significantly at long context.
- TRITON_ATTN is the last-resort fallback. Useful when both FA and FlashInfer reject the model (Gemma 4 + FP8 KV is the canonical case — FA falls back to BF16 KV there, TRITON_ATTN is needed if you must keep FP8 KV).
- FLEX_ATTENTION uniformly rejects FP8 KV cache on this stack — not viable for FP8-KV deployments on H100.
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
| `cuda-driver` | _(not recorded — rerun matrix on this GPU to capture)_ |
| `cuda-toolkit` | _(not recorded — rerun matrix on this GPU to capture)_ |

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

Default precision: **FP4 W4A4 weights + FP8 KV cache**. Last measured **2026-06-12**.

| Package | Version |
|---|---|
| `vllm` | 0.22.1 |
| `flashinfer-python` | 0.6.12 |
| `flashinfer-cubin` | 0.6.12 |
| `triton` | 3.7.0 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `cuda-driver` | 610.43.02 |
| `cuda-toolkit` | 13.3 |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**Version-drift note:** `vllm 0.22.1` and `flashinfer 0.6.12` are **newer** than the H100 (0.21.0 / 0.6.11.post3), L40S, B200, and A100 rows. Cross-GPU comparisons against those sections aren't strictly apples-to-apples until they're refreshed on the same stack. `vllm 0.22.1` requires `torch==2.11.0` (pinned) and declares `flashinfer-python==0.6.11.post2`; we override flashinfer to 0.6.12 and triton to 3.7.0 — the pip dep-resolver warnings about this combination are harmless (the wheels are ABI-compatible and the smoke test passes).

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E4B (`cosmicproc/gemma-4-E4B-it-NVFP4`) | ❌ kv_cache_dtype² | ❌ kernel template missing³ | **6726** | ❌ KV sharing not supported |
| Llama 3.2 3B Instruct (local NVFP4, modelopt from `unsloth/Llama-3.2-3B-Instruct`) | **6885¹** | 7244 | 14858 | ❌ kv_cache_dtype not supported |
| Ministral 3-3B Instruct (`Firworks/Ministral-3-3B-Instruct-2512-nvfp4`) | **8302¹** | 8720 | 16783 | ❌ kv_cache_dtype not supported |

All cells are measured with a LoRA adapter loaded (r=16, α=16, 7 standard projection modules) — see the "LoRA adapters used for every benchmark" table at the top of this file. Adapters used here: `Semaj90/gemma4-e4b-legal-grpo` for Gemma 4, synthetic LoRAs (built via `scripts/build_synthetic_lora.py` against the BF16 source models) for Llama and Ministral. Comparable no-LoRA numbers from the previous matrix run (vllm 0.21.0 + flashinfer 0.6.8.post1 + triton 3.6.0 + driver 580 + toolkit 13.2) were Gemma-TRITON 5010 / Llama-FA 6378 / Llama-FLASHINFER 6707 / Llama-TRITON 15433 / Ministral-FA 7781 / Ministral-FLASHINFER 8145 / Ministral-TRITON 17819. LoRA + stack-bump together added ~7-8% on FA / FLASHINFER, ~34% on Gemma TRITON, and gave a ~5% speedup on Llama/Ministral TRITON_ATTN.

**Footnotes**:
- ¹ FA's cute kernel asserts on Q dtype when KV cache is FP8 on SM120 → falls back to **BF16 KV cache** for this cell (`fp4 + auto KV`). All other working cells use FP8 KV. Despite using 2× the KV memory, FA still wins at 100k latency because its SM120 path is the most tuned. Watch for vllm/flashinfer updates that fix this assert.
- ² FA doesn't work for Gemma 4 at *any* KV dtype on SM120 because Gemma 4's full-attention layers have `global_head_dim=512` (sliding-attention layers use 256). FA's SM120 build requires FA4 to support `head_size > 256`, which isn't available here. With FP8 KV the `kv_cache_dtype` check fires first; with BF16 KV the head_size check fires.
- ³ flashinfer 0.6.12 added head_size=512 to its dtype gate (no longer "head_size not supported"), but the kernel template for the specific (head_size=512, FP8 KV, paged) combination isn't compiled in the prebuilt cubin set for SM120. Reported by FlashInfer at first kernel dispatch as "Invalid configuration."

**Notes**:
- **FLASH_ATTN** is the default for Llama-family and Ministral-style dense GQA models — fastest where it works. Rejects Gemma 4 because of `global_head_dim=512` (needs FA4, not available on SM120).
- **FLASHINFER** is the FP8-KV-respecting fallback for Llama/Ministral; ~5% slower than FA at 100k. Rejects Gemma 4 (kernel-template miss in this flashinfer version).
- **TRITON_ATTN** is the only backend that runs Gemma 4 here. On standard GQA models it's 2× slower than FA.
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
| `cuda-driver` | _(not recorded — rerun matrix on this GPU to capture)_ |
| `cuda-toolkit` | _(not recorded — rerun matrix on this GPU to capture)_ |

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

## NVIDIA A100 SXM4 80GB (SM80, Ampere)

Default precision: **INT8 W8A8 weights + BF16 KV cache**. Last measured **2026-05-23**.

| Package | Version |
|---|---|
| `vllm` | 0.21.0 |
| `flashinfer-python` | 0.6.11.post3 |
| `flashinfer-cubin` | 0.6.11.post3 |
| `triton` | 3.6.0 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `cuda-driver` | _(not recorded — rerun matrix on this GPU to capture)_ |
| `cuda-toolkit` | _(not recorded — rerun matrix on this GPU to capture)_ |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**Version-drift note:** `flashinfer-python` / `flashinfer-cubin` 0.6.11.post3 here matches the B200 section; H100, L40S, and SM120 sections were measured on the older 0.6.8.post1 and should be refreshed before cross-GPU comparisons against this row.

**Precision-drift note:** Ampere has **no native FP8 or FP4 tensor cores** — those paths would fall back to BF16 dequant and defeat the point. INT8 tensor cores have been available since Turing (SM 7.5), so W8A8 INT8 (CompressedTensors `int-quantized`) is the natural quantized baseline. `kv_cache_dtype=auto` resolves to **BF16** here (the model's compute dtype); vLLM's FP8 KV path requires an FP8-capable SM (≥ 8.9).

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E4B (`nunusadmqk/gemma-4-E4B-it-W8A8-INT8-v10-datafree`) | ❌ head_size unsupported¹ | ❌ head_size=512 unsupported | **20923** | ❌ KV-sharing not supported |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8`) | 11463 | **10508** | 40756 | ❌ OOM² |
| Ministral 3-3B Instruct (`Ministral-3-3B-Instruct-2512-w8a8`, locally quantized³) | 13960 | **12758** | 47729 | ❌ OOM² |

**Footnotes**:
- ¹ On Ampere, the vendored FA uses **FlashAttention v2** (FA-cute / FA4 are Hopper+). FA2 rejects `head_dim>256`, so Gemma 4's global-attention layers (`head_dim=512`) trip the head-size check. Same pattern as the L40S section. FLASHINFER rejects for the same reason on this SM.
- ² FLEX_ATTENTION OOMs at 100k on Ampere even with 80 GB headroom — the compile-time block-mask metadata plus BF16 KV cache exceed any reasonable utilization budget. Same pattern as the H100 (80 GB) section. Lower `--gpu_memory_utilization` or shorter context could unblock it; default config does not.
- ³ Ministral 3-3B's published checkpoints are FP8/NVFP4; the Mistral-recommended W8A8 INT8 quantization for this revision was produced locally with `auto-round` against the BF16 base `unsloth/Ministral-3-3B-Instruct-2512` (RTN mode — calibration doesn't affect latency-only benchmarks). Only the language-model linear layers are quantized; the model's vision tower stays BF16.

**Notes**:
- **FLASHINFER is the default on A100** for W8A8 INT8 long-context prefill — fastest on both dense models (~8% over FA on Llama, ~9% on Ministral). vLLM picks `CutlassInt8ScaledMMLinearKernel` for the linear-layer matmul; attention runs in BF16 against the BF16 KV cache.
- **FLASH_ATTN** uses the FA2 codepath on Ampere (FA-cute / FA4 are Hopper+). Works on dense models but trails FlashInfer here; rejects Gemma 4 outright (no `head_dim>256` support in FA2).
- **TRITON_ATTN** is ~3.5–4× slower than FlashInfer/FA — last-resort baseline. It's also the **only working backend for Gemma 4 on Ampere** because no other backend supports `head_dim=512` on SM 8.x.
- **FLEX_ATTENTION** is unusable at 100k on Ampere across the board: rejects Gemma 4's sliding/global KV-sharing, OOMs on the dense models even with 80 GB of HBM. Same pattern as the H100 row.
- **80 GB is comfortable** — only FLEX runs out of memory; all FlashInfer/FA/Triton cells that compile run cleanly with default `--gpu_memory_utilization=0.92`. Multimodal models (e.g. Ministral 3-3B with its BF16 vision tower) need the larger card; on 40 GB they OOM across all backends.
- **INT8 W8A8 vs other quant schemes**: vLLM picks `CutlassInt8ScaledMMLinearKernel` for the matmul; attention runs in BF16 against the BF16 KV cache. INT8 activations are quantized per-token dynamically (the `int-quantized` strategy with `act_group_size=-1`).
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.
