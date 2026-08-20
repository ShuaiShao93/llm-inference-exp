# vLLM Attention Backend Compatibility Matrix

Empirical compatibility and latency data for vLLM attention backends at **10k and 100k input / 1 output** (both prefill-dominated). Refreshed by the `vllm-backend-matrix` skill.

## How to read

- **Rows**: models. **Columns**: backends. One table per GPU.
- **Cells**: `10k / 100k` mean latency in ms — the two input lengths measured. A `—` on either side means that length wasn't measured on this GPU (sections predating the two-length sweep have 100k only).
- A backend that rejects the configuration / OOMs / errors out shows `❌ <reason>` instead of numbers. Failures are almost always identical at both lengths; where they differ, the cell spells out both (e.g. `168 / ❌ OOM`).
- **Default precision** for each GPU is listed under the section header. Cells using a non-default precision/KV-dtype carry a superscript footnote.
- Bold = best backend for that model, bolded **per length** (so a row can have one backend bolded at 10k and a different one at 100k).
- **Why two lengths:** at 100k, attention dominates and the ranking tracks raw kernel quality. At 10k, fixed overheads (engine dispatch, LoRA shrink/expand, GEMM launch tails) are a much larger fraction of wall clock. The best backend is **not always the same at both lengths** — so don't pick a production default from the 100k number alone if your real traffic has short prompts. Any ranking flip is called out in that section's Notes.

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

The adapter is keyed to the **model family**, not to a specific checkpoint: each GPU section benchmarks the checkpoint matching its precision tier (FP8 / NVFP4 / INT8), but the same adapter loads against all of them since LoRA targets the same module names regardless of base quantization.

| Model family | LoRA adapter | r / α | Source |
|---|---|---|---|
| Gemma 4 E4B | `Semaj90/gemma4-e4b-legal-grpo` | 16 / 16 | HF (real). Already excludes `vision_tower.*`, `audio_tower.*`, `multi_modal_projector.*`. |
| Gemma 4 E2B | `tekkaadan/litcoin-gemma-mobile` | 16 / 16 | HF (real). Natively language-model-only — 410 tensors, 24.2M params, no tower weights, so no post-processing needed. |
| Llama 3.2 3B | `~/model_ckpt/synthetic-loras/llama-3.2-3b-r16` | 16 / 16 | Synthetic (built by `scripts/build_synthetic_lora.py`; random weights). |

Local adapter paths under `~/model_ckpt/synthetic-loras/` are **machine-local and not in git** — they won't exist on a freshly provisioned host and must be rebuilt before a sweep (the skill's Step 3 covers this).

The Gemma 4 rows use real HF adapters because PEFT can't currently wrap Gemma 4's `Gemma4ClippableLinear` (raises *"Target module ... is not supported"*) — we can't generate one synthetically. Both pinned Gemma 4 adapters are tower-clean as published, so neither needs post-processing; `scripts/strip_tower_lora.py` stays available for adapters that aren't.

Matching r/α is not sufficient — the adapter's **module scope and per-layer coverage** determine punica cost. Public E2B adapters at r=16/α=16 range from ~2.7M to ~48M params depending on whether they also adapt the vision/audio towers and whether they carry `k_proj`/`v_proj` on every layer (E2B shares KV across layers, so a language-only adapter has k/v on only 15 of 35 layers). Compare tensor and param counts against the target production adapter, not just the config.

To regenerate the synthetic adapter (e.g. when the base model is replaced), run:

```bash
/usr/bin/python3.12 scripts/build_synthetic_lora.py \
  --base RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic \
  --out ~/model_ckpt/synthetic-loras/llama-3.2-3b-r16
```

Synthetic-weight performance is identical to real-weight performance (LoRA dispatch only cares about r/α/target_modules, not the weight values), so this lets us match `r/α` across models without hunting for an HF adapter at every desired rank.

---

## NVIDIA H100 80GB HBM3 (SM90, Hopper)

Default precision: **FP8 W8A8 weights + FP8 KV cache**. Last measured **2026-08-19**; the Gemma 4 E2B row was re-measured **2026-08-20** (new LoRA adapter + the footnote-³ patch).

| Package | Version |
|---|---|
| `vllm` | 0.27.1 |
| `flashinfer-python` | 0.6.16.post3 |
| `flashinfer-cubin` | not installed (cubins fetched at runtime) |
| `triton` | 3.7.1 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `cuda-driver` | 610.43.02 |
| `cuda-toolkit` | 13.3 |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**vLLM local patch in effect for the E2B 100k cells:** the int64 row-index cast from [vllm#53034](https://github.com/vllm-project/vllm/pull/53034), applied to `vllm/lora/ops/triton_ops/kernel_utils.py` in the installed 0.27.1. Without it those cells crash — see footnote ³. Every other cell in this table is stock.

**Version-drift note:** this section and the SM120 section are on the same stack (`vllm 0.27.1` / `flashinfer 0.6.16.post3` / `triton 3.7.1`), so those two are comparable. The L40S, B200, and A100 rows are still on 0.21.0-era stacks and are **not** apples-to-apples until refreshed. `flashinfer-python` 0.6.17 is available but is ahead of vLLM 0.27.1's own pin; refresh both 0.27.1 sections together when adopting it.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`prithivMLmods/gemma-4-E2B-it-FP8`) | **154**¹ / **3181**¹˒³ | ❌ not implemented on SM90⁴ | 238 / 12490³ | ❌ KV sharing not supported⁵ |
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | **223**¹ / **3858**¹ | ❌ not implemented on SM90⁴ | 316 / 13599² | ❌ KV sharing not supported⁵ |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | 158 / **4002** | **154** / ❌ OOM⁶ | 393 / 26508 | 675 / ❌ OOM⁶ |

**Footnotes**:
- ¹ FA on Gemma 4 (`head_dim=512`) rejects FP8 KV on Hopper (`FP8 is only supported on SM100 for FA4 CuTe`), so these cells fall back to **BF16 KV cache** (`fp8 + auto KV`). Every other working cell in the table uses FP8 KV.
- ² TRITON_ATTN on Gemma 4 `head_dim=512` is not tuned for this shape — the `head_dim≥512` tile/warp tuning from [vllm#43257](https://github.com/vllm-project/vllm/pull/43257) is still absent from the bundled `triton_unified_attention.py`, which has been rewritten twice since. Expect ~3.5× FA on E4B at 100k until it is re-upstreamed.
- ³ **These two 100k cells require a patched vLLM and do not run on any release.** On stock 0.27.1, Gemma 4 E2B + LoRA dies at 100k with `an illegal memory access was encountered` on both working backends (10k is fine). It is not an attention bug: with `CUDA_LAUNCH_BLOCKING=1` the faulting launch is vLLM's own LoRA punica kernel (`lora_expand_op.py`), firing during the engine-init profile run before any request is served. It reproduces identically on SM120 — see that section's footnote for the root cause and the exact length boundary. Filed as [vllm#53028](https://github.com/vllm-project/vllm/issues/53028) with a two-line fix in [vllm#53034](https://github.com/vllm-project/vllm/pull/53034), which is **still open and unmerged** as of this measurement; the numbers above were taken with that patch applied to the installed 0.27.1. Re-measure once it lands, and treat any release before then as ❌ for these cells.
- ⁴ FLASHINFER reaches dispatch for `head_dim=512` (the whitelist from [vllm#38822](https://github.com/vllm-project/vllm/pull/38822) is in) but then fails with *not implemented* because the prebuilt cubins for that head_dim target Blackwell SM100+ only. Still unusable for Gemma 4 on Hopper.
- ⁵ FLEX rejects FP8 KV outright, and on its BF16-KV fallback tier it rejects Gemma 4's sliding/global KV sharing. No viable tier for Gemma 4 on this GPU.
- ⁶ FLASHINFER's post-init multi-GB workspace and FLEX's compile-time block-mask metadata both push past 80 GB at 100k on the BF16-KV tier. Lowering `--gpu_memory_utilization` may unblock them; the default config does not.

**Notes**:
- All cells are measured with a LoRA adapter loaded (r=16, α=16, 7 standard projection modules). See the "LoRA adapters used for every benchmark" table near the top of this file.
- **FLASH_ATTN is the backend to reach for on Hopper**, at both lengths and on every model it accepts — including Gemma 4, where it now beats TRITON_ATTN by ~3.5× at 100k (it was the only Gemma-capable option in this stack besides Triton). Note it pays a 2× KV-memory cost on Gemma 4 via the BF16-KV fallback.
- **FLASHINFER is competitive only at short context** (marginally ahead of FA on Llama at 10k) and OOMs at 100k on this card. On Hopper it is a fallback, not a default — unlike SM120, where it wins on Llama and E4B.
- TRITON_ATTN is the universal fallback: it runs every shape that runs at all, and it is the only way to keep FP8 KV on Gemma 4 here, but it is 1.5-6.6× slower than FA.
- FLEX_ATTENTION is not viable on Hopper — rejects FP8 KV everywhere, rejects Gemma 4's KV sharing, and OOMs at 100k on its one working model.
- **Gemma 4 E2B at 100k needs a patched vLLM** (footnote ³) on Hopper *and* consumer Blackwell. On any current release, run long-context E2B without LoRA or stay under ~87k tokens.
- **On E2B, FA's ~3.9× lead over TRITON_ATTN at 100k** (3181 vs 12490) is the widest gap in this table — the same `head_dim=512` Triton tuning gap as footnote ², and it costs FA the BF16-KV fallback.
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
| Gemma 4 E2B (`prithivMLmods/gemma-4-E2B-it-FP8`) | _not measured_ | _not measured_ | _not measured_ | _not measured_ |
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | ❌ head_size unsupported¹ | ❌ head_size=512 unsupported | — / **9114** | ❌ KV-sharing not supported² |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | — / **11179³** | — / 12617 | — / 28094³ | ❌ OOM at 100k (BF16 KV) |

**Footnotes**:
- ¹ On Ada, the vendored FA falls back to **FlashAttention v2** (the cute / FA4 path is Hopper+ only). FA2 doesn't support `head_dim=512`, so both the default FP8 KV cell and the BF16-KV fallback fail — unlike H100, where the FA-cute fallback handles Gemma 4 via BF16 KV.
- ² FLEX_ATTENTION rejects FP8 KV; falling back to BF16 KV trips Gemma 4's sliding-window/global KV-sharing path which FlexAttention doesn't support.
- ³ FA's cute kernel-path Q-dtype assert isn't reachable here (FA2 instead of FA4), but FA2 itself rejects FP8 KV for these models → falls back to **BF16 KV cache** for this cell. FLASHINFER's cell uses default FP8 KV.

**Notes**:
- **TRITON_ATTN is the only working backend for Gemma 4** on Ada (head_dim=512 has no FA2 / FlashInfer support).
- For Llama 3.2 3B, FLASH_ATTN (with BF16-KV fallback) is the fastest, with FLASHINFER close behind at default FP8/FP8. TRITON_ATTN is ~2.5× slower.
- **46 GB is tight at 100k context** — this is the smallest card in the file, and a model that fits everywhere else can still OOM here across every backend. When a whole row comes back OOM, drop `--max_model_len`, lower `--gpu_memory_utilization`, or use a shorter context on L40S rather than reading it as a backend incompatibility.
- Gemma 4 E2B has not been measured on this GPU yet — rerun the `vllm-backend-matrix` skill on an L40S to fill the row.
- FLEX_ATTENTION is not viable at 100k on Ada for the same reasons as Hopper (metadata OOM on BF16-KV fallback) plus the FP8 KV rejection.

---

## NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, Consumer Blackwell)

Default precision: **FP4 W4A4 weights + FP8 KV cache**. Last measured **2026-08-19**; the Gemma 4 E2B row was re-measured **2026-08-20** (new LoRA adapter + the footnote-⁴ patch).

| Package | Version |
|---|---|
| `vllm` | 0.27.1 |
| `flashinfer-python` | 0.6.16.post3 |
| `flashinfer-cubin` | not installed (cubins fetched at runtime) |
| `triton` | 3.7.1 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `cuda-driver` | 610.43.02 |
| `cuda-toolkit` | 13.3 |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**vLLM local patch in effect for the E2B 100k cells:** the int64 row-index cast from [vllm#53034](https://github.com/vllm-project/vllm/pull/53034), applied to `vllm/lora/ops/triton_ops/kernel_utils.py` in the installed 0.27.1. Without it those cells crash — see footnote ⁴. Every other cell in this table is stock.

**Version-drift note:** this section and the H100 section are on the same stack (`vllm 0.27.1` / `flashinfer 0.6.16.post3` / `triton 3.7.1`), so those two are comparable. The L40S, B200, and A100 rows are still on 0.21.0-era stacks and are **not** apples-to-apples until refreshed. `flashinfer-python` 0.6.17 is available but is ahead of vLLM 0.27.1's own pin; refresh both 0.27.1 sections together when adopting it.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`Neural-ICE/Gemma-4-E2B-it-NVFP4`) | ❌ kv_cache_dtype² | 165 / 4902³˒⁴ | **163** / **4743**⁴ | ❌ kv_cache_dtype |
| Gemma 4 E4B (`cosmicproc/gemma-4-E4B-it-NVFP4`) | ❌ kv_cache_dtype² | **230** / **5608**³ | 244 / 6730 | ❌ KV sharing not supported |
| Llama 3.2 3B Instruct (local NVFP4, modelopt from `unsloth/Llama-3.2-3B-Instruct`) | 177 / 6844¹ | **174** / **6008** | 257 / 14860 | 461 / ❌ OOM⁵ |

All cells are measured with a LoRA adapter loaded (r=16, α=16, 7 standard projection modules) — see the "LoRA adapters used for every benchmark" table at the top of this file. Loading a LoRA is not free at this context length: against comparable no-LoRA runs, LoRA cost single-digit percent on FA / FLASHINFER and substantially more on Gemma's TRITON_ATTN cell, so don't compare a LoRA cell here against a no-LoRA number elsewhere.

**Footnotes**:
- ¹ FA's cute kernel asserts on Q dtype when KV cache is FP8 on SM120 → falls back to **BF16 KV cache** for this cell (`fp4 + auto KV`). All other working cells use FP8 KV. On the previous stack FA still won at 100k despite the 2× KV memory; it no longer does (see Notes). Watch for vllm/flashinfer updates that fix this assert.
- ² FA doesn't work for Gemma 4 at *any* KV dtype on SM120 because Gemma 4's full-attention layers have `global_head_dim=512` (sliding-attention layers use 256). FA's SM120 build requires FA4 to support `head_size > 256`, which isn't available here. With FP8 KV the `kv_cache_dtype` check fires first; on E4B with BF16 KV the head_size check fires instead.
- ³ These cells OOM at the default `gpu_memory_utilization` and only run at **0.75** — FlashInfer allocates a multi-GB workspace *after* engine init, by which point vLLM has already sized the KV cache to fill the budget. Latency is insensitive to the budget (Llama × FLASHINFER measured 174 / 5999 at 0.75 vs 174 / 6008 at 0.92), so these numbers are comparable to the rest of the table. Both Gemma 4 rows need it: E2B fails at the default budget at *both* lengths, and the failure is a 1 GiB sampling-buffer allocation with ~640 MiB free, identically on the FP8-KV and BF16-KV tiers. Worth knowing that the budget also decides the *precision* the harness lands on: at the default budget E2B's FP8-KV tier OOMs and the run silently falls back to BF16 KV, whereas at 0.75 it keeps FP8 KV. A backend that appears to "reject" a KV dtype here may just be out of memory — check the per-cell log before recording a dtype gate.
- ⁴ **These two 100k cells require a patched vLLM and do not run on any release.** On stock 0.27.1, Gemma 4 E2B runs fine at 10k but dies with `an illegal memory access was encountered` at 100k on **two independent backends**, and identically on H100. Root-caused: it is not an attention bug. Under `CUDA_LAUNCH_BLOCKING=1` the faulting launch is vLLM's own LoRA punica kernel (`lora_expand_op.py`), and it fires inside the engine-init profile run — a bare `LLM(..., enable_lora=True)` with no adapter and no request is enough. It needs `enable_lora=True` *and* a long `max_model_len`. Bisecting the length gives an exact boundary of **87383** tokens — the kernel's token-row index is int32, and `2**31 / 24576 = 87383`, where 24576 is the byte stride of one row of E2B's merged `gate_up` activation. Not a memory-budget artifact — it still crashes at `gpu_memory_utilization 0.75`. Filed as [vllm#53028](https://github.com/vllm-project/vllm/issues/53028); fixed by casting the row index to int64 ([vllm#53034](https://github.com/vllm-project/vllm/pull/53034)), which is **still open and unmerged** as of this measurement. The numbers above were taken with that patch applied to the installed 0.27.1; it costs nothing at lengths that already worked. Re-measure once it lands, and treat any release before then as ❌ for these cells.
- ⁵ FLEX at 100k OOMs on its only viable tier (`fp4 + auto`, i.e. BF16 KV — FLEX refuses FP8 KV) and still OOMs at `gpu_memory_utilization 0.75`.

**Notes**:
- **FLASHINFER is the best backend on SM120 for Llama and E4B**, at both lengths. This is a reversal: on `vllm 0.22.1` / `flashinfer 0.6.12`, FA won on Llama at 100k (6885 vs 7244); on 0.27.1 / 0.6.16.post3 FLASHINFER wins (6008 vs 6844). Re-check this ordering after every flashinfer bump.
- **E2B is the exception: TRITON_ATTN wins there** at both lengths (163 / 4743 vs 165 / 4902), and it does so at the default memory budget while FLASHINFER needs 0.75. The margin is a few percent, so treat them as equivalent on E2B and prefer Triton for the simpler config.
- **FLASHINFER also now runs Gemma 4 E4B**, and beats TRITON_ATTN there (5608 vs 6730 at 100k). The head_size=512 kernel-template miss that blocked it on flashinfer 0.6.12 is fixed. TRITON_ATTN is no longer the only backend that runs Gemma 4 here.
- **FLASH_ATTN** still rejects Gemma 4 entirely (`global_head_dim=512` needs FA4, unavailable on SM120) and is now second-best on dense GQA.
- **TRITON_ATTN** is the universal fallback — it runs everything that runs at all, but is ~2.5× slower than FLASHINFER on dense GQA.
- **FLEX_ATTENTION** is effectively unusable on SM120: it refuses FP8 KV everywhere, rejects Gemma 4 E4B's KV-sharing, and its one working cell (Llama @ 10k) is 2.6× slower than FLASHINFER.
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

**Version-drift note:** this section is on a 0.21.0-era stack and is **older** than the H100 and SM120 sections above (now `vllm 0.27.1` / `flashinfer 0.6.16.post3`) and newer than L40S. Cross-GPU comparisons against those sections are not apples-to-apples until this one is refreshed on B200 hardware. The flashinfer bump recorded here was originally required for `head_dim=512` cubin coverage ([flashinfer#2959](https://github.com/flashinfer-ai/flashinfer/pull/2959)).

**vLLM local patches in effect for these numbers** (both in `vllm/v1/attention/backends/flashinfer.py`):
1. [vllm#38822](https://github.com/vllm-project/vllm/pull/38822) — `head_dim=512` added to `FlashInferBackend.get_supported_head_sizes()`. Unblocks Gemma 4 full-attention layers from reaching the FlashInfer call.
2. uint8 → `torch.float8_e4m3fn` view bridge in `FlashInferImpl.forward` right after `kv_cache_permute = fixed`. vLLM stores FP8 KV with uint8 backing; since flashinfer#2954, the trtllm-gen kernels treat uint8 unambiguously as NVFP4 and raise `kv_cache_sf must be provided for NVFP4 KV cache.` Other vLLM v1 backends already do this view (`triton_attn.py:570-571`, `rocm_attn.py:416-417`, `rocm_aiter_unified_attn.py:204-205`); FlashInfer was the only one missing it. Without this patch, every FLASHINFER cell here (and on any GPU with flashinfer ≥ 0.6.11 and FP8 KV) would fail.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`Neural-ICE/Gemma-4-E2B-it-NVFP4`) | _not measured_ | _not measured_ | _not measured_ | _not measured_ |
| Gemma 4 E4B (`cosmicproc/gemma-4-E4B-it-NVFP4`) | ❌ kv_cache_dtype not supported² | — / **913³** | — / 9781 | ❌ kv_cache_dtype not supported |
| Llama 3.2 3B Instruct (`inference-optimization/Llama-3.2-3B-Instruct-NVFP4`) | — / 1643¹ | — / **1275** | — / 22705 | ❌ kv_cache_dtype not supported |

**Footnotes**:
- ¹ FA's cute (FA4) kernel asserts on Q dtype when KV cache is FP8 on Blackwell → falls back to **BF16 KV cache** (`fp4 + auto KV`). All other working cells use FP8 KV. Same SM120 footnote ¹ pattern — the assert hasn't been fixed in vLLM 0.21.0.
- ² FA doesn't work for Gemma 4 at *any* KV dtype on B200 either: `head_dim=512` plus FP8 KV trips the kv_cache_dtype check; FA4 supports head_size up to 512 on SM100 but the FP8-KV gate fires first. With BF16 KV fallback the head-size check would clear, but FA's KV-sharing handling for Gemma 4's sliding/global mix doesn't apply here — same architectural mismatch as on SM120.
- ³ FLASHINFER on Gemma 4 only works with **both** local vLLM patches above. Without #38822 it fails at backend selection (head_size 512); without the uint8-view bridge it fails at first kernel dispatch with `kv_cache_sf must be provided for NVFP4 KV cache.` The kernel actually used is `fmhaSm100fKernel_QkvE4m3OBfloat16H512HVPerCta256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` (split-CTA, per-CTA-V=256 — see the `benchmark-gemma4` skill for why head_dim=512 attention is ~2.4× the cost of head_dim=256).

**Notes**:
- **FLASHINFER is the default on B200** for every model tested — beats FLASH_ATTN by a wide margin on dense GQA models, and is the only viable option for Gemma 4 (after patches). This is the opposite of SM120, where FA wins on GQA models; the inversion comes from FlashInfer 0.6.11's SM100 trtllm-gen tuning being further along than FA4's SM100 path for these shapes.
- **FLASH_ATTN** still works on dense GQA models but pays the BF16-KV memory tax. Rejects Gemma 4 outright. Same Q-dtype assert as SM120 — watch for vllm 0.22 / flashinfer upgrades.
- **TRITON_ATTN** is 7-17× slower than FLASHINFER on B200 — only useful as a correctness baseline. Works on every model including Gemma 4.
- **FLEX_ATTENTION** is unusable for the same kv_cache_dtype reason as SM120.
- B200 has plenty of HBM (180 GB), so OOM is not a constraint at 100k context for these models, unlike L40S (46 GB).
- Gemma 4 E2B has not been measured on this GPU yet — rerun the `vllm-backend-matrix` skill on a B200 to fill the row.
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

**Version-drift note:** `flashinfer-python` / `flashinfer-cubin` 0.6.11.post3 here matches the B200 section. The H100 and SM120 sections are now on a **newer** stack (`vllm 0.27.1` / `flashinfer 0.6.16.post3`) and L40S on an older one, so this row should be refreshed on A100 hardware before comparing across GPUs.

**Precision-drift note:** Ampere has **no native FP8 or FP4 tensor cores** — those paths would fall back to BF16 dequant and defeat the point. INT8 tensor cores have been available since Turing (SM 7.5), so W8A8 INT8 (CompressedTensors `int-quantized`) is the natural quantized baseline. `kv_cache_dtype=auto` resolves to **BF16** here (the model's compute dtype); vLLM's FP8 KV path requires an FP8-capable SM (≥ 8.9).

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`glenic/gemma-4-E2B-it-W8A8-INT8`) | _not measured_ | _not measured_ | _not measured_ | _not measured_ |
| Gemma 4 E4B (`nunusadmqk/gemma-4-E4B-it-W8A8-INT8-v10-datafree`) | ❌ head_size unsupported¹ | ❌ head_size=512 unsupported | — / **20923** | ❌ KV-sharing not supported |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8`) | — / 11463 | — / **10508** | — / 40756 | ❌ OOM² |

**Footnotes**:
- ¹ On Ampere, the vendored FA uses **FlashAttention v2** (FA-cute / FA4 are Hopper+). FA2 rejects `head_dim>256`, so Gemma 4's global-attention layers (`head_dim=512`) trip the head-size check. Same pattern as the L40S section. FLASHINFER rejects for the same reason on this SM.
- ² FLEX_ATTENTION OOMs at 100k on Ampere even with 80 GB headroom — the compile-time block-mask metadata plus BF16 KV cache exceed any reasonable utilization budget. Same pattern as the H100 (80 GB) section. Lower `--gpu_memory_utilization` or shorter context could unblock it; default config does not.

**Notes**:
- **FLASHINFER is the default on A100** for W8A8 INT8 long-context prefill — modestly ahead of FA on dense models. vLLM picks `CutlassInt8ScaledMMLinearKernel` for the linear-layer matmul; attention runs in BF16 against the BF16 KV cache.
- **FLASH_ATTN** uses the FA2 codepath on Ampere (FA-cute / FA4 are Hopper+). Works on dense models but trails FlashInfer here; rejects Gemma 4 outright (no `head_dim>256` support in FA2).
- **TRITON_ATTN** is ~3.5–4× slower than FlashInfer/FA — last-resort baseline. It's also the **only working backend for Gemma 4 on Ampere** because no other backend supports `head_dim=512` on SM 8.x.
- **FLEX_ATTENTION** is unusable at 100k on Ampere across the board: rejects Gemma 4's sliding/global KV-sharing, OOMs on the dense models even with 80 GB of HBM. Same pattern as the H100 row.
- **80 GB is comfortable** — only FLEX runs out of memory; all FlashInfer/FA/Triton cells that compile run cleanly with default `--gpu_memory_utilization=0.92`. Multimodal checkpoints that keep a BF16 vision tower need the larger card; on 40 GB they OOM across all backends.
- Gemma 4 E2B has not been measured on this GPU yet — rerun the `vllm-backend-matrix` skill on an A100 to fill the row.
- **INT8 W8A8 vs other quant schemes**: vLLM picks `CutlassInt8ScaledMMLinearKernel` for the matmul; attention runs in BF16 against the BF16 KV cache. INT8 activations are quantized per-token dynamically (the `int-quantized` strategy with `act_group_size=-1`).
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.
