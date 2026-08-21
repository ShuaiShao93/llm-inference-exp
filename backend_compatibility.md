# vLLM Attention Backend Compatibility Matrix

Empirical compatibility and latency data for vLLM attention backends at **10k and 100k input / 1 output** (both prefill-dominated). Refreshed by the `vllm-backend-matrix` skill.

## How to read

- **Rows**: models. **Columns**: backends. One table per GPU.
- **GPU sections are ordered newest architecture first** (datacenter Blackwell → consumer Blackwell → Hopper → Ada → Ampere), not by SM number. Insert a new GPU at its generation, not at the end.
- **Cells**: `10k / 100k` mean latency in ms — the two input lengths measured. A `—` on either side means that length wasn't measured on this GPU (sections predating the two-length sweep have 100k only).
- A backend that rejects the configuration / OOMs / errors out shows `❌ <reason>` instead of numbers. Failures are almost always identical at both lengths; where they differ, the cell spells out both (e.g. `168 / ❌ OOM`).
- **Default precision** for each GPU is listed under the section header. Cells using a non-default precision/KV-dtype carry a superscript footnote.
- Bold = best backend for that model, bolded **per length** (so a row can have one backend bolded at 10k and a different one at 100k). When two backends land within run-to-run noise, **both are bolded** — that is a deliberate "no winner here, choose on something other than speed", not a formatting slip.
- **Why two lengths:** at 100k, attention dominates and the ranking tracks raw kernel quality. At 10k, fixed overheads (engine dispatch, LoRA shrink/expand, GEMM launch tails) are a much larger fraction of wall clock. The best backend is **not always the same at both lengths** — so don't pick a production default from the 100k number alone if your real traffic has short prompts. Any ranking flip is called out in that section's Notes.

## When to regenerate this file

Run the `vllm-backend-matrix` skill when any of these change:
- **Pinned package version** moves (each GPU section lists `vllm`, `flashinfer-python`, `flashinfer-cubin`, `triton`, `transformers`, `python`, `cuda-driver`, `cuda-toolkit`). The skill auto-checks current versus pinned and flags drift. `transformers` and `python` are tracked because vLLM's own pins are loose enough to let a fresh install pick up a version that breaks every cell — see the L40S section for both failure modes. Even patch releases of flashinfer / triton routinely change attention autotune defaults; CUDA driver/toolkit bumps can change JIT-compiled kernel codegen and trigger FlashInfer cubin redownloads.
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

Synthetic-weight performance is identical to real-weight performance (LoRA dispatch only cares about r/α/target_modules, not the weight values), so this lets us match `r/α` across models without hunting for an HF adapter at every desired rank. For the same reason the `--base` can be **any** Llama-3.2-3B checkpoint — BF16 or quantized — since only the module dimensions are read; use whichever is already cached rather than downloading the one named above. Building it needs `peft`, which is not a vLLM dependency: install it with `--no-deps` so it can't drag `transformers` back to a version vLLM rejects.

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

**Version-drift note:** this is now the **only** section still on a 0.21.0-era stack — the H100, SM120, A100, and L40S sections are all on `vllm 0.27.1` / `flashinfer 0.6.16.post3`. Cross-GPU comparisons against those sections are not apples-to-apples until this one is refreshed on B200 hardware. The flashinfer bump recorded here was originally required for `head_dim=512` cubin coverage ([flashinfer#2959](https://github.com/flashinfer-ai/flashinfer/pull/2959)).

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

**Version-drift note:** this section, the H100 section, the L40S section, and the A100 section are on the same pip stack (`vllm 0.27.1` / `flashinfer 0.6.16.post3` / `triton 3.7.1`), so those four are comparable — with the caveat that A100 and L40S are on different CUDA driver/toolkit versions. Only the B200 row is still on a 0.21.0-era stack and is **not** apples-to-apples until refreshed. `flashinfer-python` 0.6.17 is available but is ahead of vLLM 0.27.1's own pin; refresh all four 0.27.1 sections together when adopting it.

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

**Version-drift note:** this section, the SM120 section, the L40S section, and the A100 section are on the same pip stack (`vllm 0.27.1` / `flashinfer 0.6.16.post3` / `triton 3.7.1`), so those four are comparable — with the caveat that A100 and L40S are on different CUDA driver/toolkit versions. Only the B200 row is still on a 0.21.0-era stack and is **not** apples-to-apples until refreshed. `flashinfer-python` 0.6.17 is available but is ahead of vLLM 0.27.1's own pin; refresh all four 0.27.1 sections together when adopting it.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`prithivMLmods/gemma-4-E2B-it-FP8`) | **154**¹ / **3181**¹˒³ | ❌ not implemented on SM90⁴ | 238 / 12490³ | ❌ KV sharing not supported⁵ |
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | **223**¹ / **3858**¹ | ❌ not implemented on SM90⁴ | 316 / 13599² | ❌ KV sharing not supported⁵ |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | 158 / **4002** | **154** / ❌ OOM⁶ | 393 / 26508 | 675 / ❌ OOM⁶ |

**Footnotes**:
- ¹ FA on Gemma 4 (`head_dim=512`) rejects FP8 KV on Hopper (`FP8 is only supported on SM100 for FA4 CuTe`), so these cells fall back to **BF16 KV cache** (`fp8 + auto KV`). Every other working cell in the table uses FP8 KV.
- ² TRITON_ATTN on Gemma 4 `head_dim=512` is not tuned for this shape — the `head_dim≥512` tile/warp tuning from [vllm#43257](https://github.com/vllm-project/vllm/pull/43257) is still absent from the bundled `triton_unified_attention.py`, which has been rewritten twice since. Expect ~3.5× FA on E4B at 100k until it is re-upstreamed.
- ³ **These two 100k cells require a patched vLLM and do not run on any release.** On stock 0.27.1, Gemma 4 E2B + LoRA dies at 100k with `an illegal memory access was encountered` on both working backends (10k is fine). It is not an attention bug: with `CUDA_LAUNCH_BLOCKING=1` the faulting launch is vLLM's own LoRA punica kernel (`lora_expand_op.py`), firing during the engine-init profile run before any request is served. It reproduces identically on SM120 — see that section's footnote for the root cause and the exact length boundary. Filed as [vllm#53028](https://github.com/vllm-project/vllm/issues/53028) with a two-line fix in [vllm#53034](https://github.com/vllm-project/vllm/pull/53034), which is **still open and unmerged** as of this measurement; the numbers above were taken with that patch applied to the installed 0.27.1. Re-measure once it lands, and treat any release before then as ❌ for these cells.
- ⁴ FLASHINFER reaches dispatch for `head_dim=512` (the whitelist from [vllm#38822](https://github.com/vllm-project/vllm/pull/38822) is in) but then fails with *not implemented*. Unusable for Gemma 4 on Hopper. **The "Blackwell-only cubins" explanation is now in doubt:** the A100 section runs Gemma 4 under FLASHINFER on the *same* flashinfer 0.6.16.post3, and SM80 is older than SM90 — so head_dim=512 coverage can't simply be SM100+. The likelier discriminator is the **KV dtype**: Hopper defaults to FP8 KV here while Ampere uses BF16 KV, and the FP8-KV head_dim=512 path is the one that needs trtllm-gen cubins. **The L40S refresh supports this** — on Ada, FLASHINFER's FP8-KV tier for Gemma 4 fails inside flashinfer's JIT arch-flag helper with `No supported CUDA architectures found for major versions [10, 11, 12]` (that path is gated to CC 10.x+), while its BF16-KV tier runs fine. That is a KV-dtype-specific arch gate, not a blanket head_dim=512 gate. Still unresolved for Hopper specifically, since the failure mode there is *not implemented* rather than the arch-flag error — re-test FLASHINFER on H100 with a forced BF16 KV cache on the next refresh before repeating either claim.
- ⁵ FLEX rejects FP8 KV outright, and on its BF16-KV fallback tier it rejects Gemma 4's sliding/global KV sharing. No viable tier for Gemma 4 on this GPU.
- ⁶ FLASHINFER's post-init multi-GB workspace and FLEX's compile-time block-mask metadata both push past 80 GB at 100k on the BF16-KV tier. The default budget sizes the KV cache to fill the card, leaving no room for those workspaces. **The L40S section shows `--gpu_memory_utilization 0.75` recovers the FLASHINFER cell** (on a smaller card, no less) while leaving FLEX OOMed, and moves already-working cells by under 2% — so this FLASHINFER cell is likely a budget artifact rather than a real limit, and deserves a 0.75 retest on the next H100 refresh.

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

Default precision: **FP8 W8A8 weights + FP8 KV cache**. Last measured **2026-08-21**.

**Every cell in this section is measured at `--gpu_memory_utilization 0.75`, not the default.** This is the only section that departs from the default budget, and it is deliberate: on 46 GB the default budget sizes the KV cache so aggressively that it starves the backends' own workspaces, and 6 of the 7 cells that OOM at the default run fine at 0.75 (only FLEX at 100k still fails). Measuring at the default here would populate the table with ❌ cells that describe the budget rather than the backend. Working cells are unaffected by the change — re-measuring the cells that already passed shifted them by under 2% (e.g. Llama TRITON_ATTN at 100k: 24903 ms at default vs 25153 ms at 0.75), so these numbers stay comparable with the other sections.

| Package | Version |
|---|---|
| `vllm` | 0.27.1 |
| `flashinfer-python` | 0.6.16.post3 |
| `flashinfer-cubin` | not installed (cubins fetched at runtime) |
| `triton` | 3.7.1 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `transformers` | 5.14.1 (**pinned — see below**) |
| `python` | 3.12.14 (**3.12+ required — see below**) |
| `cuda-driver` | 610.57.04 |
| `cuda-toolkit` | 13.3 |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**Two environment constraints that vLLM does not enforce for you.** Both were hit on a clean install of this exact stack, and both fail in ways that look like backend incompatibilities:
- **`transformers` must be < 5.15.0.** vLLM 0.27.1 requires only `transformers>=5.5.3`, so a fresh install resolves to the newest release. 5.15.0 made `config.head_dim` raise `AmbiguousGlobalPerLayerAttributeError` for models with heterogeneous per-layer configs, which is exactly what Gemma 4 has, and vLLM reads it unguarded in `transformers_utils/model_arch_config_convertor.get_head_size()`. Every Gemma 4 cell dies during config validation, before a backend is ever selected — on any GPU.
- **Python must be 3.12+.** `flashinfer` 0.6.16.post3 annotates a return type with `array.array[int]` in `comm/fd_exchange.py`, and `array.array` only became subscriptable in 3.12. The annotation is evaluated at import time, so on 3.10/3.11 `import flashinfer.comm` raises `TypeError: 'type' object is not subscriptable`. vLLM imports it from its compilation pass manager regardless of which backend you chose, so *every* model fails at load with a stack trace that names `flashinfer` even for a FLASH_ATTN run.

**vLLM local patch in effect for the E2B 100k cells:** the int64 row-index cast from [vllm#53034](https://github.com/vllm-project/vllm/pull/53034), applied to `vllm/lora/ops/triton_ops/kernel_utils.py` in the installed 0.27.1. Without it those cells crash — see footnote ⁴. Every other cell in this table is stock.

**Version-drift note:** this section is now on the same pip stack as the H100, SM120, and A100 sections (`vllm 0.27.1` / `flashinfer 0.6.16.post3` / `triton 3.7.1`), so all four are comparable; the CUDA driver differs slightly across them (610.57.04 here, 610.43.02 on H100/SM120, 595.71.05 on A100). Only the B200 section is still on a 0.21.0-era stack. Note this host runs Python 3.12.14 via a standalone interpreter rather than the system 3.12 the other hosts use.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`prithivMLmods/gemma-4-E2B-it-FP8`) | ❌ head_size unsupported¹ | **373**³ / **8549**³˒⁴ | **373** / **8514**⁴ | ❌ KV sharing not supported² |
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | ❌ head_size unsupported¹ | **558**³ / **10544**³ | **569** / **10555** | ❌ KV sharing not supported² |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | **402**⁵ / 12224⁵ | **405** / **11012** | 542 / 25153 | 838⁵ / ❌ OOM⁶ |

**Footnotes**:
- ¹ On Ada the vendored FA is **FlashAttention v2** (the cute / FA4 path is Hopper+ only). FA2 rejects Gemma 4's `head_size=512`, and the FP8-KV tier is rejected first on `kv_cache_dtype`, so both tiers fail. Budget-independent — unchanged at 0.75. (vLLM reports `head_size=512` for Gemma 4; the HF per-layer `head_dim` is 256, so don't cross-check these two numbers against each other.)
- ² FLEX_ATTENTION rejects FP8 KV; on its BF16-KV fallback it rejects Gemma 4's sliding-window/global KV-sharing. No viable tier, at any budget.
- ³ FLASHINFER runs Gemma 4 here only on the **BF16-KV** fallback tier. The FP8-KV tier fails in flashinfer's JIT arch-flag helper (`jit/attention/modules.py::_fa2_head_dim_nvcc_flags`) with `No supported CUDA architectures found for major versions [10, 11, 12]` — that path is gated to compute capability 10.x+ (Blackwell and newer), and Ada is 8.9. These cells therefore pay a 2× KV-memory cost relative to the FP8-KV Triton cells beside them. **This is evidence for the open question in the H100 footnote ⁴**: the arch gate applies to the FP8-KV path, not to `head_size=512` in general, which is why A100 (CC 8.0) runs Gemma 4 under FLASHINFER on its BF16-KV tier while Ada and Hopper fail on FP8 KV.
- ⁴ **These two 100k cells require a patched vLLM and do not run on any release** — same int64 punica overflow as the H100 and SM120 sections (boundary at 87383 tokens; see the SM120 footnote for the root cause). Reproduced here on a third GPU and third compute capability, confirming it is a Triton-kernel bug and entirely GPU-independent. Taken with [vllm#53034](https://github.com/vllm-project/vllm/pull/53034) applied; treat any unpatched release as ❌ for these cells.
- ⁵ FA2 rejects FP8 KV for Llama, and FLEX rejects it outright, so these cells fall back to a **BF16 KV cache**. The FLASHINFER and TRITON_ATTN Llama cells use the default FP8 KV.
- ⁶ The only cell that still OOMs at 0.75, so unlike the other OOMs this is a real limit rather than a budget artifact: FLEX's compile-time block-mask metadata plus a BF16 KV cache do not fit at 100k on 46 GB. The A100 section reports the same cell failing the same way at 0.75 on 80 GB.

**Notes**:
- All cells are measured with a LoRA adapter loaded (r=16, α=16, 7 standard projection modules). See the "LoRA adapters used for every benchmark" table near the top of this file.
- **The old "TRITON_ATTN is the only working backend for Gemma 4 on Ada" claim is retired.** It was an artifact of a 0.21.0-era flashinfer plus the default memory budget: FLASHINFER now runs both Gemma 4 models at both lengths, and at 100k the two backends are a **statistical tie** (E4B 10544 vs 10555; E2B 8549 vs 8514 — well inside run-to-run noise). Pick between them on KV memory, not speed: Triton keeps FP8 KV while FlashInfer is forced to BF16 KV (footnote ³), which matters on a 46 GB card.
- **FLASHINFER is the backend to reach for on Llama at 100k** (11012 ms, ~10% ahead of FA's 12224) and ties FA at 10k. This is a **ranking flip** versus the previous 0.21.0-era measurement, where FA led at 100k and FLASHINFER OOMed. TRITON_ATTN is ~2.3× slower at 100k and is a correctness fallback only.
- **46 GB is tight at 100k, but read an OOM as a budget symptom before a backend limit.** That is the single most load-bearing lesson from this refresh: at the default budget this card looked like it had lost FlashInfer support entirely for all three models. Lower `--gpu_memory_utilization` (0.75 works for everything except footnote ⁶) before concluding a backend is unsupported here.
- FLEX_ATTENTION is not viable on Ada — it rejects FP8 KV everywhere, rejects Gemma 4's KV sharing, and is both the slowest option at 10k and the only OOM at 100k.
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM (`SHORT_CONV`, `LINEAR`, `GDN_ATTN`) backends are not applicable to standard dense LLMs on NVIDIA datacenter GPUs.

---

## NVIDIA A100 SXM4 80GB (SM80, Ampere)

Default precision: **INT8 W8A8 weights + BF16 KV cache**. Last measured **2026-08-20**.

| Package | Version |
|---|---|
| `vllm` | 0.27.1 |
| `flashinfer-python` | 0.6.16.post3 |
| `flashinfer-cubin` | not installed (cubins fetched at runtime) |
| `triton` | 3.7.1 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `cuda-driver` | 595.71.05 |
| `cuda-toolkit` | 13.2 |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**Version-drift note:** the pip stack now matches the H100, SM120, and L40S sections exactly, so those four GPUs are directly comparable. This host is the furthest behind on driver/toolkit (`595.71.05` / `13.2` vs `610.43.02` / `13.3` on H100 and SM120, `610.57.04` / `13.3` on L40S), and driver/toolkit bumps can change JIT codegen — so read small cross-GPU deltas with that caveat. `flashinfer-python` 0.6.17 is available but is ahead of vLLM 0.27.1's own pin; refresh all four 0.27.1 sections together when adopting it.

**Precision-drift note:** Ampere has **no native FP8 or FP4 tensor cores** — those paths would fall back to BF16 dequant and defeat the point. INT8 tensor cores have been available since Turing (SM 7.5), so W8A8 INT8 (CompressedTensors `int-quantized`) is the natural quantized baseline. `kv_cache_dtype=auto` resolves to **BF16** here (the model's compute dtype); vLLM's FP8 KV path requires an FP8-capable SM (≥ 8.9). Ampere therefore has a **single precision tier with no fallbacks**, so unlike the Blackwell sections no cell here can silently land on a different precision than the header claims.

**vLLM local patch in effect:** the int64 row-index cast from [vllm#53034](https://github.com/vllm-project/vllm/pull/53034), applied to `vllm/lora/ops/triton_ops/kernel_utils.py` in the installed 0.27.1 — the same patch as the H100 and SM120 sections, and still open and unmerged when measured. The overflow it fixes is in a Triton kernel and is GPU-independent, so the Gemma 4 E2B 100k cells would be expected to crash without it; that was not re-verified here, since the patch was applied before the sweep.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`glenic/gemma-4-E2B-it-W8A8-INT8`) | ❌ head_size unsupported¹ | **351** / **9291**² | 459 / 17945 | ❌ KV sharing not supported³ |
| Gemma 4 E4B (`nunusadmqk/gemma-4-E4B-it-W8A8-INT8-v10-datafree`) | ❌ head_size unsupported¹ | **492** / **10778**² | 614 / 19878 | ❌ KV sharing not supported³ |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8`) | 364 / 12274 | **363**⁴ / **11380**² | 615 / 37643 | 1153 / ❌ OOM⁵ |

**Footnotes**:
- ¹ On Ampere, the vendored FA uses **FlashAttention v2** (FA-cute / FA4 are Hopper+). FA2 rejects `head_dim>256`, so Gemma 4's global-attention layers (`head_dim=512`) trip the head-size check. Same pattern as the L40S section.
- ² Measured at `--gpu_memory_utilization 0.75`. At the default 0.92 **all three of these cells report OOM**: the KV cache expands to 47–51 GiB to fill the budget, and FlashInfer's post-init workspace (1.9–4.6 GiB depending on model) then finds nothing left. This is a budget artifact, not a capability limit — latency is insensitive to the budget once the model fits, so these numbers are comparable to the rest of the table.
- ³ FlexAttention rejects Gemma 4's cross-layer KV sharing — a length-invariant configuration rejection, identical at both lengths.
- ⁴ Within run-to-run noise of FLASH_ATTN's 364 ms; treat 10k on Llama as a tie between the two rather than a FlashInfer win.
- ⁵ FLEX_ATTENTION runs at 10k but OOMs at 100k **even at 0.75**, so unlike the FlashInfer cells this one is a real limit: the compile-time block-mask metadata plus BF16 KV cache don't fit at 100k on 80 GB.

**Notes**:
- **FLASHINFER is the default on A100** — fastest at both lengths for all three models, provided you drop the memory budget to 0.75 at 100k (footnote ²). vLLM picks `CutlassInt8ScaledMMLinearKernel` for the linear-layer matmul; attention runs in BF16 against the BF16 KV cache.
- **"Triton is the only backend that works for Gemma 4 on Ampere" is no longer true.** FLASHINFER now accepts `head_dim=512` on SM80 and is ~1.8–1.9× faster than TRITON_ATTN at 100k on both Gemma models. The durable lesson: **a head-size rejection is a fact about a specific FlashInfer version's cubin coverage, not a property of the SM** — recheck it after every FlashInfer bump instead of treating it as a permanent hardware limit.
- **Gemma 4 under FLASHINFER works on SM80 but not SM90, on identical flashinfer 0.6.16.post3** — an inversion worth understanding before assuming newer hardware is strictly more capable. Since SM80 is the *older* architecture, `head_dim=512` support can't be gated purely on SM version; the difference is most likely the KV dtype (Ampere is forced to BF16 KV, Hopper defaults to FP8 KV, and the FP8-KV head_dim=512 path is the one needing trtllm-gen cubins). Not yet confirmed — see H100 footnote ⁴.
- **The expensive mistake on this GPU is trusting a default-budget OOM.** Taken at face value, the 0.92 sweep says FlashInfer is unusable at 100k and Triton is the only option; the 0.75 retest says FlashInfer is the fastest backend by ~2× on Gemma. A backend that allocates its workspace *after* engine init will always lose a race against a KV cache sized to fill the budget.
- **TRITON_ATTN** is the portable fallback and always runs, but costs 1.8–1.9× (Gemma) to 3.3× (Llama) versus the best backend at 100k. Its 10k penalty is much smaller, so it's a more defensible choice for short-prompt traffic than the 100k column suggests.
- **FLASH_ATTN** uses the FA2 codepath on Ampere. It ties FlashInfer at 10k on the dense model and trails it ~8% at 100k, and rejects both Gemma 4 checkpoints outright.
- **No ranking flip between 10k and 100k on this GPU** — FLASHINFER wins or ties at both lengths for every model (it ties FLASH_ATTN at 10k on Llama). That makes A100 simpler to configure than SM120, where the winner depends on the model.
- **FLEX_ATTENTION diverges by length on the dense model** (1153 ms at 10k, OOM at 100k) and is rejected outright on both Gemma models. It is not a viable long-context backend on Ampere.
- **80 GB is not as roomy as it looks at 100k.** Only TRITON_ATTN and FLASH_ATTN run at the default 0.92; FlashInfer needs 0.75 and FLEX doesn't fit at all. Multimodal checkpoints that keep a BF16 vision tower need the larger card; on 40 GB they OOM across all backends.
- **INT8 W8A8 vs other quant schemes**: vLLM picks `CutlassInt8ScaledMMLinearKernel` for the matmul; attention runs in BF16 against the BF16 KV cache. INT8 activations are quantized per-token dynamically (the `int-quantized` strategy with `act_group_size=-1`).
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.
