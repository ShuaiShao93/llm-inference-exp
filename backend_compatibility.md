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

## When to regenerate

Run the `vllm-backend-matrix` skill — it owns the staleness triggers (version drift, a new GPU or model, a changed checkpoint or adapter, a relevant upstream PR, or any measurement that contradicts a cell). Each GPU section's version block is what the skill diffs against.

## Stack and patches — applies to every section

**All sections are on the same pip stack** (`vllm 0.27.1` / `flashinfer 0.6.16.post3` / `triton 3.7.1`), so every GPU here is comparable. Only the CUDA driver and toolkit differ per host; each section's version block records them. Those pins are vLLM's own — `vllm 0.27.1` requires `flashinfer-python==0.6.16.post3` exactly — not choices we made, and they are never bumped independently.

**One local patch is in effect everywhere:** the int64 row-index cast from [vllm#53034](https://github.com/vllm-project/vllm/pull/53034), applied to `vllm/lora/ops/triton_ops/kernel_utils.py` in the installed 0.27.1. It merged upstream on 2026-08-21, after 0.27.1 shipped, so **no released vLLM carries it**. Without it, Gemma 4 E2B runs fine at 10k and dies at 100k with `an illegal memory access was encountered`: under `CUDA_LAUNCH_BLOCKING=1` the faulting launch is vLLM's own LoRA punica kernel (`lora_expand_op.py`), whose int32 token-row index overflows at exactly **87383** tokens (`2**31 / 24576`, where 24576 is the byte stride of one row of E2B's merged `gate_up` activation). It reproduces on four compute capabilities and on backends that share no attention code, so it is a GPU-independent Triton bug rather than an attention bug — `enable_lora=True` plus a long `max_model_len` is enough to trigger it during engine init, with no adapter and no request, and it is not a memory-budget artifact. Filed as [vllm#53028](https://github.com/vllm-project/vllm/issues/53028). The patch costs nothing at lengths that already worked. **Treat every Gemma 4 E2B 100k cell below as ❌ on any release up to and including 0.27.1**, and drop the patch once a release includes it. Every other cell is stock.

**Two environment constraints vLLM does not enforce**, both of which present as backend incompatibilities rather than env problems: `transformers` must be **< 5.15.0** (5.15.0 makes `config.head_dim` raise on heterogeneous per-layer configs, killing every Gemma 4 cell during config validation before a backend is even selected, on any GPU), and **Python must be 3.12+** (vLLM imports `flashinfer.comm` from its compilation pass manager whatever backend you asked for, and that module only parses on 3.12+ — so every model fails at load with a traceback naming `flashinfer`, even for a FLASH_ATTN run). The `vllm-backend-matrix` skill checks both before sweeping.

**How to read an OOM.** Some cells only run at `--gpu_memory_utilization 0.75` (vLLM's default is 0.92) and say so in a footnote. That is almost always a **budget artifact, not a capability limit**: FLASHINFER and FLEX allocate multi-GB workspaces *after* engine init, by which point vLLM has already sized the KV cache to fill the budget — a race the late allocator always loses. Latency is insensitive to the budget once the model fits (re-measured cells move by under 2%), so 0.75 cells stay comparable to the rest of the file. Two corollaries: **a backend that appears to reject a KV dtype may just be out of memory** — at a tight budget an FP8-KV tier can OOM and the run silently falls back to BF16 KV, so check the per-cell log before recording a dtype gate. And cells that **still** OOM at 0.75 are called out individually as real limits.

## LoRA adapters used for every benchmark

Each model in the matrix is benchmarked with a LoRA adapter loaded, so the cells exercise the backend's LoRA dispatch path (closer to production deployment than a no-LoRA baseline). All adapters are pinned at **rank 16, alpha 16**, targeting the 7 standard projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, `down_proj`) so the LoRA compute shape is identical across models and only the base architecture varies.

The adapter is keyed to the **model family**, not to a specific checkpoint: each GPU section benchmarks the checkpoint matching its precision tier (FP8 / NVFP4 / INT8), but the same adapter loads against all of them since LoRA targets the same module names regardless of base quantization.

| Model family | LoRA adapter | r / α | Source |
|---|---|---|---|
| Gemma 4 E4B | `Semaj90/gemma4-e4b-legal-grpo` | 16 / 16 | HF (real). Excludes `vision_tower.*`, `audio_tower.*`, `multi_modal_projector.*`. 588 tensors, 36.7M params — k/v on **all 42** layers; `use_rslora: True`. |
| Gemma 4 E2B | `tekkaadan/litcoin-gemma-mobile` | 16 / 16 | HF (real). Natively language-model-only — 410 tensors, 24.2M params, no tower weights, so no post-processing needed. k/v on 15 of 35 layers. |
| Llama 3.2 3B | `~/model_ckpt/synthetic-loras/llama-3.2-3b-r16` | 16 / 16 | Synthetic (built by `scripts/build_synthetic_lora.py`; random weights). |

Local adapter paths under `~/model_ckpt/synthetic-loras/` are **machine-local and not in git** — they won't exist on a freshly provisioned host and must be rebuilt before a sweep (the skill's Step 3 covers this).

The Gemma 4 rows use real HF adapters because PEFT can't currently wrap Gemma 4's `Gemma4ClippableLinear` — we can't generate one synthetically. Both pinned Gemma 4 adapters are tower-clean as published, so neither needs post-processing.

Two caveats that apply to **every number in this file**:

- **Every cell was measured at `max_lora_rank=64`, which is not vLLM's default of 16.** Padding the rank costs ~1.33× on the LoRA path, so absolute LoRA-on latencies here are pessimistic relative to what a stock user sees. Within-GPU backend comparisons are unaffected — all cells used 64 consistently.
- **A LoRA-on number is not comparable to a LoRA-off number from an earlier epoch.** Rows measured before adapters were introduced are systematically faster for that reason alone. When a GPU appears to have "regressed," check whether the comparison straddles that change before suspecting hardware, driver, or vLLM version.

For what the adapter costs and why, how to choose one, how to build a synthetic one, and how to profile the punica kernels, see the **`lora-cost` skill**.

---

## Infra cost per request

Latency only tells you which GPU is fastest. This section converts it to money, because **the cost ranking is not the latency ranking** — the fastest card in the file is not the cheapest per request, and the second-cheapest per hour is not the cheapest per request either.

```
cost_per_request = latency_seconds × ($/GPU-hour) ÷ 3600
```

On-demand list prices, cheapest mainstream instance exposing each GPU at **2xlarge or larger** (8+ vCPU — an `xlarge` is CPU-starved for prefill-heavy work, so it is not a usable floor even when it is cheaper per hour), AWS us-east-1, checked **2026-08-25**:

| GPU | cheapest on-demand ≥2xlarge | GPUs | vCPU | instance $/hr | **$/GPU-hr** | $/mo @730h |
|---|---|---|---|---|---|---|
| L40S | AWS `g6e.2xlarge` | 1 | 8 | 2.242 | **2.242** | 1,637 |
| RTX PRO 6000 Blackwell SE | AWS `g7e.2xlarge` | 1 | 8 | 3.363 | **3.363** | 2,455 |
| A100 80GB SXM | AWS `p4de.24xlarge` | 8 | 96 | 27.447 | **3.431**† | 2,505 |
| H100 80GB SXM | AWS `p5.4xlarge` | 1 | 16 | 6.880 | **6.880** | 5,022 |
| B200 | AWS `p6-b200.48xlarge` | 8 | 192 | 113.933 | **14.242**† | 10,396 |

GCP is dearer on every one of these: A100 `a2-ultragpu-1g` 5.069, H100 `a3-highgpu-1g` 11.06, B200 `a4-highgpu-8g` 16.11/GPU, RTX PRO 6000 `g4-standard-48` 4.500. GCP does not offer the L40S at all.

Cost per request at each GPU's **best backend** for that model and length:

| GPU | precision | $/GPU-hr | model | best 10k | ¢/req | best 100k | ¢/req |
|---|---|---|---|---|---|---|---|
| **RTX PRO 6000** | FP4 W4A4 | 3.363 | Gemma 4 E2B | 163 ms | **0.0152** | 4743 ms | **0.443** |
| | | | Gemma 4 E4B | 230 ms | 0.0215 | 5608 ms | 0.524 |
| | | | Llama 3.2 3B | 174 ms | 0.0163 | 6008 ms | 0.561 |
| **RTX PRO 6000** | FP8 W8A8 (online) | 3.363 | Gemma 4 E2B | 184 ms | 0.0172 | 4914 ms | 0.459 |
| | | | Gemma 4 E4B | 279 ms | 0.0260 | 6013 ms | 0.562 |
| | | | Llama 3.2 3B | 208 ms | 0.0195 | 6440 ms | 0.602 |
| **L40S** | FP8 W8A8 | 2.242 | Gemma 4 E2B | 373 ms | 0.0232 | 8514 ms | 0.530 |
| | | | Gemma 4 E4B | 558 ms | 0.0348 | 10544 ms | 0.657 |
| | | | Llama 3.2 3B | 402 ms | 0.0250 | 11012 ms | 0.686 |
| **B200**† | FP4 W4A4 | 14.242 | Gemma 4 E2B | 114 ms | 0.0451 | 1394 ms | 0.551 |
| | | | Gemma 4 E4B | 150 ms | 0.0593 | 1708 ms | 0.676 |
| | | | Llama 3.2 3B | 102 ms | 0.0404 | 1763 ms | 0.697 |
| **B200**† | FP8 W8A8 | 14.242 | all three | — not measured‡ | — | — not measured‡ | — |
| **H100** | FP8 W8A8 | 6.880 | Gemma 4 E2B | 154 ms | 0.0294 | 3181 ms | 0.608 |
| | | | Gemma 4 E4B | 223 ms | 0.0426 | 3858 ms | 0.737 |
| | | | Llama 3.2 3B | 154 ms | 0.0294 | 4002 ms | 0.765 |
| **A100 80GB**† | INT8 W8A8 | 3.431 | Gemma 4 E2B | 348 ms | 0.0332 | 9388 ms | 0.895 |
| | | | Gemma 4 E4B | 492 ms | 0.0469 | 10756 ms | 1.025 |
| | | | Llama 3.2 3B | 362 ms | 0.0345 | 11423 ms | 1.089 |

‡ The B200 FP8 tier is **not measured**, so there is no FP8 row for it. Two independent blockers: its spot capacity returned `STOCKOUT` on 50+ consecutive attempts, and on the one occasion the box did come up, preflight found no working CUDA device and a stale vLLM (0.21.0) rather than the 0.27.1 this file is pinned to. Do not interpolate it from the RTX FP4→FP8 ratio — the two cards have different attention-backend availability, and B200's FP4→FP8 gap would be measured on a different FMHA cubin set.

What this says:

- **The RTX PRO 6000 is the cost-optimal card at both context lengths, on every model.** It is neither the fastest nor the cheapest per hour, but it is the only card that is *both* FP4-capable and priced like a workstation GPU. At 100k on E2B: RTX 0.443 ¢, L40S 0.530 (1.20×), B200 0.551 (1.25×), H100 0.608 (1.37×), A100 0.895 (2.02×).
- **And it wins on cost even handicapped to FP8, so this is not an FP4 artifact.** At matched FP8 W8A8, RTX is 0.459 ¢/req at 100k on E2B against the L40S's 0.530 (1.15×) and the H100's 0.608 (1.32×) — and RTX-in-FP8 still beats B200-in-FP4 (0.551). Dropping the RTX to FP8 costs only 4–7% more per request at 100k, far less than the 13–21% it costs at 10k, so the precision choice barely moves the cross-GPU ranking at long context.
- **Pay for a B200 to buy latency, not throughput-per-dollar.** It is 3.4× faster than the RTX at 100k but costs 4.2× more per hour, so it loses on cost while winning decisively on time-to-first-token. It is the right choice only when the 1.4 s response is worth the premium.
- **A100 is the worst cost-per-request at both lengths** and is dominated on every axis by the RTX PRO 6000, which is cheaper per hour *and* ~2× faster. Ampere is not a cost-effective choice for this workload.
- **The gaps narrow as context grows.** The L40S is 1.53× the RTX's cost at 10k but only 1.20× at 100k, because the cheap-slow card claws back ground once latency is dominated by prefill compute rather than fixed overhead. Extrapolating past 100k, the cheapest card keeps closing — but we have not measured beyond 100k, so that is a trend, not a projection.
- **Instance sizing changes the answer**, which is why the ≥2xlarge floor is stated rather than assumed: on the `g6e.xlarge` (4 vCPU) the L40S prices at 1.861 $/GPU-hr and ties the RTX at 100k. That instance is too CPU-starved to trust for prefill-heavy serving, so the honest comparison uses the 2xlarge and the RTX wins outright.

Caveats that materially affect these numbers:

- **These are batch-1, serial, 100%-duty-cycle costs, so they are an upper bound, not a production figure.** The benchmark issues one request at a time and the whole GPU is billed to it. Real serving batches concurrent requests and amortizes the GPU-hour across all of them, which cuts cost per request by roughly the achieved batch size. Treat these as *marginal cost of a dedicated GPU*, and never quote them as what serving actually costs.
- **† A100 (AWS) and B200 have no 1-GPU SKU**, so their $/GPU-hr is the whole box ÷ 8 — a notional floor. Actual minimum spend is $27.45/hr and $113.93/hr respectively, and you only reach the per-GPU figure if all 8 GPUs stay busy. The H100 figure is real: `p5.4xlarge` is exactly `p5.48xlarge`/8.
- **Precision is not matched across every GPU**, so read the FP4 rows as a best-case-per-card comparison rather than an equal-quality one: only Blackwell (B200, RTX PRO 6000) can do FP4 W4A4, H100 and L40S are FP8 W8A8, and A100 is INT8 W8A8 because Ampere has no FP8 or FP4 tensor cores. The RTX FP8 rows exist precisely so there is one matched-precision comparison against the H100 and L40S — and the RTX still wins it.
- **Every cell has a LoRA adapter loaded** at `max_lora_rank=64`, which costs ~33% more on the LoRA path than the rank-16 the adapter needs — so all costs here carry that padding.
- Excludes storage, egress, and idle time. Spot/preemptible pricing is substantially lower but cannot be relied on: B200 spot returned `STOCKOUT` on 50+ consecutive attempts over ~2.5 hours on 2026-08-24.
- Prices are list, single region (AWS us-east-1), and drift. Re-check before quoting.

---

## NVIDIA B200 180GB HBM3e (SM100, Datacenter Blackwell)

Default precision: **FP4 W4A4 weights + FP8 KV cache**. Last measured **2026-08-21**.

| Package | Version |
|---|---|
| `vllm` | 0.27.1 + 1 local patch (see §) |
| `flashinfer-python` | 0.6.16.post3 |
| `flashinfer-cubin` | not installed (cubins fetched at runtime) |
| `triton` | 3.7.1 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `transformers` | 5.14.1 |
| `python` | 3.12.3 |
| `cuda-driver` | 580.126.20 |
| `cuda-toolkit` | 13.0 |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**Both flashinfer patches this section used to carry are now upstream in 0.27.1** and no longer need applying: `head_dim=512` is in `FlashInferBackend.get_supported_head_sizes()` ([vllm#38822](https://github.com/vllm-project/vllm/pull/38822)), and the uint8 → `torch.float8_e4m3fn` view bridge is present in `FlashInferImpl.forward`. Verify both are still there after a vLLM upgrade before trusting the Gemma 4 FLASHINFER cells — without them those cells fail at backend selection and at first kernel dispatch respectively.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`Neural-ICE/Gemma-4-E2B-it-NVFP4`) | ❌ head_dim 512 → FA2¹ | **114** / **1394**² | 184 / 8812 | ❌ kv_cache_dtype³ |
| Gemma 4 E4B (`cosmicproc/gemma-4-E4B-it-NVFP4`) | ❌ head_dim 512 → FA2¹ | **150** / **1708** | 239 / 10739 | ❌ KV sharing³ |
| Llama 3.2 3B Instruct (locally quantized NVFP4⁶) | **102** / 1865 | **103** / **1763** | 302 / 21566 | 571⁴ / 47662⁴˒⁵ |

**Footnotes**:
- ¹ FA4 on Blackwell **rejects `head_size=512` outright — "due to TMEM capacity limits"** — and silently falls back to FA2, which then fails on either KV dtype: with FP8 KV, `FP8 KV cache requires FA3 on SM90 or FA4 on SM100`; with BF16 KV, `FlashAttention forward only supports head dimension at most 256`. So Gemma 4 is unreachable for FLASH_ATTN on B200 at *any* KV dtype, and the root cause is a Blackwell tensor-memory limit rather than a dtype gate. E4B additionally logs `FA4 on Blackwell does not support local attention with head_size=256`, so even its sliding-window layers can't use FA4. This supersedes the older reading that the FP8-KV gate merely fired first.
- ² Requires the int64 punica patch — see *Stack and patches*. Unpatched, this cell does not run on any release.
- ³ FLEX rejects both Gemma 4 models, but for different reasons: E2B fails `kv_cache_dtype not supported` at **both** FP8 and BF16 KV, while E4B fails that check on FP8 KV and then `FlexAttention does not support kv sharing yet.` on BF16 KV.
- ⁴ **BF16 KV cache** (`fp4 + auto`), not the section default FP8 KV — FLEX rejects FP8 KV on every model here.
- ⁵ Measured at `--gpu_memory_utilization 0.75`; every other cell in this table is at the default. At the default this cell OOMs, and **not for lack of capacity**: vLLM reserves 154 GiB of the 178 GiB visible for KV cache, leaving 2.96 GiB free, and FLEX then asks for a single 4.21 GiB workspace block.
- ⁶ Built on the host with `scripts/quantize_trtllm.py --qformat fp4` from `unsloth/Llama-3.2-3B-Instruct` (modelopt `NVFP4_DEFAULT_CFG`, true W4A4 — verified by the presence of per-layer `input_scale` tensors). The `inference-optimization/Llama-3.2-3B-Instruct-NVFP4` checkpoint this row previously used has been deleted from HF. Quantizing the BF16 base locally also makes this row method-identical to the SM120 one instead of depending on a third-party repo.

**Notes**:
- **Not comparable with the previous B200 table.** Those numbers predate the LoRA commit, so they were taken with no adapter loaded, whereas every cell here has one. The apparent slowdowns against the old table are that methodology change, not a regression — do not read them as one.
- **FLASHINFER is the default on B200** for every model tested, and the only viable option for Gemma 4.
- **The FLASH_ATTN gap has closed on dense GQA.** At 10k, FA (102) and FLASHINFER (103) are a statistical tie — the per-run ranges overlap almost exactly (101.0–104.4 vs 101.5–104.4), hence both bolded. At 100k FLASHINFER wins by only ~5.5%, with non-overlapping ranges. The old table's wide FLASHINFER margin no longer holds.
- **FLASH_ATTN now runs with FP8 KV** on dense GQA models; the Blackwell Q-dtype assert that used to force a BF16-KV fallback is gone in 0.27.1, so FA no longer pays the KV memory tax. It still rejects Gemma 4 for the unrelated head-size reason in footnote ¹.
- **TRITON_ATTN** is 6-12× slower than FLASHINFER at 100k (only 1.6-3× at 10k, so the gap widens sharply with context) — a correctness baseline only. Works on every model including Gemma 4.
- **FLEX_ATTENTION** is only usable on Llama here, and at 100k it is ~27× slower than FLASHINFER. Treat it as unavailable in practice.
- **180 GB of HBM does not make the memory budget irrelevant.** The one OOM in this table is a reservation-sizing artifact, not a capacity limit (footnote ⁵) — the same failure mode as on a 46 GB L40S.
- Gemma 4 E2B is measured on this GPU for the first time in this refresh.
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.

---

## NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, Consumer Blackwell)

This GPU is measured at **two weight precisions**, both with **FP8 KV cache**: NVFP4 W4A4 (the default, from a pre-quantized checkpoint) and FP8 W8A8 (quantized online from the BF16 base). KV stays FP8 in every cell of both tiers — that is a fixed choice here, not a variable, so a backend that refuses FP8 KV is recorded as ❌ rather than re-run at BF16 KV. Last measured **2026-08-19**; the Gemma 4 E2B row was re-measured **2026-08-20** (new LoRA adapter + the footnote-⁴ patch); the FP8 tier was added **2026-08-25** and converted from offline checkpoints to online quantization the same day.

### NVFP4 W4A4 tier (default)

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

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`Neural-ICE/Gemma-4-E2B-it-NVFP4`) | ❌ kv_cache_dtype² | 165 / 4902³˒⁴ | **163** / **4743**⁴ | ❌ kv_cache_dtype |
| Gemma 4 E4B (`cosmicproc/gemma-4-E4B-it-NVFP4`) | ❌ kv_cache_dtype² | **230** / **5608**³ | 244 / 6730 | ❌ KV sharing not supported |
| Llama 3.2 3B Instruct (local NVFP4, modelopt from `unsloth/Llama-3.2-3B-Instruct`) | 177 / 6844¹ | **174** / **6008** | 257 / 14860 | 461 / ❌ OOM⁵ |

All cells are measured with a LoRA adapter loaded (r=16, α=16, 7 standard projection modules) — see the "LoRA adapters used for every benchmark" table at the top of this file. Loading a LoRA is not free at this context length: against comparable no-LoRA runs, LoRA cost single-digit percent on FA / FLASHINFER and substantially more on Gemma's TRITON_ATTN cell, so don't compare a LoRA cell here against a no-LoRA number elsewhere.

**Footnotes**:
- ¹ FA's cute kernel asserts on Q dtype when KV cache is FP8 on SM120 → falls back to **BF16 KV cache** for this cell (`fp4 + auto KV`). All other working cells use FP8 KV. On the previous stack FA still won at 100k despite the 2× KV memory; it no longer does (see Notes). Watch for vllm/flashinfer updates that fix this assert.
- ² FA doesn't work for Gemma 4 at *any* KV dtype on SM120 because Gemma 4's full-attention layers have `global_head_dim=512` (sliding-attention layers use 256). FA's SM120 build requires FA4 to support `head_size > 256`, which isn't available here. With FP8 KV the `kv_cache_dtype` check fires first; on E4B with BF16 KV the head_size check fires instead.
- ³ These cells OOM at the default budget and only run at **0.75**. Budget-insensitive latency confirmed here: Llama × FLASHINFER is 174 / 5999 at 0.75 vs 174 / 6008 at 0.92. Both Gemma 4 rows need it — E2B fails at *both* lengths on a 1 GiB sampling-buffer allocation with ~640 MiB free, identically on the FP8-KV and BF16-KV tiers. This section is also where the silent FP8-KV → BF16-KV fallback was found (see *How to read an OOM*): at the default budget E2B loses FP8 KV, at 0.75 it keeps it.
- ⁴ **These two 100k cells require the int64 punica patch and do not run on any release** — see *Stack and patches*. This is the GPU the bug was root-caused on: it hit two independent backends here, and the length boundary was bisected on this host.
- ⁵ FLEX at 100k OOMs on its only viable tier (`fp4 + auto`, i.e. BF16 KV — FLEX refuses FP8 KV) and still OOMs at `gpu_memory_utilization 0.75`.

**Notes**:
- **FLASHINFER is the best backend on SM120 for Llama and E4B**, at both lengths. This is a reversal: on `vllm 0.22.1` / `flashinfer 0.6.12`, FA won on Llama at 100k (6885 vs 7244); on 0.27.1 / 0.6.16.post3 FLASHINFER wins (6008 vs 6844). Re-check this ordering after every flashinfer bump.
- **E2B is the exception: TRITON_ATTN wins there** at both lengths (163 / 4743 vs 165 / 4902), and it does so at the default memory budget while FLASHINFER needs 0.75. The margin is a few percent, so treat them as equivalent on E2B and prefer Triton for the simpler config.
- **FLASHINFER also now runs Gemma 4 E4B**, and beats TRITON_ATTN there (5608 vs 6730 at 100k). The head_size=512 kernel-template miss that blocked it on flashinfer 0.6.12 is fixed. TRITON_ATTN is no longer the only backend that runs Gemma 4 here.
- **FLASH_ATTN** still rejects Gemma 4 entirely (`global_head_dim=512` needs FA4, unavailable on SM120) and is now second-best on dense GQA.
- **TRITON_ATTN** is the universal fallback — it runs everything that runs at all, but is ~2.5× slower than FLASHINFER on dense GQA.
- **FLEX_ATTENTION** is effectively unusable on SM120: it refuses FP8 KV everywhere, rejects Gemma 4 E4B's KV-sharing, and its one working cell (Llama @ 10k) is 2.6× slower than FLASHINFER.
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.

### FP8 W8A8 tier (quantized online)

Same GPU, same host, same stack, same adapters, same `max_lora_rank` — only the weight precision changes. Weights are quantized **at load time from the BF16 base** (`--precision fp8_per_channel`) rather than loaded from a pre-quantized checkpoint, so one base serves this tier on every GPU. Measured **2026-08-25**.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`google/gemma-4-E2B-it`) | ❌ FP8 KV needs FA3/FA4¹ | 186 / 5074² | **184** / **4914** | ❌ kv_cache_dtype¹ |
| Gemma 4 E4B (`google/gemma-4-E4B-it`) | ❌ FP8 KV needs FA3/FA4¹ | **279**² / **6013**² | 291 / 7135 | ❌ kv_cache_dtype¹ |
| Llama 3.2 3B Instruct (`unsloth/Llama-3.2-3B-Instruct`) | ❌ FP8 KV needs FA3/FA4¹ | **208** / **6440** | 291 / 15185 | ❌ kv_cache_dtype¹ |

`fp8_per_channel` is per-output-channel weight scale + dynamic per-token activation — the same recipe as the `compressed-tensors` W8A8 checkpoints this tier previously used, and as llmcompressor's `FP8_DYNAMIC`. That equivalence is measured, not assumed: see footnote ³.

**Footnotes**:
- ¹ Both backends are rejected at backend selection by the **KV dtype gate**, not by anything about the weights: FLASH_ATTN reports `['kv_cache_dtype not supported', 'FP8 KV cache requires FA3 on SM90 or FA4 on SM100']` and FLEX_ATTENTION reports `['kv_cache_dtype not supported']`, identically for all three models and identically to what the offline checkpoints produced. **Switching a tier from offline to online changes nothing about which backends accept FP8 KV** — weight precision and KV dtype are independent gates. Since KV is pinned to FP8 here these are ❌ by policy rather than unmeasurable: FLASH_ATTN does run at BF16 KV, and a spot check on Llama puts it at **211 / 7275** against FLASHINFER's 208 / 6440 at FP8 KV — 1% slower at 10k and 13% slower at 100k, so the pin costs nothing and at long context is clearly the better choice. The 2× KV footprint is what widens that gap with length. Note this makes the FP8 tier's FA column ❌ for *dense GQA too*, whereas the FP4 tier's Llama × FLASH_ATTN cell has a number only because it was allowed to fall back to BF16 KV.
- ² Requires `--gpu_memory_utilization 0.75`; the other cells are at the default. Same post-init-workspace pattern as footnote ³ in the FP4 tier, and slightly worse here because FP8 weights are ~2× the NVFP4 bytes: E4B needs the lower budget at *both* lengths where its FP4 counterpart needed it too, and E2B additionally needs it at 100k where FP4 did not.
- ³ **Online and offline FP8 measure the same, so this tier's history is continuous.** Every cell that was previously measured from a pre-quantized `compressed-tensors` checkpoint was re-measured online at matched recipe; the largest deviation across all six is **1.1%**, and four of six are under 0.6% — noise, not a change in what's being measured.

  | cell | online | offline | Δ |
  |---|---|---|---|
  | E2B × FLASHINFER | 185.9 / 5073.7 | 188 / 5068 | −1.1% / +0.1% |
  | E2B × TRITON_ATTN | 183.8 / 4913.5 | 185 / 4917 | −0.6% / −0.1% |
  | E4B × FLASHINFER | 278.5 / 6012.5 | 279 / 6012 | −0.2% / +0.0% |
  | E4B × TRITON_ATTN | 290.5 / 7135.3 | 292 / 7135 | −0.5% / +0.0% |
  | Llama × FLASHINFER | 208.4 / 6440.4 | 210 / 6376 | −0.8% / +1.0% |
  | Llama × TRITON_ATTN | 291.2 / 15185.1 | 292 / 15180 | −0.3% / +0.0% |

  The FA-at-BF16-KV spot check agrees too: 211.1 / 7275.2 online vs 211.2 / 7228.3 offline. So online rows here stay comparable with the other GPU sections' offline rows until those are converted.
- ⁴ **`fp8_per_tensor` is not measurably faster than `fp8_per_channel`**, so the coarser recipe buys nothing on this card: E2B × TRITON_ATTN 182.8 / 4910.1, E4B × FLASHINFER 277.1 / 6005.7, Llama × FLASHINFER 208.0 / 6423.3 — every delta is under 0.5% and inside run-to-run noise. Per-channel remains the tier default because it matches the checkpoint recipe (footnote ³) at no cost.
- ⁵ **`mxfp8` is unusable on Gemma 4 and slower on Llama.** On both Gemma models it fails at engine init with `AssertionError: Input dtype must be float16 or bfloat16, got torch.float8_e4m3fn`, raised from FlashInfer's `mxfp8_quantize_cute_dsl`. This is not a backend or hardware limit — it reproduces identically on E2B × TRITON_ATTN and E4B × FLASHINFER (two models, two backends), and the trace is the same every time: online quantization replaces the vision tower's `patch_embedder.input_proj`, after which transformers' `modeling_gemma4.py` does `pixel_values.to(self.input_proj.weight.dtype)` — an idiom that assumes weight dtype equals activation dtype, which is false for any quantized linear. The activation arrives as FP8 and the MXFP8 quantize kernel asserts. It fires during the multimodal encoder profile run, so it happens even with `limit_mm_per_prompt={"image": 0}` and a text-only workload. On Llama (no tower) mxfp8 runs but is **+35% at 10k / +12% at 100k** vs per-channel (281.9 / 7241.0), so there is no reason to prefer it here regardless.

**Notes**:
- **FP4 beats FP8 on every comparable cell**, which is the expected result on Blackwell — but the margin depends strongly on context length: **+13–21% at 10k, only +4–7% at 100k** (best-backend to best-backend: E2B 163→184 / 4743→4914, E4B 230→279 / 5608→6013, Llama 174→208 / 6008→6440).
- **The reason the FP4 advantage shrinks with context is structural, and worth remembering when choosing a precision.** Weight precision only accelerates the GEMMs; Q/K/V into the attention kernel are FP8 in both tiers regardless. At 10k the GEMMs are a large share of wall clock so W4A4 shows up strongly, while at 100k attention dominates and both tiers run the same FP8 attention math. **So FP4's benefit is largest exactly where absolute latency is already smallest** — if your traffic is long-context, FP8 costs you only single-digit percent.
- **Backend ranking is unchanged from FP4 to FP8**: TRITON_ATTN edges E2B, FLASHINFER wins E4B and Llama. Precision does not reorder the backends on this GPU, so the FP4 tier's backend guidance carries over.
- **TRITON_ATTN degrades much worse on Llama at 100k** (15185 vs FLASHINFER's 6440, a 2.4× gap) than on the Gemma models — the same shape as the FP4 tier, so this is a backend property rather than a precision effect.
- **This tier needs no checkpoint hunt, but the FP4 tier above still does.** vLLM's online schemes register FP4 and INT8 methods for routed-expert (MoE) weights only, never for dense Linear layers, so on these three dense models an online FP4 or INT8 request quantizes *nothing* and would report BF16 latency under an FP4 label. Probed and refused for all three models at both schemes; the harness hard-errors rather than measuring it. Hence FP4 above stays on pre-quantized checkpoints. Worth stressing that Gemma 4 E2B/E4B *look* like MoE from their config keys — `num_experts` etc. are present but null, with `enable_moe_block: False` — so check the values, not the keys.

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

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`prithivMLmods/gemma-4-E2B-it-FP8`) | **154**¹ / **3181**¹˒³ | ❌ not implemented on SM90⁴ | 238 / 12490³ | ❌ KV sharing not supported⁵ |
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | **223**¹ / **3858**¹ | ❌ not implemented on SM90⁴ | 316 / 13599² | ❌ KV sharing not supported⁵ |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | 158 / **4002** | **154** / ❌ OOM⁶ | 393 / 26508 | 675 / ❌ OOM⁶ |

**Footnotes**:
- ¹ FA on Gemma 4 (`head_dim=512`) rejects FP8 KV on Hopper (`FP8 is only supported on SM100 for FA4 CuTe`), so these cells fall back to **BF16 KV cache** (`fp8 + auto KV`). Every other working cell in the table uses FP8 KV.
- ² TRITON_ATTN on Gemma 4 `head_dim=512` is not tuned for this shape — the `head_dim≥512` tile/warp tuning from [vllm#43257](https://github.com/vllm-project/vllm/pull/43257) is still absent from the bundled `triton_unified_attention.py`, which has been rewritten twice since. Expect ~3.5× FA on E4B at 100k until it is re-upstreamed.
- ³ **These two 100k cells require the int64 punica patch and do not run on any release** — see *Stack and patches*. Both working backends fail identically here without it.
- ⁴ FLASHINFER reaches dispatch for `head_dim=512` (the whitelist from [vllm#38822](https://github.com/vllm-project/vllm/pull/38822) is in) but then fails with *not implemented*. Unusable for Gemma 4 on Hopper. **The "Blackwell-only cubins" explanation is now in doubt:** the A100 section runs Gemma 4 under FLASHINFER on the *same* flashinfer 0.6.16.post3, and SM80 is older than SM90 — so head_dim=512 coverage can't simply be SM100+. The likelier discriminator is the **KV dtype**: Hopper defaults to FP8 KV here while Ampere uses BF16 KV, and the FP8-KV head_dim=512 path is the one that needs trtllm-gen cubins. **The L40S refresh supports this** — on Ada, FLASHINFER's FP8-KV tier for Gemma 4 fails inside flashinfer's JIT arch-flag helper with `No supported CUDA architectures found for major versions [10, 11, 12]` (that path is gated to CC 10.x+), while its BF16-KV tier runs fine. That is a KV-dtype-specific arch gate, not a blanket head_dim=512 gate. Still unresolved for Hopper specifically, since the failure mode there is *not implemented* rather than the arch-flag error — re-test FLASHINFER on H100 with a forced BF16 KV cache on the next refresh before repeating either claim.
- ⁵ FLEX rejects FP8 KV outright, and on its BF16-KV fallback tier it rejects Gemma 4's sliding/global KV sharing. No viable tier for Gemma 4 on this GPU.
- ⁶ FLASHINFER's post-init workspace and FLEX's compile-time block-mask metadata both push past 80 GB at 100k on the BF16-KV tier. **The L40S section recovers the FLASHINFER cell at 0.75** — on a smaller card, no less — while leaving FLEX OOMed, so this FLASHINFER cell is likely a budget artifact and deserves a 0.75 retest on the next H100 refresh.

**Notes**:
- All cells are measured with a LoRA adapter loaded (r=16, α=16, 7 standard projection modules). See the "LoRA adapters used for every benchmark" table near the top of this file.
- **FLASH_ATTN is the backend to reach for on Hopper**, at both lengths and on every model it accepts — including Gemma 4, where it now beats TRITON_ATTN by ~3.5× at 100k (it was the only Gemma-capable option in this stack besides Triton). Note it pays a 2× KV-memory cost on Gemma 4 via the BF16-KV fallback.
- **FLASHINFER is competitive only at short context** (marginally ahead of FA on Llama at 10k) and OOMs at 100k on this card. On Hopper it is a fallback, not a default — unlike SM120, where it wins on Llama and E4B.
- TRITON_ATTN is the universal fallback: it runs every shape that runs at all, and it is the only way to keep FP8 KV on Gemma 4 here, but it is 1.5-6.6× slower than FA.
- FLEX_ATTENTION is not viable on Hopper — rejects FP8 KV everywhere, rejects Gemma 4's KV sharing, and OOMs at 100k on its one working model.
- **Gemma 4 E2B at 100k needs a patched vLLM** (footnote ³) on Hopper *and* consumer Blackwell. On any current release, run long-context E2B without LoRA or stay under ~87k tokens.

**LoRA A/B on this GPU** (2026-08-24, Gemma 4 E2B FP8 / FLASH_ATTN / BF16 KV, stock punica configs, `--max_lora_rank 16`, 3 runs each, run-to-run spread ≤0.3%):

| input tokens | LoRA off | LoRA on | delta | ratio | LoRA share of total |
|---|---|---|---|---|---|
| 10,000 | 93.9 | 137.9 | +44.0 | 1.47× | 32% |
| 100,000 | 2582.2 | 2972.6 | +390.4 | 1.15× | 13% |

Fitting both points gives a small fixed per-forward overhead (~6 ms) and a marginal cost of ~1384 GB/s of LoRA traffic — see the cross-GPU table above, and its warning that these percentages do not transfer between cards.

**These are faster than the matrix row above (154 / 3181) because the matrix ran `max_lora_rank=64`.** A same-session control at r=64 gives 150.4 / 3100.9, closing the gap to +2.4% / +2.6% — ordinary variance on a shared host. See the rank-padding section above: the 4× rank padding alone is worth 1.33× on the LoRA path. **The matrix row is left as measured rather than overwritten**, since every other LoRA cell in this file carries the same r=64 padding and overwriting one row would break internal comparability.
- **On E2B, FA's ~3.9× lead over TRITON_ATTN at 100k** (3181 vs 12490) is the widest gap in this table — the same `head_dim=512` Triton tuning gap as footnote ², and it costs FA the BF16-KV fallback.
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM (`SHORT_CONV`, `LINEAR`, `GDN_ATTN`) backends are not applicable to standard dense LLMs on NVIDIA datacenter GPUs.

---

## NVIDIA L40S 46GB (SM89, Ada)

Default precision: **FP8 W8A8 weights + FP8 KV cache**. Last measured **2026-08-21**.

**Every cell in this section is measured at `--gpu_memory_utilization 0.75`, not the default** — the only section that departs wholesale. On 46 GB, 6 of the 7 cells that OOM at the default run fine at 0.75 (only FLEX at 100k still fails), so measuring at the default would fill the table with ❌ cells describing the budget rather than the backend. Cells that already passed shifted by under 2% (Llama TRITON_ATTN at 100k: 24903 ms at default vs 25153 ms at 0.75).

| Package | Version |
|---|---|
| `vllm` | 0.27.1 |
| `flashinfer-python` | 0.6.16.post3 |
| `flashinfer-cubin` | not installed (cubins fetched at runtime) |
| `triton` | 3.7.1 |
| `flash-attn` | vendored in vllm (tracks vllm version) |
| `transformers` | 5.14.1 (must be < 5.15.0) |
| `python` | 3.12.14 (3.12+ required) |
| `cuda-driver` | 610.57.04 |
| `cuda-toolkit` | 13.3 |

If any of these has a newer release, the table below is likely stale — rerun the `vllm-backend-matrix` skill.

**Host note:** this box runs Python 3.12.14 via a standalone interpreter rather than the system 3.12 the other hosts use. Both environment constraints in *Stack and patches* were first hit here.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`prithivMLmods/gemma-4-E2B-it-FP8`) | ❌ head_size unsupported¹ | **373**³ / **8549**³˒⁴ | **373** / **8514**⁴ | ❌ KV sharing not supported² |
| Gemma 4 E4B (`prithivMLmods/gemma-4-E4B-it-FP8`) | ❌ head_size unsupported¹ | **558**³ / **10544**³ | **569** / **10555** | ❌ KV sharing not supported² |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) | **402**⁵ / 12224⁵ | **405** / **11012** | 542 / 25153 | 838⁵ / ❌ OOM⁶ |

**Footnotes**:
- ¹ On Ada the vendored FA is **FlashAttention v2** (the cute / FA4 path is Hopper+ only). FA2 rejects Gemma 4's `head_size=512`, and the FP8-KV tier is rejected first on `kv_cache_dtype`, so both tiers fail. Budget-independent — unchanged at 0.75. (vLLM reports `head_size=512` for Gemma 4; the HF per-layer `head_dim` is 256, so don't cross-check these two numbers against each other.)
- ² FLEX_ATTENTION rejects FP8 KV; on its BF16-KV fallback it rejects Gemma 4's sliding-window/global KV-sharing. No viable tier, at any budget.
- ³ FLASHINFER runs Gemma 4 here only on the **BF16-KV** fallback tier. The FP8-KV tier fails in flashinfer's JIT arch-flag helper (`jit/attention/modules.py::_fa2_head_dim_nvcc_flags`) with `No supported CUDA architectures found for major versions [10, 11, 12]` — that path is gated to compute capability 10.x+ (Blackwell and newer), and Ada is 8.9. These cells therefore pay a 2× KV-memory cost relative to the FP8-KV Triton cells beside them. **This is evidence for the open question in the H100 footnote ⁴**: the arch gate applies to the FP8-KV path, not to `head_size=512` in general, which is why A100 (CC 8.0) runs Gemma 4 under FLASHINFER on its BF16-KV tier while Ada and Hopper fail on FP8 KV.
- ⁴ **These two 100k cells require the int64 punica patch and do not run on any release** — see *Stack and patches*. This was the third compute capability to reproduce it, which is what established the bug as GPU-independent.
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

Default precision: **INT8 W8A8 weights + BF16 KV cache**. Last measured **2026-08-21**.

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

**A driver + toolkit bump moved nothing measurable here.** This section was re-measured immediately before and after `595.71.05` / `13.2` → `610.43.02` / `13.3` with the pip stack held fixed: every cell landed within ~1% (E4B FLASHINFER 10k and Llama FLEX 10k reproduced to the millisecond), and every rejection and OOM reproduced identically. The durable lesson is that **driver/toolkit drift is a weaker confound than pip-level drift** — when two GPU sections disagree, suspect the vLLM/FlashInfer versions first. Don't read this as license to skip the bump: it buys comparability cheaply, and the conclusion could differ on an architecture whose kernels are newer than Ampere's.

**Precision-drift note:** Ampere has **no native FP8 or FP4 tensor cores** — those paths would fall back to BF16 dequant and defeat the point. INT8 tensor cores have been available since Turing (SM 7.5), so W8A8 INT8 (CompressedTensors `int-quantized`) is the natural quantized baseline. `kv_cache_dtype=auto` resolves to **BF16** here (the model's compute dtype); vLLM's FP8 KV path requires an FP8-capable SM (≥ 8.9). Ampere therefore has a **single precision tier with no fallbacks**, so unlike the Blackwell sections no cell here can silently land on a different precision than the header claims.

**Patch caveat:** the int64 punica patch was applied before this sweep, so the E2B 100k crash was never re-verified on this GPU. The bug is GPU-independent, so those cells would be expected to fail without it.

| Model | FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION |
|---|---|---|---|---|
| Gemma 4 E2B (`glenic/gemma-4-E2B-it-W8A8-INT8`) | ❌ head_size unsupported¹ | **348** / **9388**² | 456 / 17904 | ❌ KV sharing not supported³ |
| Gemma 4 E4B (`nunusadmqk/gemma-4-E4B-it-W8A8-INT8-v10-datafree`) | ❌ head_size unsupported¹ | **492** / **10756**² | 613 / 19863 | ❌ KV sharing not supported³ |
| Llama 3.2 3B Instruct (`RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8`) | 363 / 12407 | **362**⁴ / **11423**² | 614 / 37602 | 1153 / ❌ OOM⁵ |

**Footnotes**:
- ¹ On Ampere, the vendored FA uses **FlashAttention v2** (FA-cute / FA4 are Hopper+). FA2 rejects `head_dim>256`, so Gemma 4's global-attention layers (`head_dim=512`) trip the head-size check. Same pattern as the L40S section.
- ² Measured at `--gpu_memory_utilization 0.75`. At the default **all three of these cells report OOM**: the KV cache expands to 47–51 GiB to fill the budget, and FlashInfer's post-init workspace (1.9–4.6 GiB depending on model) then finds nothing left.
- ³ FlexAttention rejects Gemma 4's cross-layer KV sharing — a length-invariant configuration rejection, identical at both lengths.
- ⁴ Within run-to-run noise of FLASH_ATTN's 363 ms; treat 10k on Llama as a tie between the two rather than a FlashInfer win.
- ⁵ FLEX_ATTENTION runs at 10k but OOMs at 100k **even at 0.75** (re-verified on this stack), so unlike the FlashInfer cells this one is a real limit: the compile-time block-mask metadata plus BF16 KV cache don't fit at 100k on 80 GB.

**Notes**:
- **FLASHINFER is the default on A100** — fastest at both lengths for all three models, provided you drop the memory budget to 0.75 at 100k (footnote ²). vLLM picks `CutlassInt8ScaledMMLinearKernel` for the linear-layer matmul; attention runs in BF16 against the BF16 KV cache.
- **"Triton is the only backend that works for Gemma 4 on Ampere" is no longer true.** FLASHINFER now accepts `head_dim=512` on SM80 and is ~1.8–1.9× faster than TRITON_ATTN at 100k on both Gemma models. The durable lesson: **a head-size rejection is a fact about a specific FlashInfer version's cubin coverage, not a property of the SM** — recheck it after every FlashInfer bump instead of treating it as a permanent hardware limit.
- **Gemma 4 under FLASHINFER works on SM80 but not SM90, on identical flashinfer 0.6.16.post3** — an inversion worth understanding before assuming newer hardware is strictly more capable. Since SM80 is the *older* architecture, `head_dim=512` support can't be gated purely on SM version; the difference is most likely the KV dtype (Ampere is forced to BF16 KV, Hopper defaults to FP8 KV, and the FP8-KV head_dim=512 path is the one needing trtllm-gen cubins). Not yet confirmed — see H100 footnote ⁴.
- **The expensive mistake on this GPU is trusting a default-budget OOM.** Taken at face value, the default-budget sweep says FlashInfer is unusable at 100k and Triton is the only option; the 0.75 retest says FlashInfer is the fastest backend by ~2× on Gemma.
- **TRITON_ATTN** is the portable fallback and always runs, but costs 1.8–1.9× (Gemma) to 3.3× (Llama) versus the best backend at 100k. Its 10k penalty is much smaller, so it's a more defensible choice for short-prompt traffic than the 100k column suggests.
- **FLASH_ATTN** uses the FA2 codepath on Ampere. It ties FlashInfer at 10k on the dense model and trails it ~8% at 100k, and rejects both Gemma 4 checkpoints outright.
- **No ranking flip between 10k and 100k on this GPU** — FLASHINFER wins or ties at both lengths for every model (it ties FLASH_ATTN at 10k on Llama). That makes A100 simpler to configure than SM120, where the winner depends on the model.
- **FLEX_ATTENTION diverges by length on the dense model** (1153 ms at 10k, OOM at 100k) and is rejected outright on both Gemma models. It is not a viable long-context backend on Ampere.
- **80 GB is not as roomy as it looks at 100k.** Only TRITON_ATTN and FLASH_ATTN run at the default 0.92; FlashInfer needs 0.75 and FLEX doesn't fit at all. Multimodal checkpoints that keep a BF16 vision tower need the larger card; on 40 GB they OOM across all backends.
- **INT8 W8A8 vs other quant schemes**: vLLM picks `CutlassInt8ScaledMMLinearKernel` for the matmul; attention runs in BF16 against the BF16 KV cache. INT8 activations are quantized per-token dynamically (the `int-quantized` strategy with `act_group_size=-1`).
- MLA-only backends (`*_MLA`), AMD (`ROCM_*`), Intel XPU, CPU, and hybrid/SSM backends are not applicable here.
