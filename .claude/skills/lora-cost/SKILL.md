---
name: lora-cost
description: Measure and reason about what a LoRA adapter costs at inference time — the fixed-vs-per-token split, punica kernel traffic, `max_lora_rank` padding, and why utilization does not transfer between GPUs. Use when asked how much LoRA slows inference, why a LoRA-on number doesn't match a LoRA-off one, which adapter to benchmark with, or when modelling/profiling the punica shrink/expand kernels.
argument-hint: [--model <hf_id>] [--adapter <path_or_hf_id>] [--gpu <name>]
allowed-tools: [Bash, Read, Write, Edit]
---

# What a LoRA adapter costs at inference

Measured findings plus the method to reproduce them on a new GPU or model. The
per-GPU latency matrix lives in `backend_compatibility.md`; **every cell there
has an adapter loaded**, so the caveats here apply to all of it.

## The one rule that matters most

**Never carry a LoRA utilization percentage from one card to another, and never
project one.** Measure `modelled bytes ÷ A/B delta` on the GPU in front of you,
with `max_lora_rank` pinned to the adapter's true rank. The section below on the
retracted "flat ~500 GB/s" claim is what happens when this rule is broken.

## Adapter selection and pinning

Pin every adapter at **rank 16, alpha 16**, targeting the 7 standard projection
modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`,
`down_proj`) so the LoRA compute shape is identical across models and only the
base architecture varies. The adapter is keyed to the **model family**, not to a
checkpoint: the same adapter loads against FP8 / NVFP4 / INT8 bases alike,
because LoRA targets module names regardless of base quantization.

Constraints worth knowing before hunting for an adapter:

- **PEFT cannot currently wrap Gemma 4's `Gemma4ClippableLinear`** (raises
  *"Target module ... is not supported"*), so a synthetic adapter can't be
  generated for that family — use a real HF one and check it's tower-clean.
- **Adapter scope changes memory and load time, not punica cost.** Public
  adapters at the same r/α can differ by >10× in parameter count depending on
  whether they also adapt vision/audio towers and whether they carry
  `k_proj`/`v_proj` on every layer.
- **Verify coverage from the adapter's tensor list, not its config.** The config
  looks identical whether coverage is dense or sparse. `target_modules` given as
  a regex or as bare projection names is **not** inherently over-broad — PEFT
  matches by suffix against modules that actually exist, so a bare `k_proj`
  cannot conjure LoRA onto a layer that has no k/v.
- `scripts/strip_tower_lora.py` post-processes adapters that aren't tower-clean.

**Parameter count is not what punica costs.** vLLM wraps LoRA-capable layers at
model init from the *base model's* module inventory, before any adapter loads,
and `reset_lora` only zeroes the weights of modules an adapter misses. So a
merged QKV layer runs all three slices on every layer that has k/v projections
whether or not the adapter carries them, and a module the adapter never targets
still pays a full shrink+expand with zero weights.

**Consequence: estimate punica cost from the base model's wrappable-module
inventory, not from the adapter's `target_modules`.** Denser adapter coverage
fills slices that were already being computed, so it is close to free; sparser
coverage saves nothing.

### Building a synthetic adapter

```bash
/usr/bin/python3.12 scripts/build_synthetic_lora.py \
  --base unsloth/Llama-3.2-3B-Instruct \
  --out ~/model_ckpt/synthetic-loras/llama-3.2-3b-r16
```

Synthetic-weight performance is identical to real-weight performance (LoRA
dispatch only cares about r/α/target_modules, not weight values), so this
matches r/α across models without hunting for an HF adapter at every rank. For
the same reason `--base` can be **any** checkpoint of that model — BF16 or
quantized — since only module dimensions are read; use whichever is already
cached. Building it needs `peft`, which is not a vLLM dependency: install with
`--no-deps` so it can't drag `transformers` back to a version vLLM rejects.

Local adapter paths under `~/model_ckpt/synthetic-loras/` are **machine-local
and not in git** — they won't exist on a freshly provisioned host and must be
rebuilt before a sweep.

## What LoRA actually costs

A/B (identical cell, adapter loaded vs omitted), Gemma 4 E2B / FLASHINFER /
INT8 / A100, 5 runs per point:

| input tokens | LoRA off | LoRA on | delta | ratio |
|---|---|---|---|---|
| 1,000 | 37 | 113 | +76 | **3.07×** |
| 10,000 | 238 | 345 | +107 | 1.45× |
| 50,000 | 2247 | 2768 | +521 | 1.23× |
| 100,000 | 8253 | 9395 | +1142 | 1.14× |

The cost has **two components**, and conflating them is the easy mistake:

- A **fixed per-forward overhead** (~65 ms here), independent of prompt length —
  punica launches two small GEMMs per adapted module on every forward pass
  whether the batch is 1k tokens or 100k.
- A **per-token GEMM cost** (~11 ms per 1k tokens here), linear in prompt
  length. Fitting only the ≥10k points gives a near-zero intercept.

Two consequences worth internalizing:

- **LoRA's relative penalty is worst at short context, not long** (3.07× at 1k
  vs 1.14× at 100k), because base prefill grows superlinearly with attention
  while LoRA grows linearly. Judging LoRA overhead from a long-context benchmark
  alone understates it badly for short-prompt traffic.
- **A LoRA-on number is not comparable to a LoRA-off number from an earlier
  epoch.** When a GPU appears to have "regressed," check whether the comparison
  straddles the introduction of adapters before suspecting hardware, driver, or
  vLLM version.

## Where LoRA time goes: traffic is known, cross-GPU scaling is not

Always measure with vLLM's **default** punica kernel configs — the only thing
vLLM ships. Tuned configs exist only as a user-supplied
`VLLM_TUNED_CONFIG_FOLDER` keyed on `torch.cuda.get_device_name()`; nothing
loads them out of the box, so they do not describe what anyone runs. **Do not
sweep punica tile configs unless explicitly asked to.**

Traffic split for Gemma 4 E2B at 100k input, from ncu `dram__bytes.sum` weighted
to one forward pass. Shares are byte counts, so they hold on any GPU:

| | share of LoRA traffic |
|---|---|
| shrink (all shapes) | 26% |
| expand, narrow outputs | 21% |
| expand, widest outputs (`gate_up`) | 53% |

What that traffic achieves, per GPU — same measurement each row, an end-to-end
LoRA-on/off A/B on the same model and adapter, `100k delta ÷ modelled bytes`,
**all matched at `max_lora_rank=64`**:

| GPU | peak BW | 100k delta | achieved | % of peak | other run config |
|---|---|---|---|---|---|
| L40S (Ada) | 864 GB/s | 1078 ms | 513 GB/s | **59%** | |
| A100 80GB SXM (Ampere) | 2039 GB/s | 1142 ms | 484 GB/s | **24%** | FLASHINFER, INT8 |
| H100 SXM (Hopper) | 3350 GB/s | 519 ms | 1065 GB/s | **32%** | FLASH_ATTN, FP8 + BF16 KV |

**Achieved throughput is not monotonic in peak bandwidth, so peak is not the
explanatory variable.** L40S → A100 widens the bus 2.4× and achieved throughput
*falls* 6%; A100 → H100 widens it 1.6× and throughput *jumps 2.2×*. No single
scaling law fits all three, and any extrapolation from them is unsupported.
Backend and base precision also differ across these rows, so the residual spread
isn't attributable to hardware either.

### Why this section carries a retraction

This analysis previously claimed the LoRA path was *flat* at ~500 GB/s
regardless of peak bandwidth, that LoRA's absolute cost was therefore
GPU-independent, and that a B200 would land at ~6% of peak. **The H100 row
falsified all three.** The flat claim rested on the first two rows only, where
L40S and A100 happen to coincide; two points that agree are not a law, and the
third is 2.9× off it. The retracted figures also divided the deltas by an *r=16*
byte model when both runs were at r=64. The B200 projection was that non-law
extended another 2.4×, and it is withdrawn — **not replaced**, because three
non-monotonic points do not support a replacement.

Method lessons from that failure, which is why they are here rather than in a
commit message:

- **Two agreeing points are not a law.** Require a third before generalizing,
  and prefer one that spans a different architecture.
- **Never put a microbenchmark and an end-to-end A/B in the same column.** An
  isolated-kernel A100 microbenchmark (expand 490–590 GB/s, shrink 906–1273
  GB/s) is arithmetically incompatible with the H100 row and must not be used:
  expand is 387 GB of the 100k pass and the whole LoRA delta is 390 ms, so
  expand on Hopper exceeds **990 GB/s** even if shrink were free. Mixing the two
  is what let the flat story stand as long as it did.
- **Divide by the byte model matching the rank you actually ran.**

## `max_lora_rank` padding costs 1.33× on the LoRA path

One H100, same session, **only `max_lora_rank` changed** (Gemma 4 E2B, r=16
adapter, LoRA-off baseline 93.9 / 2582.2 ms):

| `max_lora_rank` | 10k LoRA-on | delta | 100k LoRA-on | delta | achieved | % of peak |
|---|---|---|---|---|---|---|
| 16 (= adapter's true rank) | 137.9 | +44.0 | 2972.6 | +390.4 | 1384 GB/s | 41% |
| 64 | 150.4 | +56.5 | 3100.9 | +518.7 | 1065 GB/s | 32% |

**Padding the rank 4× costs 1.33× on the LoRA path while moving only 1.022× the
bytes.** So this is not a bandwidth effect — it is loop structure. Expand's K
dimension *is* the padded rank against a fixed `block_k=32`, and shrink's N is
the padded rank, so both do 4× the rank-dimension tile work to produce the same
result. Achieved throughput therefore *falls* from 1384 to 1065 GB/s: the same
kernel gets less efficient purely from a capacity knob.

**Adapter rank is not the same as `max_lora_rank`.** vLLM sizes the punica
buffers to `max_lora_rank` and pads. Pass a `max_lora_rank` matching the adapter
whenever the point of the run is LoRA cost.

Note `scripts/vllm_local.py` defaults `--max_lora_rank` to 64 while vLLM's own
default is 16 — so a default run overstates LoRA cost by ~33% relative to stock
vLLM. Check which the numbers you're comparing used before reading a difference
as real.

## The durable conclusions

- **Byte reduction is the lever; how much it buys is a per-GPU empirical
  question.** These kernels sit at 3–13% SM throughput, so they do almost no
  arithmetic — traffic is the only thing to attack.
- **Expand dominates traffic** (~72% modelled, ~74% measured), because
  `add_inputs=True` read-modify-writes the base projection's output, so expand
  pays `2 × output_width` while shrink's output is a rank-sized sliver. That
  share is a byte count off real per-layer shapes, so it holds on any GPU.
- **The per-kernel throughput split is not measured anywhere trustworthy.**
  Expand's share of *bytes* is solid; the claim that shrink is the efficient half
  came from the retracted microbenchmark. Profile it on the GPU in question. One
  structural lever is known from source rather than measurement: shrink re-reads
  its input once per merged slice (~6% of LoRA traffic), because each slice gets
  its own grid row.
- **Model LoRA traffic from real per-layer shapes, never from a uniform reading
  of the config.** Heterogeneous stacks are common — Gemma 4 E2B has a 2× wider
  FFN in later layers and 2× attention width every 5th layer — and vLLM issues
  LoRA for more module groups per layer than the q/k/v/o/gate/up/down set. A
  uniform model undercounted traffic by ~1.7× here, which read out as "34% of
  peak" when the honest figure was 59%. Read widths off the adapter's `lora_B`
  tensors and confirm any modelled byte count against `dram__bytes.sum`.

## Reproducing this on a new GPU or model

| script | what it gives |
|---|---|
| `scripts/analyze_lora_launches.py` | nsys → launches and durations per shape |
| `scripts/analyze_ncu_lora.py` | ncu → bytes per shape, weighted to a pass |
| `scripts/model_lora_traffic.py` | modelled byte floor to compare against |

Procedure: run an end-to-end A/B (adapter loaded vs omitted) at matched
`max_lora_rank`, at **more than one input length** so the fixed and per-token
components separate, then divide the delta by the modelled bytes. The
`optimize-llm` skill covers the general two-level utilization workflow;
`profile-llm` covers getting nsys/ncu captures out of vLLM subprocesses.
