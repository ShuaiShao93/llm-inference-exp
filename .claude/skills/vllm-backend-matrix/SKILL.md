---
name: vllm-backend-matrix
description: Build (or refresh) the vLLM attention backend × model latency matrix for the current GPU at 100k input / 1 output. Use when the user wants to know which backend / precision is best for a model on a given GPU, or when `backend_compatibility.md` is missing data for the current hardware. Updates `backend_compatibility.md` in place.
argument-hint: [--model <hf_id>]... [--backends <NAME>...]
allowed-tools: [Bash, Read, Write, Edit]
---

# vLLM Backend × Precision Matrix Builder

Empirically determine which (backend, precision, KV-cache dtype) combinations work for a given set of models on the current GPU at long-context prefill (100k input, 1 output), then write the result to `backend_compatibility.md` at the repo root.

## Arguments

$ARGUMENTS

Parse optional repeatable `--model <hf_id>` and `--backends <NAME ...>`. Defaults are read from `backend_compatibility.md`'s current model set (so re-running the skill regenerates the existing tables). If the file has no entry for the current GPU, prompt the user for the model set if they didn't pass `--model`.

## Key Caveats (Read First)

- **Never run two benchmark processes at the same time** (CLAUDE.md rule). The helper script `scripts/bench_vllm_backends.py` enforces this — don't bypass it.
- **First check `backend_compatibility.md`** before running anything. If the current GPU already has a table and the user didn't ask to refresh, point them at it.
- **Run on a quiet GPU** — vllm grabs ~90% of memory by default; warmup compares badly if another process is contending.
- **Tier ordering matters**: the script picks the most aggressive precision first (FP4 on Blackwell, FP8 on Hopper/Ada) and falls back. Don't override the order unless you have a hardware reason — the matrix records non-default cells in footnotes.
- **The script's `--per_cell_timeout_s` default is 30 minutes** (1800s). Long enough for engine init + 5 runs of a 26B MoE at 100k. If you cut it shorter and a real run gets killed, the cell is reported as a false negative.

## Precision tiers

The script chooses tiers from `nvidia-smi --query-gpu=compute_cap`:

| Compute capability | Tier order tried (precision, kv_cache) |
|---|---|
| SM 8.9 (Ada), SM 9.0 (Hopper) | `(fp8, fp8)` → `(fp8, auto-BF16)` |
| SM 10.0/10.3 (Blackwell datacenter), SM 12.x (Blackwell consumer) | `(fp4, fp8)` → `(fp4, auto)` → `(fp8, fp8)` → `(fp8, auto)` |

First success wins. Any cell that uses a non-default tier is annotated in the matrix footnote.

Override the tier list by editing `detect_precision_tiers()` in the helper script — for instance if a new Blackwell SKU adds FP4-KV FMHA kernels.

## Step 1: Check current state + version drift

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
/usr/bin/python3.12 -c "
import importlib.metadata as md
for p in ['vllm', 'flashinfer-python', 'flashinfer-cubin', 'triton']:
    try:    print(f'{p}: {md.version(p)}')
    except: print(f'{p}: not installed')
"
grep -B1 -A8 'NVIDIA' backend_compatibility.md | head -40
```

Then diff the live versions against the version block in the matching GPU section of `backend_compatibility.md`:

- **No GPU section yet for this hardware** → also check the OTHER GPU sections in the file: if any of them was measured against a *newer* `vllm` / `flashinfer-python` / `flashinfer-cubin` / `triton` than what's currently installed, **stop and prompt the user to upgrade first**. A matrix row measured on an older vLLM is misleading next to peers measured on a newer one. Quote the gap:

  > "Existing rows in backend_compatibility.md (H100) were measured on `vllm==0.21.0`. Current env has `vllm==0.20.2`. Recommend upgrading vllm/flashinfer/triton via `pip install --upgrade vllm` (and `pip index versions vllm` to confirm latest) before populating the new GPU section. Otherwise the SM120 row will be on a different vLLM minor version than the H100 row and not comparable."

  If the user confirms upgrade, do that first, then proceed with the sweep (Step 2 onward). Only skip the upgrade if the user explicitly says "use the current installed versions".
- **GPU section exists, all versions match** → the table is current. If the user just wants to know which backend to use, read it off the table and stop. The empirical sweep is expensive (1-2h).
- **GPU section exists, any version differs** → table is stale. Flag the drift to the user and recommend a refresh, naming which packages changed:

  > "The matrix for H100 was measured against `vllm==0.21.0 / flashinfer==0.6.8.post1 / triton==3.6.0`. Current env has `vllm==0.22.0`. Recommend refreshing — kernel autotune defaults often change between vLLM minor versions."

  Proceed with the sweep if the user confirms.

## Step 2: Pick the model set

By default, refresh with the same models already in the file (so cell numbers update with new vLLM / kernel versions). If the user asks to add a model, append it to the args.

Reasonable default set for a "what is the state of vLLM long-context attention" sweep:
- A model with **exotic head_dim** (Gemma 4 E4B — `head_dim=512` globals)
- A canonical **dense GQA** model (Llama 3.2 3B)
- A **different architecture** for variety (Ministral 3-3B)

Add an MoE if you have one cached (Gemma 4 A4B, DeepSeek-V2-Lite, etc.) to surface MoE-specific routing costs.

## Step 3: Run the sweep

```bash
CUDA_HOME=/usr/local/cuda /usr/bin/python3.12 scripts/bench_vllm_backends.py \
  --model prithivMLmods/gemma-4-E4B-it-FP8 \
  --model RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic \
  --model unsloth/Ministral-3-3B-Instruct-2512-FP8 \
  --backends FLASH_ATTN FLASHINFER TRITON_ATTN FLEX_ATTENTION \
  --output /tmp/vllm_backend_matrix.json
```

This runs `N_models × N_backends × tiers-until-one-works` cells. With 4 backends and tiers averaging ~1.5 attempts per cell, expect roughly `4 × 1.5 = 6` runs per model. Each run is one full vLLM engine init + 5 timed iterations at 100k context — usually 2-5 minutes per cell, longer for larger models. A 3-model × 4-backend sweep on H100 is roughly 1-2 hours wall clock.

Per-cell logs land in `/tmp/vllm_backend_matrix_logs/` for failure diagnosis.

## Step 4: Generate the markdown table

The JSON output schema (per cell):
```json
{
  "chosen": {"precision": "fp8", "kv": "fp8"},  // or null on full failure
  "attempts": [
    {"precision": "fp8", "kv": "fp8", "status": "ok", "mean_ms": 3506.0, ...}
  ]
}
```

For each GPU table in `backend_compatibility.md`:

1. **Header**: GPU name + compute capability + default precision (= first tier tried) + last-measured date.
2. **Version block** immediately under the header: a small table of `vllm`, `flashinfer-python`, `flashinfer-cubin`, `triton` versions from the JSON's `versions` field, plus a one-line note that `flash-attn` is vendored in vllm. Future runs of this skill diff against this block to detect drift.
3. **Rows**: models, formatted as ``Friendly name (`hf-id`)``.
4. **Columns**: backends.
5. **Cells**:
    - On success at default tier: `123 ms`
    - On success at a non-default tier: `123 ms¹` with a numbered footnote explaining the precision deviation (e.g. *"FA cute kernel asserts on Q dtype with FP8 KV → uses BF16 KV."*)
    - On failure: `❌ <reason>` (taken from the JSON's first-attempt failure reason)
6. **Bold the fastest non-failing cell per row** to make backend selection obvious at a glance.
7. **Footnotes**: precision deviations + relevant context (PRs, version dependencies, known issues).

Read the existing file to preserve sections for other GPUs and the "How to read" preamble.

## Step 5: Commit the matrix update

Show the diff to the user before committing. If they confirm:

```bash
git add backend_compatibility.md
git -c user.name=... -c user.email=... commit -m "Refresh backend compatibility matrix on <GPU>"
```

Per CLAUDE.md: do not commit unless explicitly asked.

## Failure-recovery patterns

The script's classifier handles the common cases (head_size, kv_cache_dtype, FlashInfer template missing, FA cute Q-dtype assert, OOM, FlexAttention KV-sharing). If a new failure mode appears, add a new `(regex, label)` to the `patterns` list in `run_one()` rather than letting it fall through to "unknown — see log."

If a cell unexpectedly takes much longer than expected (e.g. 30s for a 4B model where we'd expect ~5s), that's worth a profile dive — note it and follow up with the `profile-llm` skill rather than just recording the number.

## What this skill is *not* for

- **Picking a model** — that's a separate concern (architecture, quality, license). This skill only measures inference performance.
- **Tuning a specific kernel** — use the `profile-llm` skill (nsys → ncu → kernel-config iteration) for that.
- **Multi-GPU / distributed measurements** — single-GPU prefill only.
- **Decode-throughput sweeps** — this skill measures prefill latency at long context, where attention dominates. Decode-heavy workloads have a very different bottleneck profile (memory-bandwidth-bound, KV cache access) and would need a different harness.
