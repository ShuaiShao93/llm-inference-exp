---
name: vllm-backend-matrix
description: Build (or refresh) the vLLM attention backend × model latency matrix for the current GPU at 10k and 100k input / 1 output. Use when the user wants to know which backend / precision is best for a model on a given GPU, or when `backend_compatibility.md` is missing data for the current hardware. Updates `backend_compatibility.md` in place.
argument-hint: [--model <hf_id>]... [--backends <NAME>...] [--input_tokens <N>...]
allowed-tools: [Bash, Read, Write, Edit]
---

# vLLM Backend × Precision Matrix Builder

Empirically determine which (backend, precision, KV-cache dtype) combinations work for a given set of models on the current GPU, then write the result to `backend_compatibility.md` at the repo root.

Each cell is measured at **two input lengths — 10k and 100k — with 1 output token**. Both are prefill-dominated, but they stress different things: at 100k attention dominates and kernel quality decides the winner, while at 10k the fixed overheads (engine dispatch, LoRA shrink/expand, GEMM launch tails) are a much larger share of wall clock. **The best backend is not always the same at both lengths**, which is exactly why both are recorded — a backend chosen only from 100k numbers can be the wrong default for short-prompt production traffic.

## Arguments

$ARGUMENTS

Parse optional repeatable `--model <hf_id>`, `--backends <NAME ...>`, and `--input_tokens <N ...>`. Defaults are read from `backend_compatibility.md`'s current model set and input lengths (so re-running the skill regenerates the existing tables). If the file has no entry for the current GPU, prompt the user for the model set if they didn't pass `--model`.

## Key Caveats (Read First)

- **Never run two benchmark processes at the same time** (CLAUDE.md rule). `scripts/bench_vllm_backends.py` runs its *own* cells strictly one at a time, but it does **not** detect a benchmark someone else already started. Before launching, confirm the GPU is idle yourself:

  ```bash
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  ```

  Empty output means clear to go. If anything is resident, stop and ask — don't kill it.
- **First check `backend_compatibility.md`** before running anything. If the current GPU already has a table and the user didn't ask to refresh, point them at it.
- **Run on a quiet GPU** — vllm grabs ~90% of memory by default; warmup compares badly if another process is contending.
- **Tier ordering matters**: the script picks the most aggressive precision first (FP4 on Blackwell, FP8 on Hopper/Ada) and falls back. Don't override the order unless you have a hardware reason — the matrix records non-default cells in footnotes.
- **The script's `--per_cell_timeout_s` default is 30 minutes** (1800s). Long enough for engine init + several runs of a 26B MoE at 100k. If you cut it shorter and a real run gets killed, the cell is reported as a false negative.
- **A long sweep must survive a dropped connection.** These runs take hours; launch with `nohup ... &` (or tmux) and redirect output to a file rather than holding the foreground. The script persists its JSON incrementally after every cell, so an interrupted sweep can be read for partial results and resumed by re-running with only the missing models/backends.
- **Track and stop the sweep by PID, and get the PID right.** If you launch under `setsid`/`nohup`, a transient parent shell can appear in `pgrep` output ahead of the real Python process — match on the interpreter (`pgrep -f 'python3.12 scripts/bench_vllm_backends'`) and confirm the PID's command line before waiting on it, or you'll "wait" on a process that already exited and conclude the sweep finished instantly. To stop a sweep, kill that PID; **never `pkill -f bench_vllm_backends`** — on a remote host the pattern also matches your own shell command containing that string, so you kill your session instead of (or as well as) the job.
- **An OOM is never a backend verdict — always retest it at a lower memory budget.** vLLM sizes the KV cache to fill `gpu_memory_utilization`, and a backend that allocates its workspace *after* engine init then finds nothing left. FlashInfer is the usual victim: at the default budget it can OOM on a small model on a huge card (the KV cache having eaten ~64 of 96 GiB), which looks identical to a genuine incompatibility. Re-run the cell with `--gpu_memory_utilization 0.75` before recording `❌ OOM`:

  ```bash
  ... --backends FLASHINFER --gpu_memory_utilization 0.75 --output ~/retest.json
  ```

  Latency is largely insensitive to the budget once the model fits (a control cell moved <0.2% between 0.92 and 0.75), so a retest number is comparable to the rest of the table — record it with a footnote naming the budget it needed. Getting this wrong is expensive in the other direction too: it can bury the *fastest* backend for a model behind a fake incompatibility. Retest crashes the same way: if an illegal-memory-access or CUBLAS failure survives a lower budget, you've earned the right to record it as a real kernel bug instead of hedging.
- **A `head_size unsupported` rejection is a fact about the installed FlashInfer version, not about the GPU.** These backends dispatch to a kernel/cubin set that only covers certain head dims, and coverage is added release by release — an exotic head_dim that FLASHINFER rejected on one sweep can be *fastest* on the same card a few FlashInfer versions later. So a `❌ head_size` cell has a shelf life: re-run it after every FlashInfer bump instead of copying it forward, and word the footnote as "rejected at flashinfer X.Y.Z", never "unsupported on this SM". Corollary: if the same model+backend works on an *older* SM than one where it fails, coverage cannot be SM-gated — look for another discriminator (KV dtype is the usual one, since FP8-KV and BF16-KV take different kernel paths) and record the contradiction rather than inventing a hardware explanation.
- **A failure reproduced on two independent backends is a property of the model, not the backends.** When the same signature (illegal memory access, CUBLAS crash) appears at the same input length under, say, both FLASHINFER and TRITON_ATTN, stop looking for a backend explanation — it's the model's shape at that context length. Say so in the footnote; a reader deciding "which backend do I pick" needs to know that switching backends won't help.
- **`kill -0 <pid>` fails with EPERM across users**, so a watcher loop like `while kill -0 $PID; do sleep 60; done` exits instantly — and looks exactly like "the sweep finished" — if you're logged in as a different user than the one that launched it. Confirm ownership (`ps -o user= -p $PID`) before trusting a wait loop.
- **Failures short-circuit across input lengths.** Lengths are swept ascending; a cell that fails at 10k for a length-invariant reason (dtype gate, head_size, backend-selection rejection) is not re-run at 100k — it inherits the reason and is marked `"skipped": true` in the JSON. Length-*sensitive* failures (OOM, CUBLAS / illegal-address crashes, missing FlashInfer cubins, timeouts) always re-run, because they legitimately differ between 10k and 100k. Pass `--no_short_circuit` to force every length.

## Precision tiers

The script chooses tiers from `nvidia-smi --query-gpu=compute_cap`:

| Compute capability | Tier order tried (precision, kv_cache) |
|---|---|
| SM 8.0 / 8.6 (Ampere) | `(int8, auto-BF16)` — single tier, no fallback |
| SM 8.9 (Ada), SM 9.0 (Hopper) | `(fp8_per_channel, fp8)` → `(fp8_per_channel, auto-BF16)` |
| SM 10.0/10.3 (Blackwell datacenter), SM 12.x (Blackwell consumer) | `(fp4, fp8)` → `(fp4, auto)` → `(fp8_per_channel, fp8)` → `(fp8_per_channel, auto)` |

First success wins. Any cell that uses a non-default tier is annotated in the matrix footnote.

Ampere has no native FP8/FP4 tensor cores, so INT8 W8A8 is the only sensible tier and there is **nothing to fall back to**. That makes Ampere sweeps easier to trust: with one tier, no cell can quietly land on a different precision than the section header claims, so the "an apparent dtype rejection is really an OOM" trap below can't silently change the *precision* you measured — only whether the cell ran at all.

### Online vs offline quantization

**The 8-bit tiers quantize online.** Pass an online scheme name to `--precision` and vLLM quantizes a BF16/FP16 base at load time, so one base checkpoint serves every 8-bit tier and there's no hunt for a precision-matched checkpoint per GPU. `fp8_per_channel` is the tier default: per-output-channel weight scale + dynamic per-token activation, the same recipe as llmcompressor's `FP8_DYNAMIC`, so its numbers stay comparable with rows previously measured from those checkpoints. `fp8_per_tensor`, `fp8_per_block` and `mxfp8` are the other schemes that cover dense Linear layers.

**Sub-8-bit still needs a pre-quantized checkpoint, and the failure mode is silent.** The online schemes below 8 bits set only vLLM's `moe` spec, so on a model with no routed experts every Linear layer falls back to unquantized BF16 — the run *succeeds* and reports BF16 latency under an FP4 or INT8 label. `scripts/vllm_local.py` refuses that combination instead of recording it, so FP4 and INT8 tiers still load a matched checkpoint. Re-check after each vLLM bump: what would lift the restriction is a dense entry in vLLM's `_ONLINE_LINEAR_METHODS`, and a config having MoE *keys* is not enough — check that experts are actually enabled, since a family can ship the keys set to null.

**Weight precision is not the only thing a tier pins.** KV-cache dtype is a separate gate that some backends fail regardless of how the weights were quantized, so switching a tier from offline to online changes nothing about which backends accept FP8 KV. Expect the same backends to drop out.

A **pre-quantized checkpoint only supports the precision it was built at** — an NVFP4 checkpoint rejects the `fp8` tiers with *"requested precision 'fp8' does not match model's built-in precision 'fp4'"*. The script learns this from the first such rejection and skips the remaining mismatched tiers (recorded as `skipped — checkpoint is fp4-only`), so the effective tier count on Blackwell is 2 (KV-dtype fallbacks), not 4. This is why the checkpoint must match the GPU's tier (Step 2): a checkpoint at the wrong precision doesn't fall back gracefully, it fails the whole row.

Override the tier list by editing `detect_precision_tiers()` in the helper script — for instance if a new Blackwell SKU adds FP4-KV FMHA kernels.

## Step 1: Check current state + version drift

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
/usr/local/cuda/bin/nvcc --version | tail -3
/usr/bin/python3.12 -c "
import importlib.metadata as md, sys
print(f'python: {sys.version.split()[0]}')
assert sys.version_info >= (3, 12), 'flashinfer needs 3.12+ — see the Python version trap below'
import flashinfer.comm  # must not raise; vLLM imports this for every backend
for p in ['vllm', 'flashinfer-python', 'flashinfer-cubin', 'triton', 'transformers']:
    try:    print(f'{p}: {md.version(p)}')
    except: print(f'{p}: not installed')
"
grep -B1 -A10 'NVIDIA' backend_compatibility.md | head -60
```

Then diff the live versions against the version block in the matching GPU section of `backend_compatibility.md`. The version block tracks:

- pip packages: `vllm`, `flashinfer-python`, `flashinfer-cubin`, `triton`, `transformers`
- interpreter: `python`
- system: `cuda-driver` (from `nvidia-smi --query-gpu=driver_version`), `cuda-toolkit` (from `nvcc --version`)

(`flash-attn` is vendored inside vllm and tracks the vllm version; no separate check.)

`transformers` and `python` are recorded because vLLM's pins are loose enough that a fresh install can differ from the env a section was measured on. `transformers` in particular only has a lower bound, and a newer one can break a model during config validation — *before* backend selection, so it presents as every cell for that model failing on every backend. If a whole model row fails identically across all four backends, check `transformers` against the recorded version before suspecting the GPU.

The concrete ceiling seen so far: **`transformers` must be < 5.15.0** for Gemma 4. 5.15.0 made `config.head_dim` raise `AmbiguousGlobalPerLayerAttributeError` on models with heterogeneous per-layer configs, and vLLM reads it unguarded in `transformers_utils/model_arch_config_convertor.get_head_size()`. Treat this as the pattern rather than the specific version — any model with per-layer config variation is exposed the next time that accessor tightens.

### Which versions to pin

**Bring exactly three things to latest: `vllm`, the NVIDIA driver, and the CUDA toolkit. Let everything else follow vLLM's own dependency resolution** — install `vllm==<latest>` and take whatever `flashinfer-python` / `torch` / `triton` / `transformers` it pulls, rather than pinning or upgrading them independently.

The reason is that vLLM pins the kernel-adjacent packages *exactly* (0.27.1 declares `flashinfer-python==0.6.16.post3`, `torch==2.13.0`), so a "newer" flashinfer is by definition ahead of what vLLM was tested against — and a mismatch there doesn't fail cleanly, it fabricates kernel errors that read like genuine backend incompatibilities. Driver and toolkit are the opposite case: they sit below the whole stack, are not expressed as pip constraints, and materially change JIT codegen, so they should always be current.

Two consequences worth internalising:

- **Never install `flashinfer-cubin`.** It is published on a slower cadence than `flashinfer-python`, and installing it hard-caps the pair at the older cubin release — below vLLM's own exact pin. See "The flashinfer two-package trap" below.
- **Bump driver/toolkit on every host before comparing them.** Sections measured on different driver/toolkit are not strictly comparable even when every pip version matches. Normalise the hosts first, then sweep; otherwise a driver delta shows up as an apparent kernel regression.

Record every resolved version in the section's version block, including the ones you did not choose.

### The Python version trap

**flashinfer requires Python 3.12+, and vLLM will not tell you.** flashinfer's `comm` module uses annotations that only parse on 3.12+ (subscripted stdlib types evaluated at import time), and vLLM imports `flashinfer.comm` unconditionally while building its compilation passes — so on 3.10/3.11 **every model fails at engine init on every backend**, including FLASH_ATTN and TRITON_ATTN runs that never touch flashinfer. The classifier reports `unknown — see log` for all cells, and the traceback names `flashinfer`, which reads like a FLASHINFER-specific problem rather than an interpreter problem. Note vllm's own wheels are `abi3` and install happily on older interpreters, so pip gives no warning.

If the host's default interpreter is older than 3.12, get a standalone one instead of mutating system apt (these hosts hold large TRT-LLM engine caches you don't want to risk):

```bash
uv python install 3.12
"$(uv python find 3.12)" -m venv ~/vllm-venv
```

Then run every command in this skill through `~/vllm-venv/bin/python`. `bench_vllm_backends.py` launches its subprocesses with `sys.executable`, so the venv propagates automatically. Record the interpreter version in the GPU section's version block.

### The PATH trap (build tools invisible to the JIT)

**Driving a venv's python by absolute path does NOT put that venv's `bin` on `PATH`.** flashinfer's JIT shells out to `ninja` and `nvcc` as *subprocesses*, so neither is found — even though `ninja` installs as a vLLM pip dependency into the very venv you're invoking. Every cell then dies at engine init with `FileNotFoundError: 'ninja'`, which the classifier records as `unknown — see log`, so the result JSON reads as a wall of genuine backend incompatibilities. Nothing in the error text mentions `PATH`. Export both before sweeping:

```bash
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
```

More generally: **when a whole sweep fails identically, suspect the environment before the hardware.** Smoke-test a single cell after any env change — one cell costs a minute, a full matrix costs 1-2h.

### The CUDA_HOME trap (`cannot find -lcudart`)

**flashinfer's JIT infers `CUDA_HOME` from wherever `nvcc` sits on `PATH`, and links `-L$CUDA_HOME/lib64 -lcudart`.** If `nvcc` is reachable somewhere other than the toolkit root — e.g. a copy or symlink in `/usr/local/bin` while the toolkit is `/usr/local/cuda` — the inferred root has no `lib64`, and every engine init dies at the *link* step with `/usr/bin/ld: cannot find -lcudart`. The compiles all succeed first, so the log is thousands of lines of ordinary nvcc output before one fatal linker line. It hits **every backend**, including FLASH_ATTN and TRITON_ATTN runs that never call flashinfer, so it presents as the whole GPU being broken. Export the real toolkit root before sweeping:

```bash
export CUDA_HOME=/usr/local/cuda    # the dir that actually contains lib64/libcudart.so
```

Note CUDA 12+/13 keep the real library under `targets/<arch>/lib/`, with `lib64` as a symlink — so a shallow `find` for `libcudart.so` can come back empty and wrongly suggest the toolkit is broken. Check `$CUDA_HOME/lib64/libcudart.so` directly. A failed JIT build also **poisons the cache directory** (`$FLASHINFER_CACHE_DIR/<ver>/<arch>/cached_ops/<op>/`); move that op's dir aside after fixing the env so it rebuilds.

### The flashinfer two-package trap

`flashinfer-cubin` is **optional and not a vLLM dependency** — vLLM only requires `flashinfer-python`. FlashInfer resolves its cubins in this order (`flashinfer/jit/env.py:_get_cubin_dir`):

1. `$FLASHINFER_CUBIN_DIR` if set
2. the `flashinfer-cubin` package **if installed** — this path enforces an **exact** version equality with `flashinfer-python` and raises `RuntimeError` on any mismatch
3. otherwise `$FLASHINFER_CACHE_DIR/cubins`, **fetched on demand at runtime**

Consequences worth internalising before touching these packages:

- **The two packages are published on independent cadences**, and `flashinfer-cubin` lags. So installing it silently caps `flashinfer-python` at the newest cubin release — a ceiling that can sit *below* vLLM's declared pin.
- **A mismatch breaks every backend, not just FLASHINFER.** vLLM imports `flashinfer.comm` while building its compilation passes, so engine init dies before attention backend selection. Symptom is a bare `RuntimeError: Engine core initialization failed` and the classifier reports `unknown — see log` for *every* cell. If a whole sweep fails that way, check these two versions before suspecting the GPU or the model.
- **Prefer no `flashinfer-cubin`** unless you specifically need offline/pinned cubins: with it absent, flashinfer fetches cubins matching its own version, which cannot mismatch. Record it as `not installed (cubins fetched at runtime)` in the version block.
- Don't "fix" a mismatch with `FLASHINFER_DISABLE_VERSION_CHECK=1` for a matrix run — it loads cubins built for a different version and can fabricate `kernel template missing` failures that look like genuine backend incompatibilities.

- **No GPU section yet for this hardware** → also check the OTHER GPU sections in the file: if any of them was measured against a *newer* stack than what's currently installed, **stop and prompt the user to upgrade first**. A matrix row measured on an older vLLM is misleading next to peers measured on a newer one. Quote the gap:

  > "Existing rows in backend_compatibility.md (H100) were measured on `vllm==0.21.0`. Current env has `vllm==0.20.2`. Recommend `pip install --upgrade vllm` (and `pip index versions vllm` to confirm latest) before populating the new GPU section. Otherwise the SM120 row will be on a different vLLM minor version than the H100 row and not comparable."

  Upgrade `vllm` only — flashinfer/torch/triton come along at whatever it pins. **A newer `flashinfer-python` on PyPI is not drift and not a decision to surface**: it is normally ahead of every released vLLM, so there is nothing to ask the user about. Do not report it as an available upgrade.

  If the user confirms upgrade, do that first, then proceed with the sweep (Step 2 onward). Only skip the upgrade if the user explicitly says "use the current installed versions".
- **GPU section exists, all versions match** → the table is current. If the user just wants to know which backend to use, read it off the table and stop. The empirical sweep is expensive (1-2h).
- **GPU section exists, any version differs** → table is stale. Flag the drift to the user and recommend a refresh, naming which packages changed:

  > "The matrix for H100 was measured against `vllm==0.21.0 / flashinfer==0.6.8.post1 / triton==3.6.0 / cuda-driver==595.71.05 / cuda-toolkit==13.2`. Current env has `vllm==0.22.0` and `cuda-driver==610.43.02`. Recommend refreshing — vLLM minor bumps change attention autotune defaults, and CUDA driver/toolkit bumps trigger flashinfer cubin redownloads and can change JIT kernel codegen."

  Proceed with the sweep if the user confirms.

  CUDA driver/toolkit upgrades are the most disruptive: they invalidate the entire flashinfer cubin cache (`~/.local/lib/python3.12/site-packages/flashinfer_cubin/cubins/`) and trigger re-download on next use. Always rerun the matrix on the affected GPU after a driver or toolkit bump, not just after pip upgrades. See the `upgrade-cuda` skill for the upgrade procedure.

Beyond version drift, the other things that make a section stale:

- A **new GPU** (any compute capability not yet in the file), or a **new model architecture** added to the comparison set.
- A model's **quantization checkpoint** changed, or a **LoRA adapter** for one of the test models changed (every cell is benchmarked LoRA-on — see the `lora-cost` skill).
- A **vLLM PR landed touching the backend you care about** — attention kernel tuning, a new backend, a dtype gate change.
- A measurement anywhere **contradicts the current table**.

## Step 2: Pick the model set

By default, refresh with the same models already in the file (so cell numbers update with new vLLM / kernel versions). If the user asks to add a model, append it to the args.

Reasonable default set for a "what is the state of vLLM long-context attention" sweep:
- Two **exotic-head_dim** sizes from the same family to expose hidden_size scaling at fixed `global_head_dim=512` (Gemma 4 E2B and E4B)
- A canonical **dense GQA** model (Llama 3.2 3B)

Add an MoE if you have one cached (Gemma 4 A4B, DeepSeek-V2-Lite, etc.) to surface MoE-specific routing costs.

**For the 8-bit tiers, point `--model` at the BF16 base** and let the online scheme do the quantizing — `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `unsloth/Llama-3.2-3B-Instruct`. One base per model covers every 8-bit tier on every GPU. Prefer an ungated mirror when the canonical repo is gated, so a sweep needs no HF token; check that the mirror's dims (layers, `hidden_size`, `head_dim`, `intermediate_size`) match the original, since that is all the benchmark depends on.

Budget disk for this: a BF16 base is roughly **twice** the FP8 checkpoint it replaces, and the sweep needs all of them resident at once. Check free space against the sum of the bases before launching, or the first cell of the second model dies mid-download hours in.

**For sub-8-bit tiers, pick the checkpoint whose quantization matches the tier** — e.g. Gemma 4 E2B: `Neural-ICE/Gemma-4-E2B-it-NVFP4` (FP4 — Blackwell), `glenic/gemma-4-E2B-it-W8A8-INT8` (INT8 — Ampere). Prefer **modelopt-format** NVFP4 with 4-bit *activations*; many HF checkpoints tagged `NVFP4` are actually weight-only (`nvfp4-pack-quantized` with `input_activations: null`) or mixed-precision, which silently benchmarks a different precision than the tier claims. Verify `config.json` → `quantization_config.config_groups.*.input_activations` before trusting a checkpoint name.

## Step 3: Materialize the LoRA adapters (do this before the sweep)

Every benchmark in this skill runs with a LoRA adapter loaded. All adapters are pinned at **r=16, alpha=16** targeting the 7 standard projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, `down_proj`) so LoRA compute shape is constant across models. The per-model adapter mapping is pinned in the "LoRA adapters used for every benchmark" table at the top of `backend_compatibility.md`. Failing to load the LoRA is a benchmark failure (not silently skipped) so we catch it explicitly.

**The local adapter paths in that table are not guaranteed to exist on the machine you're on.** They live under `~/model_ckpt/synthetic-loras/` — machine-local, not in git, and not present on a freshly provisioned or reimaged host. Check first and rebuild whatever is missing, otherwise every cell for that model fails on adapter load:

```bash
ls ~/model_ckpt/synthetic-loras/
```

Rebuild recipes (see the matrix's LoRA table for which model needs which):

```bash
# Synthetic adapter — any model PEFT can wrap.
/usr/bin/python3.12 scripts/build_synthetic_lora.py \
  --base unsloth/Llama-3.2-3B-Instruct \
  --out ~/model_ckpt/synthetic-loras/llama-3.2-3b-r16
```

Two things bite on a fresh host:

- **`peft` is not a vLLM dependency**, so `build_synthetic_lora.py` fails with `ModuleNotFoundError: No module named 'peft'`. Install it **with `--no-deps`** — peft pins an older `transformers` than recent vLLM needs and a plain install will happily downgrade it out from under the engine. On a PEP 668 host (`error: externally-managed-environment`) add `--user --break-system-packages`, matching how the rest of these site-packages were installed:

  ```bash
  /usr/bin/python3.12 -m pip install --user --break-system-packages --no-deps peft
  ```

- **`--base` can be any checkpoint from the same family** — the script only reads module dimensions to shape the LoRA, and the adapter carries random weights regardless. Point it at whatever base is already in the HF cache (a BF16 one is fine) instead of downloading the precision-matched checkpoint just to build an adapter.

The Gemma 4 rows need nothing built — both pinned adapters are real HF ids that are already language-model-only, so they're passed straight through. If you swap in a Gemma 4 adapter that *was* trained on the full multimodal model (safetensors contains `vision_tower.*` / `audio_tower.*` keys — vLLM rejects those), run it through `scripts/strip_tower_lora.py` first:

```bash
SNAP=$(/usr/bin/python3.12 -c "from huggingface_hub import snapshot_download; print(snapshot_download('<hf-id>'))" | tail -1)
/usr/bin/python3.12 scripts/strip_tower_lora.py --src "$SNAP" --dst ~/model_ckpt/synthetic-loras/<name>
```

A correct strip drops the tower entries and keeps the language ones — the script prints kept/dropped counts, and dropped should be nonzero for a multimodal-trained adapter. Prefer a natively tower-free adapter over stripping: stripping leaves whatever per-layer coverage the source had, which may not match production (see the caveat below).

## Step 4: Run the sweep

Pass each mapping via `--lora MODEL=LORA_HF_ID_OR_LOCAL_PATH`, and both input lengths via `--input_tokens`:

```bash
cd ~/llm-inference-exp
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH   # ninja + nvcc must be findable as subprocesses; see "The PATH trap"
CUDA_HOME=/usr/local/cuda nohup /usr/bin/python3.12 scripts/bench_vllm_backends.py \
  --model google/gemma-4-E2B-it \
  --model google/gemma-4-E4B-it \
  --model unsloth/Llama-3.2-3B-Instruct \
  --backends FLASH_ATTN FLASHINFER TRITON_ATTN FLEX_ATTENTION \
  --input_tokens 10000 100000 \
  --num_runs 3 \
  --lora google/gemma-4-E2B-it=tekkaadan/litcoin-gemma-mobile \
  --lora google/gemma-4-E4B-it=Semaj90/gemma4-e4b-legal-grpo \
  --lora unsloth/Llama-3.2-3B-Instruct=$HOME/model_ckpt/synthetic-loras/llama-3.2-3b-r16 \
  --output ~/vllm_backend_matrix.json > ~/vllm_backend_matrix.out 2>&1 &
```

`--input_tokens 10000 100000` and `--num_runs 3` are the script defaults; they're spelled out above so the invocation is self-documenting. The ids above are the **BF16 bases** used by the online 8-bit tiers; for a sub-8-bit tier swap in the precision-matched checkpoints from Step 2. Poll progress with `tail -f ~/vllm_backend_matrix.out`; per-cell results land in the JSON as they complete.

Write the output **outside `/tmp`**. Preemptible hosts wipe `/tmp` on stop, which throws away a multi-hour sweep that had already completed most of its cells.

If you're adding a new model, first try HF search for an existing adapter at r=16/α=16 (filter: `lora` tag + base model name; check `adapter_config.json` for `target_modules ⊇ {q,k,v,o,up,gate,down}_proj`, no `modules_to_save`, no `use_dora`, no `use_rslora`, no multimodal tower targets). If nothing on HF matches:

- **Generate a synthetic adapter** via `scripts/build_synthetic_lora.py` (random weights, identical compute shape). Works for any model PEFT can wrap.
- For Gemma 4 family (`Gemma4ClippableLinear` blocks PEFT from generating one), pick the cleanest real HF adapter at r=16/α=16. If it was trained on the full multimodal model (`unsloth_fixed=true` / safetensors contains `audio_tower.*` / `vision_tower.*` keys), run it through **`scripts/strip_tower_lora.py`** to drop the tower entries before vLLM will accept it.

**Rank and alpha alone do not pin the LoRA workload.** Punica cost scales with how many tensors the adapter actually carries, so check the safetensors key set and param count, not just `adapter_config.json`. Two things vary widely between adapters at the same r/α: whether the towers are adapted (an order-of-magnitude difference), and per-layer coverage of `k_proj`/`v_proj` — a model with cross-layer KV sharing only *has* k/v on a subset of layers, but a regex-targeted adapter may still emit them for every layer. Pick the adapter whose key set matches the deployment you're modelling; a mismatch silently benchmarks a heavier or lighter LoRA than production.

Smoke-test the resulting adapter with the FP8 base model once, then add the mapping to both the matrix's LoRA table and the sweep invocation.

This runs `N_models × N_backends × N_lengths × tiers-until-one-works` cells. With 4 backends, 2 lengths, and tiers averaging ~1.5 attempts, expect roughly `4 × 2 × 1.5 = 12` runs per model — though short-circuiting on length-invariant failures claws a good chunk of that back, since a backend a model outright rejects costs only its 10k attempt. Each run is one full vLLM engine init + `--num_runs` timed iterations; the 10k cells are much cheaper than the 100k ones. Budget **2-3 hours** wall clock for a 3-model × 4-backend × 2-length sweep, and note that engine init (not the timed iterations) dominates the 10k cells.

Per-cell logs land in `~/vllm_backend_matrix_logs/` for failure diagnosis; the filename encodes model, backend, precision, KV dtype, **input length**, and adapter. The length matters: without it the 100k attempt overwrites the 10k log, so a cell that failed at 10k leaves behind the *wrong* log and reads as `unknown — see log` with no way to diagnose it short of re-running.

## Step 5: Generate the markdown table

The JSON output schema — note cells are keyed **`model → backend → input_tokens`**, with the length as a *string* key:
```json
"cells": {
  "<model>": {
    "FLASH_ATTN": {
      "10000":  {"chosen": {"precision": "fp8", "kv": "fp8"},
                 "attempts": [{"precision": "fp8", "kv": "fp8", "status": "ok", "mean_ms": 412.0}]},
      "100000": {"chosen": null,
                 "attempts": [], "skipped": true, "reason": "head_size unsupported by backend"}
    }
  }
}
```

A cell with `"skipped": true` was never executed — it inherited a length-invariant failure from a shorter length. Render it with the same `❌ <reason>` as its source cell; don't report it as untested.

For each GPU section in `backend_compatibility.md`:

1. **Header**: GPU name + compute capability + default precision (= first tier tried) + last-measured date.
2. **Version block** immediately under the header: a small table of `vllm`, `flashinfer-python`, `flashinfer-cubin`, `triton`, `cuda-driver`, `cuda-toolkit` versions from the JSON's `versions` field, plus `transformers` and `python` from the Step 1 check, plus a one-line note that `flash-attn` is vendored in vllm. Future runs of this skill diff against this block to detect drift. When `flashinfer-cubin` is absent the JSON records `null` — write `not installed (cubins fetched at runtime)`, not a blank.
3. **One table per GPU**, with both lengths packed into each cell as `10k / 100k`. Don't split into per-length tables — the rows and columns are identical, so two tables just doubles the reading effort.
4. **Rows**: models, formatted as ``Friendly name (`hf-id`)``. **Columns**: backends.
5. **Cells** — `<10k> / <100k>`:
    - Both lengths succeeded at the default tier: `168 / 12388`
    - Success at a non-default tier: `168 / 12388¹` with a numbered footnote explaining the precision deviation (e.g. *"FA cute kernel asserts on Q dtype with FP8 KV → uses BF16 KV."*)
    - Both lengths failed the same way (the common case, incl. `"skipped": true` inheriting a length-invariant reason): collapse to a single `❌ <reason>` rather than writing it twice.
    - Lengths **diverged**: spell out both, e.g. `168 / ❌ OOM`. This is the interesting case — it means the failure is length-sensitive, and it deserves a Notes bullet.
    - A length that wasn't measured: `—` on that side (e.g. `— / 12388`).
6. **Bold the fastest non-failing cell per row, independently per length.** A row can legitimately have one backend bolded on the 10k side and a different one on the 100k side — bold only the number that won, not the whole cell.
7. **Footnotes**: precision deviations + relevant context (PRs, version dependencies, known issues). Footnote numbering is per-GPU-section. Attach the marker to the specific number it applies to when a deviation affects only one length.
8. **Call out any backend whose ranking flips between 10k and 100k** in the section's Notes — that inversion is the main reason both lengths are measured, and it's the thing a reader picking a production default most needs to see.
9. Also add a Notes bullet for any cell where the two lengths *diverged* (worked at 10k, failed at 100k). A length-sensitive failure is a much more actionable finding than a flat rejection: it usually means an OOM budget or a kernel tiling bug rather than an unsupported shape.

Read the existing file to preserve sections for other GPUs and the "How to read" preamble.

## Step 6: Commit the matrix update

Show the diff to the user before committing. If they confirm:

```bash
git add backend_compatibility.md
git -c user.name=... -c user.email=... commit -m "Refresh backend compatibility matrix on <GPU>"
```

Per CLAUDE.md: do not commit unless explicitly asked.

## Failure-recovery patterns

The script's classifier handles the common cases (head_size, kv_cache_dtype, FlashInfer template missing, FA cute Q-dtype assert, OOM, FlexAttention KV-sharing). If a new failure mode appears, add a new `(regex, label)` to the `patterns` list in `run_one()` rather than letting it fall through to "unknown — see log."

**Never leave an `unknown — see log` cell in the table.** Open the log and find the root-cause line before recording anything: these are as often a broken host environment as a real incompatibility, and an unexamined one silently becomes a fake `❌` for a backend that actually works. The case that bit us: `FileNotFoundError: 'ninja'` during engine init, because the tool lives in `~/.local/bin` and that directory is **not on the PATH of a non-interactive ssh shell** — so the sweep failed where an interactive run succeeded. Export `PATH=$HOME/.local/bin:$PATH` in the sweep command, and verify build tools with `which ninja` over the *same* non-interactive channel the sweep uses, not from a login shell.

**Distinguish "failed at engine init" from "failed while serving the request."** With chunked prefill disabled, `max_num_batched_tokens` equals `max_model_len`, so vLLM's profile run compiles *and Inductor-autotunes* at the full sweep length before any request is served. A length-gated crash can therefore live entirely in `LLM()` construction and have nothing to do with attention — which is why it looks backend-independent across the whole row. Grep the log for `Engine core initialization failed` to tell the two apart; only a failure during generation is evidence about a backend. Reproduce the init-time class with a bare `LLM(...)` and no `generate()` call — if that crashes, the backend column in the table is not the story.

**Never trust the reported frame of an async CUDA error — re-run with `CUDA_LAUNCH_BLOCKING=1` before naming a kernel.** CUDA errors surface at the next synchronization point, not at the faulting launch, so the traceback routinely indicts an innocent kernel that merely happened to sync (Inductor's autotuner is a frequent false accusation, since it synchronizes constantly). We recorded a wrong root cause this way. Two cheap cross-checks on any accusation: read the *generated source* of the suspect kernel and confirm its offset arithmetic can actually overflow at the observed size, and check whether the failure lands on a `torch.cuda.synchronize()` in unrelated code.

**A crash that scales with context length is usually 32-bit index overflow, and the threshold is computable.** Triton offsets from `tl.arange` are int32, so `tokens × row_stride_bytes` wraps at 2³¹. Divide 2³¹ by the byte stride of the widest per-token activation and bisect around that number — if the empirical PASS/CRASH boundary brackets the predicted token count, the diagnosis is essentially proven and the bug report can say so instead of guessing. Bisect on `max_model_len` with everything else fixed.

**An apparent dtype rejection can really be an OOM, and the memory budget silently decides which precision tier you measure.** The harness walks tiers until one succeeds and only annotates the *winning* tier, so a cell can quietly land on BF16 KV because the FP8-KV tier ran out of memory rather than because the backend refused the dtype — and FP8 KV makes this *more* likely, since smaller blocks mean vLLM allocates more of them to fill `gpu_memory_utilization`, leaving nothing for a backend that grabs its workspace after engine init. Before recording a KV-dtype gate, open the losing tier's per-cell log and confirm the reason. If it is an OOM, re-measure at `--gpu_memory_utilization 0.75` and footnote it, rather than reporting a precision the hardware never refused.

**Verify a surprising failure on a confirmed-idle GPU before recording it.** A cell that contradicts an earlier measurement of the same configuration is a measurement artifact until proven otherwise — a previous engine still releasing memory is enough to turn a working cell into an OOM. Check `nvidia-smi --query-gpu=memory.used` and `--query-compute-apps=pid,used_memory` are both empty, then re-run that cell alone. Also delete `~/vllm_backend_matrix_logs/` before a re-run, so logs from a killed sweep can't be mistaken for the new one's.

If a cell unexpectedly takes much longer than expected (e.g. 30s for a 4B model where we'd expect ~5s), that's worth a profile dive — note it and follow up with the `profile-llm` skill rather than just recording the number.

## What this skill is *not* for

- **Picking a model** — that's a separate concern (architecture, quality, license). This skill only measures inference performance.
- **Tuning a specific kernel** — use the `profile-llm` skill (nsys → ncu → kernel-config iteration) for that.
- **Multi-GPU / distributed measurements** — single-GPU prefill only.
- **Decode-throughput sweeps** — this skill measures prefill latency at long context, where attention dominates. Decode-heavy workloads have a very different bottleneck profile (memory-bandwidth-bound, KV cache access) and would need a different harness.
