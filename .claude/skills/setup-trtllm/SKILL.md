---
name: setup-trtllm
description: Set up TensorRT-LLM (TRT-LLM) for LLM inference benchmarking on a fresh Ubuntu 24.04 VM with an NVIDIA GPU. Use when user asks to "set up trtllm", "install tensorrt-llm", "configure trtllm environment", or wants to run TRT-LLM inference on a new machine.
argument-hint: [--env-name <name>] [--trtllm-version <version>]
allowed-tools: [Bash, Read, Write, Edit]
---

# Setup TensorRT-LLM

Full setup of TRT-LLM on a fresh Ubuntu 24.04 VM. Covers CUDA upgrade, Miniconda, conda env creation, and dependency installation.

## Arguments

$ARGUMENTS

Parse optional args: `--env-name` (default: `trtllm`), `--trtllm-version` (default: latest stable or RC available on PyPI).

## Key Caveats (Read First)

- **Isolate the conda env from user-site packages**: if the system Python shares the same version (e.g. Python 3.12), packages installed into `~/.local/lib/python3.12/` will bleed into the conda env and can shadow the pinned torch or other deps. Always run TRT-LLM with `PYTHONNOUSERSITE=1`, and verify the env is self-contained (see Step 5).
- **RC releases often have broken dependency pins**: install torch separately first, then TRT-LLM with `--no-deps`, then remaining deps. This avoids pip resolver conflicts between torch and cuda-python/cuda-bindings.
- **flashinfer-cubin must match flashinfer-python**: TRT-LLM ships a specific `flashinfer-python` version; run `pip show flashinfer-python` after install and force-reinstall `flashinfer-cubin` to the same version.
- **FP4 requires offline quantization**: pre-quantized modelopt-format checkpoints may not exist on HuggingFace for all models. Use `scripts/quantize_trtllm.py --qformat fp4` to produce one from a BF16 model.
- **XQA attention kernels require SM100+** (datacenter Blackwell). Consumer Blackwell (SM120) falls back to the standard PyTorch attention path automatically.

---

## Step 1: Upgrade CUDA

TRT-LLM requires a recent CUDA toolkit. Check the TRT-LLM release notes for the minimum required version, then install via the NVIDIA apt repo.

Add the NVIDIA apt repo if not already present (get the correct `.deb` URL for your OS from https://developer.nvidia.com/cuda-downloads):

```bash
wget <cuda-keyring-deb-url>
sudo dpkg -i cuda-keyring_*_all.deb
sudo apt-get update
```

Install the required toolkit version (e.g. `cuda-toolkit-13-2`) alongside any existing CUDA without removing it:

```bash
sudo apt-get install -y cuda-toolkit-<major>-<minor>
```

Verify the right version is active:

```bash
nvcc --version
ls /usr/local/cuda   # should symlink to the new version
```

---

## Step 2: Install Miniconda

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
```

Accept the Anaconda Terms of Service (required on first run, otherwise `conda create` silently fails):

```bash
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## Step 3: Create the Conda Environment

```bash
$HOME/miniconda3/bin/conda create -n <env-name> python=3.12 -y
```

Bootstrap pip (conda's Python image may not ship pip by default):

```bash
$HOME/miniconda3/envs/<env-name>/bin/python -m pip install --upgrade pip
```

---

## Step 4: Install TRT-LLM and Dependencies

### 4a. Install torch first

TRT-LLM is compiled against a specific torch version. Check the TRT-LLM release notes or the wheel metadata for the required version and install it before TRT-LLM:

```bash
pip install "torch==<required-version>"
```

### 4b. Install TRT-LLM with --no-deps

Using `--no-deps` skips the broken version resolver in RC releases:

```bash
pip install tensorrt-llm==<version> --pre --no-deps
```

### 4c. Fix flashinfer-cubin version

TRT-LLM ships a pinned `flashinfer-python`. The system may have a different `flashinfer-cubin`. Force them to match:

```bash
pip show flashinfer-python   # note the version
pip install "flashinfer-cubin==<same-version>" --force-reinstall
```

### 4d. Install remaining dependencies

Install TRT-LLM's runtime dependencies. The full list is in the TRT-LLM wheel metadata:

```bash
pip show tensorrt-llm   # check Requires: field for the dep list
pip install --pre <deps...>
```

Two packages commonly need a final pin after other deps upgrade them:

```bash
pip install "numpy<2.4" "setuptools<80"
```

---

## Step 5: Verify the Installation

Always verify with `PYTHONNOUSERSITE=1` to confirm every dependency is inside the conda env and nothing relies on user-local packages:

```bash
PYTHONNOUSERSITE=1 $HOME/miniconda3/envs/<env-name>/bin/python \
  -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

If anything is missing, install it inside the conda env (not with `--break-system-packages`):

```bash
$HOME/miniconda3/envs/<env-name>/bin/pip install <missing-package>
```

Warnings about vllm or torchao are harmless.

---

## Step 6: Smoke Test

Use the benchmark script in this repo for a quick end-to-end check:

```bash
PYTHONNOUSERSITE=1 $HOME/miniconda3/envs/<env-name>/bin/python scripts/trtllm_local.py \
  --model <fp8-model-id> \
  --precision fp8 \
  --input_tokens 1000 \
  --num_runs 3 \
  --kv_cache_precision fp8
```

---

## Reference: Benchmark Script

`scripts/trtllm_local.py` benchmarks latency for a given model, precision, and input length. Key flags:

- `--model`: HuggingFace model ID or local checkpoint path
- `--precision`: fp4, fp8, bf16, or auto
- `--input_tokens`: number of input tokens (use random token IDs)
- `--kv_cache_precision`: fp8, auto
- `--num_runs`: number of timed iterations after warmup

## Reference: How to Quantize a Model

If a pre-quantized checkpoint at the desired precision is not available, use `scripts/quantize_trtllm.py` to quantize from BF16. This uses nvidia-modelopt and exports an HF-format checkpoint with `hf_quant_config.json` that TRT-LLM loads automatically.

```bash
python scripts/quantize_trtllm.py \
  --model <hf-model-id> \
  --qformat fp4   # or fp8
  --output_dir /tmp/quantized_model
```

Key points:
- Uses `modelopt.export_hf_checkpoint` → HF-format weights + `hf_quant_config.json`
- KV cache is not baked in; set `--kv_cache_precision fp8` at inference time
- Mirrors the NVIDIA Model-Optimizer approach: https://nvidia.github.io/TensorRT-LLM/features/quantization.html

## Reference: Expected Benchmark Numbers (RTX PRO 6000 Blackwell, SM120)

| Input tokens | Model | Precision | KV cache | Mean latency |
|---|---|---|---|---|
| 1k | Llama-3.2-3B FP8-Block | fp8 | fp8 | ~14 ms |
| 15k | Llama-3.2-3B FP8-Block | fp8 | fp8 | ~21 ms |
| 100k | Llama-3.2-3B FP8-Block | fp8 | fp8 | ~5300 ms |
| 100k | Llama-3.2-3B NVFP4 (modelopt) | fp4 | fp8 | ~4851 ms |
