---
name: benchmark-gemma4
description: Benchmark Google Gemma 4 (E4B / A4B) on vLLM. Covers checkpoint sourcing, the model's split head_dim and multimodal-bidirectional attention, and the vLLM/FlashInfer patches required for FP4 weights + FP8 KV at head_dim=512. Use when the user asks to benchmark Gemma 4, hits a "head size not supported" / "kv_cache_sf must be provided" error with Gemma 4, or needs to know what's special about this model family.
argument-hint: [--variant e4b|a4b] [--input-tokens N] [--kv-cache fp8|nvfp4]
allowed-tools: [Bash, Read, Edit, Write]
---

# Benchmark Gemma 4 on vLLM

Gemma 4 has three properties that break naive benchmarking and force kernel/config choices the other Gemma generations don't:

1. **Heterogeneous head_dim**: `head_dim=256` on sliding-window layers and `global_head_dim=512` on full-attention layers. Most kernels and backends only support one head_size per model; some backends (FlashAttention) reject head_size>256 outright. vLLM's `Gemma4Config.verify_and_update_config` auto-forces `TRITON_ATTN` when the user doesn't specify a backend.
2. **Conditional multimodal-bidirectional attention**: when `text_config.use_bidirectional_attention == "vision"`, vLLM marks the model `is_mm_prefix_lm=True` and refuses any attention backend whose `supports_mm_prefix()` is False (FlashInfer's is). Only some Gemma 4 variants (notably A4B) set that field; E4B leaves it `null`.
3. **MoE on A4B, dense on E4B**: A4B is `enable_moe_block=True` (26B total / 4B active), E4B is dense (~8B). Memory footprint and KV-cache budget differ significantly.

These are all in `text_config` and are detectable from `config.json` without loading weights.

## Arguments

$ARGUMENTS

Parse optional args: `--variant` (`e4b` or `a4b`), `--input-tokens` (default 100000), `--kv-cache` (`fp8` or `nvfp4`). If omitted, ask the user or use the most-common combo (`e4b`, 100000, `fp8`).

## Variants

| Variant | HF id (NVFP4) | total / active params | text_config.use_bidirectional_attention | head_dim / global_head_dim | layer_types |
|---|---|---|---|---|---|
| **E4B** | `bg-digitalservices/Gemma-4-E4B-it-NVFP4` (modelopt fmt) or `prithivMLmods/gemma-4-E4B-it-NVFP4` (compressed-tensors, W4A16 — slower) | ~8B dense | `null` | 256 / 512 | mostly sliding, some full |
| **A4B** | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` (compressed-tensors fmt) | 26B / ~4B (MoE) | `"vision"` → forces `is_mm_prefix_lm=True` | 256 / 512 | sliding + full, MoE FFN |

Always verify by reading `config.json` (cached under `~/.cache/huggingface/hub/.../snapshots/*/config.json`) before assuming — community uploads vary.

## Sourcing weights

**Prefer pre-quantized HF checkpoints over local quantization for Gemma 4.** `scripts/quantize_trtllm.py` uses `nvidia-modelopt`, which lags `transformers` releases by months and typically doesn't recognize the Gemma 4 architecture yet. `llm-compressor` (vLLM project) tracks `transformers` more closely but currently emits W4A16, not real W4A4 NVFP4, so the FP4 tensor cores aren't engaged in GEMM. Either way, do **not** upgrade `transformers` or `modelopt` inside the `trtllm` conda env to fix this — it breaks TRT-LLM's pinned deps.

Pull both as needed (downloads land in `~/.cache/huggingface/`):

```bash
huggingface-cli download bg-digitalservices/Gemma-4-E4B-it-NVFP4
huggingface-cli download RedHatAI/gemma-4-26B-A4B-it-NVFP4
```

Note that the modelopt-format quant config does not carry a `format` field (it uses `quant_method: modelopt` + `quant_algo: NVFP4`), while compressed-tensors uses `format: nvfp4-pack-quantized`. vLLM accepts both. `scripts/vllm_local.py`'s precision auto-detect only knows the compressed-tensors form — pass `--precision auto` when running a modelopt-format checkpoint, or extend the detector to look at `quant_algo`.

For a fair vLLM-vs-TRT-LLM comparison on Gemma 4, the cleanest path is to quantize the BF16 base yourself with `scripts/quantize_trtllm.py` — but this only works if `nvidia-modelopt` already supports Gemma 4. Check before relying on it.

## Running on vLLM

### What works out of the box

`TRITON_ATTN` + FP8 KV runs both variants end-to-end without patches. Functional but ~10× slower than FLASHINFER + TRTLLM-gen on long-context prefill, because TRITON_ATTN's kernels aren't tuned for `head_dim=512`. Useful as a sanity check / correctness baseline.

```bash
/usr/bin/python3.12 scripts/vllm_local.py \
    --model bg-digitalservices/Gemma-4-E4B-it-NVFP4 \
    --precision auto --kv_cache_precision fp8 \
    --attention_backend TRITON_ATTN \
    --input_tokens 100000 --max_output_tokens 1
```

### What needs patches: FLASHINFER + TRTLLM-gen at head_dim=512

Three independent gates have to open before this combination dispatches a kernel:

1. **vLLM `FlashInferBackend.get_supported_head_sizes()` must include 512.** Originally `[64, 128, 256]`; PR [vllm#38822](https://github.com/vllm-project/vllm/pull/38822) adds 512. If the installed vLLM predates that, edit `vllm/v1/attention/backends/flashinfer.py` and add 512 to the returned list.

2. **FlashInfer must ship the matching cubins.** PR [flashinfer#2959](https://github.com/flashinfer-ai/flashinfer/pull/2959) added `head_dim=512` to the TRT-LLM FMHA path. Released in `flashinfer-python==0.6.11` (and matching `flashinfer-cubin`). On older versions the dispatch fails with `Missing TRTLLM-GEN kernel (...) headDimQk=512 ...`. Upgrade with `pip install --user --break-system-packages --upgrade flashinfer-python==<ver> flashinfer-cubin==<ver>` — the two packages must be the same version or import-time guard rails refuse to load.

3. **vLLM must view its uint8-backed FP8 KV cache as `torch.float8_e4m3fn` before the FlashInfer call.** vLLM stores FP8 KV as `torch.uint8` (see `vllm/utils/torch_utils.py:STR_DTYPE_TO_TORCH_DTYPE`). Since flashinfer#2954, the trtllm-gen kernels treat any uint8 KV as packed NVFP4 and raise `kv_cache_sf must be provided for NVFP4 KV cache.` if no scales are passed. Every other vLLM v1 backend already does `key_cache.view(self.fp8_dtype)` to bridge this (`triton_attn.py`, `rocm_attn.py`, `rocm_aiter_unified_attn.py`); the FlashInfer backend was the only one missing it. Patch in `FlashInferImpl.forward`, right after `kv_cache_permute = fixed`:

   ```python
   if not self.is_kvcache_nvfp4 and kv_cache_permute.dtype == torch.uint8:
       _fp8_view_dtype = None
       if self.kv_cache_dtype in (torch.float8_e4m3fn, "fp8", "fp8_e4m3"):
           _fp8_view_dtype = torch.float8_e4m3fn
       elif self.kv_cache_dtype in (torch.float8_e5m2, "fp8_e5m2"):
           _fp8_view_dtype = torch.float8_e5m2
       if _fp8_view_dtype is not None:
           kv_cache_permute = kv_cache_permute.view(_fp8_view_dtype)
   ```

   `self.kv_cache_dtype` can arrive as either a `torch.dtype` or a `str` depending on which builder path constructed the impl — handle both. Check upstream vLLM before applying; this may already be merged.

### NVFP4 KV cache at head_dim=512: don't, on B200 today

The FlashInfer 0.6.11 cubin set for `head_dim=512 + KvE2m1 (NVFP4)` only ships **prefill** shapes (`Q128`) and MLA-style **`HVPerCta256`** decode shapes (DeepSeek). There is no plain `headDimPerCtaV=512` NVFP4-KV decode cubin yet, so engine init fails during cudagraph warmup with `Missing TRTLLM-GEN kernel (decode): ... headDimPerCtaV=512 ...`. Use FP8 KV until decode cubins land. (Verify by listing `flashinfer_cubin/cubins/.../fmha/trtllm-gen/ | grep H512 | grep KvE2m1`.)

On consumer Blackwell (SM120) the situation is worse — no `Sm120` cubins exist at all in the trtllm-gen FMHA tree, regardless of head_dim or KV dtype. Use a different backend there.

### Pre-Hopper (Ampere SM 8.x, Turing SM 7.5): INT8 W8A8

GPUs without native FP8 tensor cores (anything below SM 8.9) can't usefully run the FP4/FP8 checkpoints above — those paths fall back to BF16 dequant and waste tensor-core throughput. The right precision is **INT8 W8A8** (CompressedTensors `int-quantized`), which engages the INT8 IMMA tensor cores that have shipped since Turing.

**Don't try to quantize Gemma 4 to INT8 W8A8 locally** with `auto-round` or `llm-compressor`. Both tools calibrate block-by-block, feeding each transformer block its cached input from the previous block's output. Gemma 4's heterogeneous `head_dim` (256 on sliding-window layers, 512 on global-attention layers) means the rotary `cos`/`sin` tensors materialized during the first sliding layer don't match the shape expected by the next global layer — calibration crashes with `RuntimeError: The size of tensor a (512) must match the size of tensor b (256) at non-singleton dimension 3` inside `apply_rotary_pos_emb`. The model's own forward pass dispatches the right rotary per layer; the per-block calibration loop bypasses that dispatch. Same shape mismatch that breaks A4B → FlashInfer at backend-selection time, just surfaced in a different tool.

The practical option is a **pre-quantized community W8A8 INT8 checkpoint** (e.g. `nunusadmqk/gemma-4-E4B-it-W8A8-INT8-v10-datafree`). "Datafree" means RTN — no calibration data — which costs some accuracy but doesn't affect latency benchmarks at all; the kernel choice and compute pattern are identical to a GPTQ-calibrated checkpoint.

On Ampere, only **TRITON_ATTN** survives Gemma 4's `head_dim=512` global layers — `FLASH_ATTN` uses FA2 here (the cute / FA4 path is Hopper+) and FA2 rejects `head_dim>256`; FlashInfer's Ampere head-size set also tops out below 512. Same backend-availability pattern as the L40S section in `backend_compatibility.md`.

```bash
/usr/bin/python3.12 scripts/vllm_local.py \
    --model nunusadmqk/gemma-4-E4B-it-W8A8-INT8-v10-datafree \
    --precision int8 --kv_cache_precision auto \
    --attention_backend TRITON_ATTN \
    --input_tokens 100000 --max_output_tokens 1
```

`kv_cache_precision=auto` resolves to BF16 here (the model's compute dtype) since vLLM's FP8 KV path requires SM ≥ 8.9.

### A4B-specific: `use_bidirectional_attention="vision"` blocks FlashInfer

Even after the patches above, A4B will fail backend selection with:

```
ValueError: Selected backend FLASHINFER is not valid for this configuration.
Reason: ['partial multimodal token full attention not supported']
```

For **text-only** runs (which a 100k-token / 1-token-output benchmark is by construction — `limit_mm_per_prompt={"image": 0}`), this can be overridden via vLLM's `hf_overrides`:

```python
LLM(..., hf_overrides={"text_config": {"use_bidirectional_attention": None}})
```

If your script doesn't expose `hf_overrides`, the cheapest workaround is editing the cached `config.json`'s `text_config.use_bidirectional_attention` to `null`, but that mutates the checkpoint — prefer the runtime override.

Do **not** apply this override when actually running multimodal inputs — it would silently disable the bidirectional vision attention the model was trained with.

## Putting it together

E4B, FLASHINFER + FP4 + FP8 KV, 100k input / 1 output:

```bash
/usr/bin/python3.12 scripts/vllm_local.py \
    --model bg-digitalservices/Gemma-4-E4B-it-NVFP4 \
    --precision auto --kv_cache_precision fp8 \
    --attention_backend FLASHINFER \
    --input_tokens 100000 --max_output_tokens 1
```

A4B, same config — needs `hf_overrides` for text-only:

```python
# inline since vllm_local.py doesn't accept this yet
from vllm import LLM, SamplingParams
llm = LLM(
    model="RedHatAI/gemma-4-26B-A4B-it-NVFP4",
    attention_config={"backend": "FLASHINFER"},
    kv_cache_dtype="fp8",
    limit_mm_per_prompt={"image": 0},
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    max_model_len=100001,
    hf_overrides={"text_config": {"use_bidirectional_attention": None}},
)
```

## Sanity checks before benchmarking

1. `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` — confirm SM100 (B200) for the FP4+FP8-KV head_dim=512 path. Other Blackwell variants (SM120) need a different plan.
2. `python -c "import flashinfer; print(flashinfer.__version__)"` — must match `flashinfer-cubin`.
3. `grep get_supported_head_sizes ~/.local/lib/python3.12/site-packages/vllm/v1/attention/backends/flashinfer.py` — confirm 512 is in the list.
4. Read `text_config.use_bidirectional_attention` from the model's `config.json` and the architecture (MoE vs dense) before picking backend / memory budget.
