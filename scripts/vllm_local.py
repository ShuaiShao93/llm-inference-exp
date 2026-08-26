import argparse
import json
import os
import random
import sys
import time

import numpy as np
from vllm import LLM, SamplingParams

_FP4_FORMATS = {"nvfp4-pack-quantized", "nvfp4", "mxfp4", "dense-mxfp4"}
_DTYPE_ALIASES = {"bf16": "bfloat16", "fp16": "float16"}
_DTYPE_VALUES = {"float16", "bfloat16", "float32", "float", "auto", "half"}

# vLLM's online-quantization shorthands: quantize a BF16/FP16 checkpoint at load
# time instead of loading a pre-quantized one. Passed straight through as
# `quantization=`, so this set only needs to list what vLLM accepts.
_ONLINE_SCHEMES = {
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "mxfp8",
    "nvfp4_per_token",
    "int8_per_channel_weight_only",
}

# These shorthands set only vLLM's `moe` spec, so dense Linear layers fall back to
# UnquantizedLinearMethod. On a model without routed experts they quantize nothing
# and the run silently reports BF16 latency under an FP4/INT8 label.
_ONLINE_MOE_ONLY_SCHEMES = {"nvfp4_per_token", "int8_per_channel_weight_only"}
_MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vLLM inference latency")
    parser.add_argument("--model", default="Firworks/Ministral-3-8B-Instruct-2512-nvfp4")
    parser.add_argument(
        "--precision",
        default="fp4",
        help="Either an offline precision that must match the checkpoint's built-in "
             "quantization (fp4, fp8, int8, bf16, fp16, awq, gptq, auto), or an online "
             "scheme quantized at load time from a BF16/FP16 base: "
             + ", ".join(sorted(_ONLINE_SCHEMES))
             + ". Exits with an error if the checkpoint and the request disagree.",
    )
    parser.add_argument("--input_tokens", type=int, default=15000)
    parser.add_argument("--max_output_tokens", type=int, default=1)
    parser.add_argument(
        "--sliding_window",
        type=int,
        default=None,
        help="Sliding window attention size. Omit to use the model's default.",
    )
    parser.add_argument(
        "--tokenizer_mode",
        default="auto",
        help="Tokenizer mode: auto, mistral, slow (default: auto). Use 'mistral' for Mistral/Ministral models.",
    )
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument(
        "--kv_cache_precision",
        default="fp8",
        help="KV cache dtype: fp8, fp8_e5m2, fp8_e4m3, or auto (default: fp8).",
    )
    parser.add_argument(
        "--attention_backend",
        default="FLASHINFER",
        help="Attention backend: FLASH_ATTN, FLASHINFER, TRITON_ATTN, etc. (default: FLASHINFER).",
    )
    parser.add_argument(
        "--disable_chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable chunked prefill (default: True; use --no-disable_chunked_prefill to enable).",
    )
    parser.add_argument(
        "--flashinfer_autotune",
        action="store_true",
        default=False,
        help="Enable FlashInfer autotuning (only meaningful with FLASHINFER backend).",
    )
    parser.add_argument(
        "--profile_dir",
        default=None,
        help="If set, collect a torch profiler trace of the first post-warmup run "
             "and save it to this directory. Open with Perfetto UI or TensorBoard.",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Optional LoRA adapter. HuggingFace ID or local path. "
             "When set, vLLM is launched with enable_lora=True and the adapter "
             "is applied to every generate() call (warmup and timed runs).",
    )
    parser.add_argument(
        "--max_lora_rank",
        type=int,
        default=64,
        help="Max LoRA rank to budget for in vLLM. Must be >= the adapter's actual rank.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="Override vLLM's gpu_memory_utilization. Lower it when a backend OOMs "
             "allocating its own workspace: vLLM sizes the KV cache to fill this "
             "budget, leaving nothing for workspaces allocated after engine init.",
    )
    return parser.parse_args()


def _load_hf_config(model_id):
    if os.path.isdir(model_id):
        return json.load(open(os.path.join(model_id, "config.json")))
    try:
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(model_id, "config.json")))
    except Exception:
        return {}


def get_model_precision(model_id):
    """Return model's native precision as a short string (fp4, fp8, bf16, fp16, awq, gptq…)."""
    cfg = _load_hf_config(model_id)
    # Multimodal models nest the quantization_config under text_config (or sometimes language_config).
    qcfg = cfg.get("quantization_config") or {}
    if not qcfg:
        for k in ("text_config", "language_config"):
            sub = cfg.get(k) or {}
            qcfg = sub.get("quantization_config") or {}
            if qcfg:
                break
    fmt = qcfg.get("format", "").lower()
    method = qcfg.get("quant_method", "").lower()

    if fmt in _FP4_FORMATS or "fp4" in fmt:
        return "fp4"
    if "fp8" in fmt or method == "fp8":
        return "fp8"
    if method in ("awq", "gptq", "bitsandbytes"):
        return method

    if method == "modelopt":
        algo = qcfg.get("quant_algo", "").upper()
        if "NVFP4" in algo or "MXFP4" in algo or "FP4" in algo:
            return "fp4"
        if "FP8" in algo:
            return "fp8"

    # compressed-tensors models: infer precision from config_groups weight num_bits
    if method == "compressed-tensors":
        groups = qcfg.get("config_groups", {})
        if groups:
            first = next(iter(groups.values()))
            w = first.get("weights", {})
            bits, wtype = w.get("num_bits"), w.get("type", "")
            if bits == 4 and wtype == "float":
                return "fp4"
            if bits == 8 and wtype == "float":
                return "fp8"
            if bits == 8 and wtype == "int":
                return "int8"

    dtype = cfg.get("torch_dtype") or cfg.get("text_config", {}).get("torch_dtype", "bfloat16")
    return {"bfloat16": "bf16", "float16": "fp16"}.get(dtype, dtype)


def check_precision(model_id, requested):
    """Exit with an error if requested precision doesn't match the model."""
    if requested == "auto":
        return
    model_precision = get_model_precision(model_id)
    model_precision = _DTYPE_ALIASES.get(model_precision, model_precision)
    if model_precision != requested:
        sys.exit(
            f"Error: requested precision '{requested}' does not match "
            f"model's built-in precision '{model_precision}'.\n"
            f"Either use --precision {model_precision} or choose a model "
            f"that is already quantized to {requested}."
        )


def has_routed_experts(model_id):
    cfg = _load_hf_config(model_id)
    for scope in (cfg, cfg.get("text_config") or {}, cfg.get("language_config") or {}):
        if scope.get("enable_moe_block") is True:
            return True
        if any(scope.get(k) for k in _MOE_CONFIG_KEYS):
            return True
    return False


def check_online_base(model_id, scheme):
    """Online quantization starts from unquantized weights; refuse a quantized base."""
    model_precision = get_model_precision(model_id)
    if model_precision not in ("bf16", "fp16"):
        sys.exit(
            f"Error: online scheme '{scheme}' needs a BF16/FP16 base checkpoint, but "
            f"'{model_id}' is already quantized to '{model_precision}'.\n"
            f"Either point --model at the unquantized base or use --precision "
            f"{model_precision} to benchmark the pre-quantized checkpoint offline."
        )
    if scheme in _ONLINE_MOE_ONLY_SCHEMES and not has_routed_experts(model_id):
        sys.exit(
            f"Error: online scheme '{scheme}' only quantizes routed-expert (MoE) "
            f"weights, and '{model_id}' has none — every Linear layer would stay BF16, "
            f"so the run would report BF16 latency under a '{scheme}' label.\n"
            f"vLLM has no online scheme that quantizes dense Linear layers below 8 bits; "
            f"use a pre-quantized checkpoint for sub-8-bit on a dense model."
        )


def main():
    args = parse_args()

    precision = _DTYPE_ALIASES.get(args.precision.lower(), args.precision.lower())
    online = precision in _ONLINE_SCHEMES

    if online:
        check_online_base(args.model, precision)
    else:
        check_precision(args.model, precision)

    attention_backend = args.attention_backend.upper()

    llm_kwargs = {
        "model": args.model,
        "tokenizer_mode": args.tokenizer_mode,
        "attention_config": {"backend": attention_backend},
        "enable_flashinfer_autotune": args.flashinfer_autotune,
        "kv_cache_dtype": args.kv_cache_precision,
        "limit_mm_per_prompt": {"image": 0},
        "enable_prefix_caching": False,
        "enable_chunked_prefill": not args.disable_chunked_prefill,
        "max_model_len": args.input_tokens + args.max_output_tokens,
    }

    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    lora_request = None
    if args.lora is not None:
        from vllm.lora.request import LoRARequest
        from huggingface_hub import snapshot_download
        # Resolve HF id -> local path (LoRARequest needs a local path).
        if not os.path.isdir(args.lora):
            local = snapshot_download(repo_id=args.lora, allow_patterns=["adapter_*", "*.json"])
        else:
            local = args.lora
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        llm_kwargs["max_loras"] = 1
        lora_request = LoRARequest("bench-lora", 1, lora_path=local)

    if args.sliding_window is not None:
        llm_kwargs["hf_overrides"] = {"sliding_window": args.sliding_window}

    if args.profile_dir is not None:
        profile_dir = os.path.abspath(args.profile_dir)
        os.makedirs(profile_dir, exist_ok=True)
        llm_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": profile_dir,
        }

    # Online schemes and vLLM-managed methods (awq, gptq, …) are both requested via
    # `quantization`; fp4/fp8/int8 are embedded in the model config and auto-detected.
    if online:
        llm_kwargs["quantization"] = precision
    elif precision not in _DTYPE_VALUES and precision not in ("fp4", "fp8", "int8", "auto"):
        llm_kwargs["quantization"] = precision
        llm_kwargs["dtype"] = "auto"
    elif precision in _DTYPE_VALUES:
        llm_kwargs["dtype"] = precision

    print(f"Model:            {args.model}")
    print(f"Precision:        {args.precision} ({'online' if online else 'offline'})")
    print(f"Input tokens:     {args.input_tokens}")
    print(f"Max output tokens:{args.max_output_tokens}")
    print(f"Sliding window:   {args.sliding_window if args.sliding_window is not None else 'model default'}")
    print(f"KV cache prec.:   {args.kv_cache_precision}")
    print(f"Attention backend:{attention_backend}")
    print(f"Chunked prefill:  {'disabled' if args.disable_chunked_prefill else 'enabled'}")
    print(f"FlashInfer autotune:{args.flashinfer_autotune}")
    print(f"Prefix caching:   disabled")
    print(f"LoRA adapter:     {args.lora if args.lora else 'none'}")
    if args.profile_dir is not None:
        print(f"Profile dir:      {os.path.abspath(args.profile_dir)}")
    print()

    llm = LLM(**llm_kwargs)

    # vLLM exposes the tokenizer differently depending on tokenizer_mode:
    # - "mistral": llm_engine.tokenizer is a TokenizerGroup with .tokenizer
    # - "auto":    llm_engine.tokenizer is a CachedTokenizersBackend with ._tokenizer
    _backend = llm.llm_engine.tokenizer
    tok = (
        getattr(_backend, "tokenizer", None)
        or getattr(_backend, "_tokenizer", None)
        or _backend
    )
    # transformers tokenizers expose vocab_size as an attribute;
    # tokenizers.Tokenizer (Rust-backed) exposes it via get_vocab_size()
    _get_vs = getattr(tok, "get_vocab_size", None)
    vocab_size = (
        getattr(tok, "vocab_size", None)
        or getattr(tok, "_vocab_size", None)
        or (_get_vs() if callable(_get_vs) else None)
    )
    if vocab_size is None:
        raise RuntimeError(f"Cannot determine vocab_size from {type(tok)}")

    prompt_token_ids = random.choices(range(1, vocab_size - 1), k=args.input_tokens)
    prompt = {"prompt_token_ids": prompt_token_ids}

    sampling_params = SamplingParams(
        max_tokens=args.max_output_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    gen_kwargs = {"sampling_params": sampling_params, "use_tqdm": False}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request

    print("Warming up...")
    llm.generate(prompt, **gen_kwargs)

    import ctypes
    cudart = ctypes.CDLL("libcudart.so")

    latencies = []
    print(f"Running {args.num_runs} iterations...")
    cudart.cudaProfilerStart()
    for i in range(args.num_runs):
        if i == 0 and args.profile_dir is not None:
            llm.start_profile()
        start = time.perf_counter()
        llm.generate(prompt, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if i == 0 and args.profile_dir is not None:
            llm.stop_profile()
            print(f"  Trace saved to: {os.path.abspath(args.profile_dir)}")
        latencies.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.1f} ms")
    cudart.cudaProfilerStop()

    print()
    print(f"Mean latency:   {np.mean(latencies):.1f} ms")
    print(f"Median latency: {np.median(latencies):.1f} ms")
    print(f"P99 latency:    {np.percentile(latencies, 99):.1f} ms")
    print(f"Min latency:    {np.min(latencies):.1f} ms")
    print(f"Max latency:    {np.max(latencies):.1f} ms")


if __name__ == "__main__":
    main()
