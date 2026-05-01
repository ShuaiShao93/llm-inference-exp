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


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vLLM inference latency")
    parser.add_argument("--model", default="Firworks/Ministral-3-8B-Instruct-2512-nvfp4")
    parser.add_argument(
        "--precision",
        default="fp4",
        help="Target precision: fp4, fp8, bf16, fp16, awq, gptq, or auto. "
             "Exits with an error if it doesn't match the model's built-in quantization.",
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
        default="auto",
        help="KV cache dtype: fp8, fp8_e5m2, fp8_e4m3, or auto (default: auto).",
    )
    parser.add_argument(
        "--attention_backend",
        default="FLASH_ATTN",
        help="Attention backend: FLASH_ATTN, FLASHINFER, TRITON_ATTN, etc. (default: FLASH_ATTN).",
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
    qcfg = cfg.get("quantization_config", {})
    fmt = qcfg.get("format", "").lower()
    method = qcfg.get("quant_method", "").lower()

    if fmt in _FP4_FORMATS or "fp4" in fmt:
        return "fp4"
    if "fp8" in fmt or method == "fp8":
        return "fp8"
    if method in ("awq", "gptq", "bitsandbytes"):
        return method

    dtype = cfg.get("torch_dtype", "bfloat16")
    return {"bfloat16": "bf16", "float16": "fp16"}.get(dtype, dtype)


def check_precision(model_id, requested):
    """Exit with an error if requested precision doesn't match the model."""
    if requested == "auto":
        return
    model_precision = get_model_precision(model_id)
    if model_precision != requested:
        sys.exit(
            f"Error: requested precision '{requested}' does not match "
            f"model's built-in precision '{model_precision}'.\n"
            f"Either use --precision {model_precision} or choose a model "
            f"that is already quantized to {requested}."
        )


def main():
    args = parse_args()

    precision = _DTYPE_ALIASES.get(args.precision.lower(), args.precision.lower())

    check_precision(args.model, precision)

    attention_backend = args.attention_backend.upper()

    llm_kwargs = {
        "model": args.model,
        "tokenizer_mode": args.tokenizer_mode,
        "attention_config": {"backend": attention_backend},
        "enable_flashinfer_autotune": False,
        "kv_cache_dtype": args.kv_cache_precision,
        "limit_mm_per_prompt": {"image": 0},
        "enable_prefix_caching": False,
        "max_model_len": args.input_tokens + args.max_output_tokens,
    }

    if args.sliding_window is not None:
        llm_kwargs["hf_overrides"] = {"sliding_window": args.sliding_window}

    # Only set explicit quantization for vLLM-managed methods (awq, gptq, …)
    # fp4/fp8 are embedded in the model config and auto-detected by vLLM
    if precision not in _DTYPE_VALUES and precision not in ("fp4", "fp8", "auto"):
        llm_kwargs["quantization"] = precision
        llm_kwargs["dtype"] = "auto"
    elif precision in _DTYPE_VALUES:
        llm_kwargs["dtype"] = precision

    print(f"Model:            {args.model}")
    print(f"Precision:        {args.precision}")
    print(f"Input tokens:     {args.input_tokens}")
    print(f"Max output tokens:{args.max_output_tokens}")
    print(f"Sliding window:   {args.sliding_window if args.sliding_window is not None else 'model default'}")
    print(f"KV cache prec.:   {args.kv_cache_precision}")
    print(f"Attention backend:{attention_backend}")
    print(f"Prefix caching:   disabled")
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

    print("Warming up...")
    llm.generate(prompt, sampling_params, use_tqdm=False)

    latencies = []
    print(f"Running {args.num_runs} iterations...")
    for i in range(args.num_runs):
        start = time.perf_counter()
        llm.generate(prompt, sampling_params, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.1f} ms")

    print()
    print(f"Mean latency:   {np.mean(latencies):.1f} ms")
    print(f"Median latency: {np.median(latencies):.1f} ms")
    print(f"P99 latency:    {np.percentile(latencies, 99):.1f} ms")
    print(f"Min latency:    {np.min(latencies):.1f} ms")
    print(f"Max latency:    {np.max(latencies):.1f} ms")


if __name__ == "__main__":
    main()
