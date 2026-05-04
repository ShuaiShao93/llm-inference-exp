import argparse
import json
import os
import random
import sys
import time

import numpy as np

_FP4_FORMATS = {"nvfp4-pack-quantized", "nvfp4", "mxfp4", "dense-mxfp4"}
_DTYPE_ALIASES = {"bf16": "bfloat16", "fp16": "float16"}
_DTYPE_VALUES = {"float16", "bfloat16", "float32", "auto"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT-LLM inference latency")
    parser.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct-FP8-Block")
    parser.add_argument(
        "--precision",
        default="fp8",
        help="Target precision: fp4, fp8, bf16, fp16, or auto. "
             "Exits with an error if it doesn't match the model's built-in quantization.",
    )
    parser.add_argument("--input_tokens", type=int, default=15000)
    parser.add_argument("--max_output_tokens", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument(
        "--kv_cache_precision",
        default="fp8",
        help="KV cache dtype: fp8, nvfp4, auto, or a torch dtype string (default: fp8).",
    )
    parser.add_argument(
        "--tokenizer_mode",
        default="auto",
        help="Tokenizer mode: auto or slow (default: auto).",
    )
    parser.add_argument(
        "--disable_chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable chunked prefill (default: True; use --no-disable_chunked_prefill to enable).",
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

    if method == "modelopt":
        algo = qcfg.get("quant_algo", "").upper()
        if "NVFP4" in algo or "MXFP4" in algo:
            return "fp4"
        if "FP8" in algo:
            return "fp8"

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

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    max_seq_len = args.input_tokens + args.max_output_tokens

    llm_kwargs = dict(
        model=args.model,
        tokenizer_mode=args.tokenizer_mode,
        kv_cache_config=KvCacheConfig(
            dtype=args.kv_cache_precision,
            free_gpu_memory_fraction=0.9,
        ),
        enable_chunked_prefill=not args.disable_chunked_prefill,
        max_seq_len=max_seq_len,
        max_num_tokens=max_seq_len,
    )

    if precision in _DTYPE_VALUES:
        llm_kwargs["dtype"] = precision

    print(f"Model:            {args.model}")
    print(f"Precision:        {args.precision}")
    print(f"Input tokens:     {args.input_tokens}")
    print(f"Max output tokens:{args.max_output_tokens}")
    print(f"KV cache prec.:   {args.kv_cache_precision}")
    print(f"Chunked prefill:  {'disabled' if args.disable_chunked_prefill else 'enabled'}")
    print(f"Max seq len:      {max_seq_len}")
    print()

    llm = LLM(**llm_kwargs)

    tok = llm.tokenizer
    # TRT-LLM wraps the HF tokenizer; get vocab_size from the inner tokenizer
    inner = getattr(tok, "tokenizer", tok)
    vocab_size = (
        getattr(inner, "vocab_size", None)
        or getattr(inner, "_vocab_size", None)
        or len(inner.get_vocab())
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_output_tokens,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )

    print("Warming up...")
    llm.generate(random.choices(range(1, vocab_size - 1), k=args.input_tokens), sampling_params)

    import ctypes
    cudart = ctypes.CDLL("libcudart.so")

    latencies = []
    print(f"Running {args.num_runs} iterations...")
    cudart.cudaProfilerStart()
    for i in range(args.num_runs):
        prompt_token_ids = random.choices(range(1, vocab_size - 1), k=args.input_tokens)
        start = time.perf_counter()
        llm.generate(prompt_token_ids, sampling_params)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.1f} ms")
    cudart.cudaProfilerStop()

    print()
    print(f"Mean latency:   {np.mean(latencies):.1f} ms")
    print(f"Median latency: {np.median(latencies):.1f} ms")
    print(f"P99 latency:    {np.percentile(latencies, 99):.1f} ms")
    print(f"Min latency:    {np.min(latencies):.1f} ms")
    print(f"Max latency:    {np.max(latencies):.1f} ms")

    llm.shutdown()


if __name__ == "__main__":
    main()
