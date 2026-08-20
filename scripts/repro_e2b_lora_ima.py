#!/usr/bin/env python3
"""Minimal repro for the Gemma 4 E2B + LoRA illegal memory access.

The crash happens while constructing the engine (profile run), so no
generate() call is needed. Toggle --eager to bypass Inductor.
"""

import argparse
import sys

from vllm import LLM
from vllm.lora.request import LoRARequest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Neural-ICE/Gemma-4-E2B-it-NVFP4")
    p.add_argument("--max_model_len", type=int, default=100001)
    p.add_argument("--lora", default=None)
    p.add_argument("--backend", default="FLASHINFER")
    p.add_argument("--kv_cache_dtype", default="fp8")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--generate", action="store_true")
    args = p.parse_args()

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        attention_config={"backend": args.backend},
        kv_cache_dtype=args.kv_cache_dtype,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 0},
        enforce_eager=args.eager,
    )
    if args.lora:
        kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)

    llm = LLM(**kwargs)
    print("ENGINE INIT OK", flush=True)

    if args.generate:
        from vllm import SamplingParams

        prompt = {"prompt_token_ids": list(range(1, args.max_model_len))}
        req = (
            LoRARequest("l", 1, lora_path=args.lora)
            if args.lora
            else None
        )
        llm.generate(
            prompt, SamplingParams(max_tokens=1, temperature=0.0), lora_request=req
        )
        print("GENERATE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
