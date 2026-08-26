"""Microbenchmark vLLM's Triton punica LoRA kernels across tile configs.

Motivation: an end-to-end A/B showed LoRA adds ~11.5 ms per 1k prompt tokens on
a single r=16 adapter, which is far more than a bandwidth-bound estimate of the
shrink/expand math. ``get_lora_op_configs`` picks its default tile config from a
two-regime heuristic (``batch < 128`` vs ``batch >= 128``), so a 100k-token
prefill is tiled exactly like a 128-token batch. This script isolates the two
kernels from the model so that claim can be measured directly.

For each token count it times the shipped default config against a sweep of
candidate configs, and reports the speedup left on the table. Scale the
per-call numbers by the adapter's module count to compare against end-to-end
latency (an r=16 adapter with 205 adapted modules issues 205 shrink + 205
expand calls per forward pass).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_punica_configs.py \\
      --hidden-size 2048 --rank 16 --num-slices 3 \\
      --tokens 1000 10000 50000 100000 \\
      --output /tmp/punica_configs.json

``--num-slices`` mirrors how vLLM fuses projections into one call: 3 for a
merged QKV, 2 for gate/up, 1 for o_proj and down_proj.
"""

import argparse
import contextlib
import json
import statistics

import torch

from vllm.lora.ops.triton_ops import lora_expand_op, lora_shrink_op
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs

# (block_m, block_k, split_k, num_warps, num_stages) for shrink. split_k=1
# removes the atomic accumulation and the output memset it forces; large
# block_m is what a 100k-row GEMM actually wants.
SHRINK_CANDIDATES = [
    (32, 32, 8, 4, 2),  # the shipped default for batch >= 128
    (32, 32, 1, 4, 2),
    (64, 32, 1, 4, 2),
    (64, 64, 1, 4, 2),
    (128, 32, 1, 4, 2),
    (128, 64, 1, 4, 2),
    (256, 32, 1, 4, 2),
    (256, 64, 1, 4, 2),
    (128, 128, 1, 4, 2),
    (64, 32, 2, 4, 2),
    (128, 32, 2, 4, 2),
    # The 3-tuple sweep's winners clustered at block_k>=64 with split_k=1, on
    # the edge of that grid; push past it on warps and pipeline depth too.
    (256, 128, 1, 4, 2),
    (256, 64, 1, 8, 2),
    (256, 64, 1, 8, 3),
    (256, 64, 1, 4, 3),
    (512, 64, 1, 8, 2),
    (128, 64, 1, 8, 3),
]

# (block_m, block_n, block_k, num_warps, num_stages) for expand. No split_k.
EXPAND_CANDIDATES = [
    (64, 64, 32, 4, 2),  # the shipped default when num_slices > 1
    (64, 128, 32, 4, 2),
    (128, 64, 32, 4, 2),
    (128, 128, 32, 4, 2),
    (256, 64, 32, 4, 2),
    (128, 64, 16, 4, 2),
    (256, 128, 32, 4, 2),
    # m128_n64_k16 won every shape in the 3-tuple sweep while sitting on the
    # grid's boundary (smallest block_k, smaller block_n), so the optimum was
    # probably outside it. Extend along every axis that boundary touches.
    (128, 32, 16, 4, 2),
    (64, 64, 16, 4, 2),
    (256, 64, 16, 4, 2),
    (128, 64, 16, 8, 2),
    (128, 64, 16, 4, 3),
    (128, 64, 16, 8, 3),
    (128, 64, 16, 8, 4),
    (256, 64, 16, 8, 3),
    (128, 32, 16, 8, 3),
]


def build_metadata(num_tokens: int, device: torch.device):
    """Punica metadata for the simplest possible case: one adapter, all tokens."""
    return {
        "token_lora_mapping": torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        ),
        "token_indices_sorted_by_lora_ids": torch.arange(
            num_tokens, dtype=torch.int32, device=device
        ),
        "num_tokens_per_lora": torch.tensor(
            [num_tokens, 0], dtype=torch.int32, device=device
        ),
        "lora_token_start_loc": torch.tensor(
            [0, num_tokens, num_tokens], dtype=torch.int32, device=device
        ),
        "lora_ids": torch.tensor([0, -1], dtype=torch.int32, device=device),
        "no_lora_flag_cpu": torch.tensor([False], dtype=torch.bool, device="cpu"),
        "num_active_loras": torch.tensor([1], dtype=torch.int32, device="cpu"),
    }


@contextlib.contextmanager
def forced_config(module, config: dict):
    """Pin the tile config a kernel wrapper sees.

    ``get_lora_op_configs`` is lru_cached and imported into each op module's
    namespace, so patching the wrapper's own reference is both necessary and
    sufficient.
    """
    original = module.get_lora_op_configs
    module.get_lora_op_configs = lambda *a, **kw: config
    try:
        yield
    finally:
        module.get_lora_op_configs = original


def time_op(fn, warmup: int, iters: int) -> float:
    """Median milliseconds per call, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def bench_shrink(num_tokens, hidden, rank, num_slices, device, dtype, args):
    inputs = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    weights = [
        torch.randn(1, rank, hidden, dtype=dtype, device=device)
        for _ in range(num_slices)
    ]
    out = torch.zeros(num_slices, num_tokens, rank, dtype=torch.float32, device=device)
    meta = build_metadata(num_tokens, device)

    def call():
        lora_shrink_op._lora_shrink(
            inputs, weights, out, scaling=1.0, **meta
        )

    results = {}
    for block_m, block_k, split_k, num_warps, num_stages in SHRINK_CANDIDATES:
        config = {
            "block_m": block_m,
            "block_n": max(16, rank),
            "block_k": block_k,
            "split_k": split_k,
            "num_warps": num_warps,
            "num_ctas": 1,
            "group_size_m": 8,
            "num_stages": num_stages,
            "max_nreg": None,
        }
        key = f"m{block_m}_k{block_k}_sk{split_k}_w{num_warps}_s{num_stages}"
        with forced_config(lora_shrink_op, config):
            try:
                results[key] = time_op(call, args.warmup, args.iters)
            except Exception as exc:  # a tile config can exceed shared memory
                results[key] = f"FAIL: {type(exc).__name__}"
    return results


def bench_expand(num_tokens, hidden, rank, num_slices, device, dtype, args):
    inputs = torch.randn(
        num_slices, num_tokens, rank, dtype=torch.float32, device=device
    )
    weights = [
        torch.randn(1, hidden, rank, dtype=dtype, device=device)
        for _ in range(num_slices)
    ]
    out = torch.zeros(num_tokens, hidden * num_slices, dtype=dtype, device=device)
    meta = build_metadata(num_tokens, device)

    def call():
        lora_expand_op._lora_expand(
            inputs, weights, out, offset_start=0, add_inputs=True, **meta
        )

    results = {}
    for block_m, block_n, block_k, num_warps, num_stages in EXPAND_CANDIDATES:
        config = {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_warps": num_warps,
            "num_ctas": 1,
            "num_stages": num_stages,
            "max_nreg": None,
        }
        key = f"m{block_m}_n{block_n}_k{block_k}_w{num_warps}_s{num_stages}"
        with forced_config(lora_expand_op, config):
            try:
                results[key] = time_op(call, args.warmup, args.iters)
            except Exception as exc:
                results[key] = f"FAIL: {type(exc).__name__}"
    return results


def emit_tuned_configs(out_dir, gpu_name, best, hidden, rank, num_slices, max_loras=2):
    """Write configs in the layout ``VLLM_TUNED_CONFIG_FOLDER`` expects.

    vLLM ships no tuned configs, so ``get_lora_op_configs`` always falls back to
    its hardcoded default. Pointing that env var at this directory is the
    supported way to A/B a better config end-to-end without patching vLLM.

    Layout is ``config[max_loras][num_slices][m][k][n]``, and every level falls
    back to the numerically nearest key, so a few m anchors cover all lengths.
    Note the k/n axes swap between the two ops: shrink is (hidden, rank),
    expand is (rank, hidden).
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    slug = gpu_name.replace(" ", "_").replace("-", "_")

    def nest(entries, k, n):
        m_level = {}
        for num_tokens, config in entries.items():
            m_level[str(num_tokens)] = {str(k): {str(n): config}}
        return {str(max_loras): {str(num_slices): m_level}}

    written = []
    shrink_doc = nest(
        {m: c["shrink"] for m, c in best.items()}, hidden, rank
    )
    path = f"{out_dir}/{slug}_SHRINK.json"
    with open(path, "w") as f:
        json.dump(shrink_doc, f, indent=2)
    written.append(path)

    # add_inputs is True on the real expand path (LoRA accumulates into the
    # base projection output), so only the TRUE variant is consulted.
    expand_doc = nest({m: c["expand"] for m, c in best.items()}, rank, hidden)
    path = f"{out_dir}/{slug}_EXPAND_TRUE.json"
    with open(path, "w") as f:
        json.dump(expand_doc, f, indent=2)
    written.append(path)

    print("\nwrote tuned configs:")
    for p in written:
        print(f"  {p}")
    print(f"use with: VLLM_TUNED_CONFIG_FOLDER={out_dir}")


BASE_FIELDS = {
    "num_ctas": 1,
    "group_size_m": 8,
    "max_nreg": None,
}


def parse_shrink_key(key, rank):
    """'m128_k64_sk1_w4_s2' -> full shrink config dict."""
    m, k, sk, w, s = key.split("_")
    return {
        "block_m": int(m[1:]),
        "block_n": max(16, rank),
        "block_k": int(k[1:]),
        "split_k": int(sk[2:]),
        "num_warps": int(w[1:]),
        "num_stages": int(s[1:]),
        **BASE_FIELDS,
    }


def parse_expand_key(key):
    """'m64_n128_k32_w4_s2' -> full expand config dict."""
    m, n, k, w, s = key.split("_")
    return {
        "block_m": int(m[1:]),
        "block_n": int(n[1:]),
        "block_k": int(k[1:]),
        "num_warps": int(w[1:]),
        "num_stages": int(s[1:]),
        **BASE_FIELDS,
    }


def summarize(label, results, default_key):
    numeric = {k: v for k, v in results.items() if isinstance(v, float)}
    default = numeric.get(default_key)
    best_key = min(numeric, key=numeric.get)
    best = numeric[best_key]
    print(f"  {label}:")
    for key, val in sorted(numeric.items(), key=lambda kv: kv[1]):
        marks = []
        if key == default_key:
            marks.append("<- shipped default")
        if key == best_key:
            marks.append("<- best")
        print(f"    {key:<24} {val:8.3f} ms  {' '.join(marks)}")
    if default is not None and best > 0:
        print(f"    default/best = {default / best:.2f}x")
    return {"default": default, "best": best, "best_config": best_key}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1000, 10000, 50000, 100000])
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--num-slices", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--modules",
        type=int,
        default=205,
        help="adapted module count, used to scale per-call times to a full forward pass",
    )
    p.add_argument("--output")
    p.add_argument(
        "--emit-config-dir",
        help="write the winning configs here in VLLM_TUNED_CONFIG_FOLDER layout",
    )
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"hidden={args.hidden_size} rank={args.rank} num_slices={args.num_slices} "
        f"dtype={args.dtype}"
    )
    # Confirms we are actually measuring the untuned path.
    print(f"default shrink config @100k: {get_lora_op_configs('shrink', 2, 100000, args.hidden_size, args.rank, args.num_slices)}")

    out = {
        "gpu": torch.cuda.get_device_name(),
        "hidden_size": args.hidden_size,
        "rank": args.rank,
        "num_slices": args.num_slices,
        "dtype": args.dtype,
        "modules": args.modules,
        "by_tokens": {},
    }

    best_configs = {}
    for num_tokens in args.tokens:
        print(f"\n=== {num_tokens} tokens ===")
        shrink = bench_shrink(
            num_tokens, args.hidden_size, args.rank, args.num_slices, device, dtype, args
        )
        expand = bench_expand(
            num_tokens, args.hidden_size, args.rank, args.num_slices, device, dtype, args
        )
        s = summarize("shrink", shrink, "m32_k32_sk8_w4_s2")
        e = summarize("expand", expand, "m64_n64_k32_w4_s2")
        best_configs[num_tokens] = {
            "shrink": parse_shrink_key(s["best_config"], args.rank),
            "expand": parse_expand_key(e["best_config"]),
        }

        # num_slices projections are fused into one call, so a module group of
        # this shape costs one shrink + one expand.
        groups = max(1, args.modules // args.num_slices)
        if s["default"] and e["default"]:
            cur = (s["default"] + e["default"]) * groups
            opt = (s["best"] + e["best"]) * groups
            print(f"  scaled to {groups} module groups: {cur:.1f} ms -> {opt:.1f} ms")
            out["by_tokens"][num_tokens] = {
                "shrink": shrink,
                "expand": expand,
                "scaled_default_ms": cur,
                "scaled_best_ms": opt,
            }
        else:
            out["by_tokens"][num_tokens] = {"shrink": shrink, "expand": expand}

    if args.emit_config_dir:
        emit_tuned_configs(
            args.emit_config_dir,
            torch.cuda.get_device_name(),
            best_configs,
            args.hidden_size,
            args.rank,
            args.num_slices,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
