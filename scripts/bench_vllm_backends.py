"""Iterate vLLM attention backends and precisions over a list of models.

Driven by the ``vllm-backend-matrix`` skill. Produces a JSON result file that
the agent then formats into a markdown table in ``backend_compatibility.md``.

Usage:
    PYTHONNOUSERSITE=0 CUDA_HOME=/usr/local/cuda \\
    /usr/bin/python3.12 scripts/bench_vllm_backends.py \\
      --model prithivMLmods/gemma-4-E4B-it-FP8 \\
      --model RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic \\
      --backends FLASH_ATTN FLASHINFER TRITON_ATTN FLEX_ATTENTION \\
      --output /tmp/vllm_backend_matrix.json

The script shells out to ``scripts/vllm_local.py`` once per (model, backend,
precision-tier). It never runs two benchmark processes concurrently and always
captures stdout/stderr for failure diagnosis.

Precision tiers tried, in order, until one succeeds:
  Hopper / Ada (SM 8.9/9.0):  fp8 + fp8 KV  ->  fp8 + auto (BF16) KV
  Blackwell  (SM 10.x/12.x):  fp4 + fp8 KV  ->  fp4 + auto KV  ->  fp8 + fp8 KV  ->  fp8 + auto KV

If all tiers fail, the cell is recorded as a failure with the first error reason.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VLLM_LOCAL = Path(__file__).parent / "vllm_local.py"

# Backends to try by default. The script tolerates unknown backends — vllm itself
# will reject them at engine init with a clean "head_size not supported" /
# "kv_cache_dtype not supported" / "not valid for this configuration" message.
DEFAULT_BACKENDS = [
    "FLASH_ATTN",
    "FLASHINFER",
    "TRITON_ATTN",
    "FLEX_ATTENTION",
]


def detect_precision_tiers(compute_cap: str) -> list[tuple[str, str]]:
    """Return ordered (precision, kv_cache_precision) tiers for this GPU.

    First success wins. The default cell precision for a row in the matrix is
    the first entry — any cell that ends up using a later entry is annotated.
    """
    cc = float(compute_cap)  # "9.0" -> 9.0
    if cc >= 10.0:
        # Blackwell — native FP4 tensor cores. Per CLAUDE.md: SM120 has no
        # FP4-KV FMHA, so FP8 KV cache is the floor regardless of SM.
        return [
            ("fp4", "fp8"),
            ("fp4", "auto"),
            ("fp8", "fp8"),
            ("fp8", "auto"),
        ]
    if cc >= 8.9:
        # Hopper / Ada — native FP8 tensor cores.
        return [
            ("fp8", "fp8"),
            ("fp8", "auto"),
        ]
    # Ampere (SM 8.0/8.6) — no native FP8/FP4 tensor cores, but INT8 tensor
    # cores are available (added in Turing SM 7.5). W8A8 INT8 is the natural
    # quantized default; "auto" KV picks the model's compute dtype (typ. BF16).
    return [
        ("int8", "auto"),
    ]


def run_one(
    model: str,
    backend: str,
    precision: str,
    kv: str,
    input_tokens: int,
    num_runs: int,
    timeout_s: int,
    log_dir: Path,
    tokenizer_mode: str = "auto",
) -> dict:
    """Run a single (model, backend, precision, kv) cell. Return result dict.

    Always writes a per-cell log under ``log_dir`` so the agent can diagnose
    failures without re-running.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", f"{model}__{backend}__{precision}__{kv}")
    logfile = log_dir / f"{safe}.log"
    cmd = [
        sys.executable,
        str(VLLM_LOCAL),
        "--model", model,
        "--precision", precision,
        "--kv_cache_precision", kv,
        "--attention_backend", backend,
        "--input_tokens", str(input_tokens),
        "--max_output_tokens", "1",
        "--num_runs", str(num_runs),
        "--tokenizer_mode", tokenizer_mode,
    ]
    env = {**os.environ, "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda")}
    start = time.time()
    try:
        with open(logfile, "wb") as f:
            proc = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s, env=env
            )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "reason": f"exceeded {timeout_s}s",
            "logfile": str(logfile),
            "wall_s": time.time() - start,
        }

    text = logfile.read_text(errors="replace")

    if proc.returncode == 0:
        m = re.search(r"Mean latency:\s+([\d.]+) ms", text)
        if m:
            return {
                "status": "ok",
                "mean_ms": float(m.group(1)),
                "logfile": str(logfile),
                "wall_s": elapsed,
            }
        return {
            "status": "fail",
            "reason": "no Mean latency in output",
            "logfile": str(logfile),
        }

    # Classify failure reasons from common patterns.
    patterns = [
        (r"head_size not supported", "head_size unsupported by backend"),
        (r"kv_cache_dtype not supported", "kv_cache_dtype unsupported by backend"),
        (r"partial multimodal token full attention not supported",
            "multimodal-prefix not supported by backend"),
        (r"FlashInfer Internal Error: Invalid configuration", "FlashInfer kernel template missing for this shape"),
        (r"AssertionError: inputs must be float16 or bfloat16",
            "FA cute kernel asserts on Q dtype (FP8 KV triggers FP8-Q quant)"),
        (r"torch\.OutOfMemoryError: CUDA out of memory", "OOM"),
        (r"FlexAttention does not support kv sharing", "FlexAttention: KV sharing not supported"),
        (r"NotImplementedError", "not implemented"),
        (r"head_size=\d+ on SM\d+: upgrading FlashAttention", "FA3 head_size unsupported on this SM; FA4 fallback may have its own issues"),
        (r"is not valid for this configuration\. Reason: \['([^']+)'", lambda m: m.group(1)),
        (r"RuntimeError: shape '\[[^\]]+\]' is invalid", "shape mismatch"),
    ]
    for pat, label in patterns:
        m = re.search(pat, text)
        if m:
            reason = label(m) if callable(label) else label
            return {"status": "fail", "reason": reason, "logfile": str(logfile)}
    return {"status": "fail", "reason": "unknown — see log", "logfile": str(logfile)}


def run_cell(
    model: str,
    backend: str,
    tiers: list[tuple[str, str]],
    input_tokens: int,
    num_runs: int,
    timeout_s: int,
    log_dir: Path,
    tokenizer_mode: str = "auto",
) -> dict:
    """Try each precision tier in order until one succeeds. Return the result
    plus the (precision, kv) tuple that worked (or None if all failed)."""
    attempts = []
    for precision, kv in tiers:
        r = run_one(model, backend, precision, kv, input_tokens, num_runs, timeout_s, log_dir, tokenizer_mode)
        attempts.append({"precision": precision, "kv": kv, **r})
        if r["status"] == "ok":
            return {"chosen": {"precision": precision, "kv": kv}, "attempts": attempts}
    return {"chosen": None, "attempts": attempts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    help="Repeatable. HuggingFace model id (FP8 pre-quantized recommended).")
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                    help=f"Backend names. Default: {' '.join(DEFAULT_BACKENDS)}")
    ap.add_argument("--input_tokens", type=int, default=100000)
    ap.add_argument("--num_runs", type=int, default=5)
    ap.add_argument("--per_cell_timeout_s", type=int, default=1800,
                    help="Per (model, backend, precision-tier) timeout. Default 30 min.")
    ap.add_argument("--output", default="/tmp/vllm_backend_matrix.json",
                    help="JSON results path. The agent reads this to write the markdown.")
    ap.add_argument("--log_dir", default="/tmp/vllm_backend_matrix_logs")
    ap.add_argument("--tokenizer_mode", default="auto",
                    help="Pass-through to vllm_local.py (auto/mistral/slow). Set 'mistral' for Mistral/Ministral checkpoints with only tekken.json.")
    args = ap.parse_args()

    # GPU info
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    gpu_name, compute_cap, mem = [s.strip() for s in gpu.split(",")]
    tiers = detect_precision_tiers(compute_cap)
    default_tier = tiers[0]
    print(f"GPU: {gpu_name} (SM {compute_cap}, {mem})", file=sys.stderr)
    print(f"Default precision tier: {default_tier}", file=sys.stderr)
    print(f"Fallback tiers (in order): {tiers[1:]}", file=sys.stderr)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Capture versions of every package that materially affects attention kernel
    # selection or compilation. Matrix consumers diff these against current to
    # decide whether to re-measure. flash-attn is vendored inside vllm
    # (vllm/vllm_flash_attn/) so it has no independent version — it tracks vllm.
    import importlib.metadata as md
    versions = {}
    for pkg in ["vllm", "flashinfer-python", "flashinfer-cubin", "triton"]:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None

    results = {
        "gpu_name": gpu_name,
        "compute_cap": compute_cap,
        "memory": mem,
        "versions": versions,
        # Kept for backwards compatibility with prior consumers; same as versions["vllm"].
        "vllm_version": versions.get("vllm"),
        "input_tokens": args.input_tokens,
        "num_runs": args.num_runs,
        "default_tier": default_tier,
        "fallback_tiers": tiers[1:],
        "models": args.model,
        "backends": args.backends,
        "cells": {},   # cells[model][backend] = run_cell result
    }

    for model in args.model:
        results["cells"][model] = {}
        for backend in args.backends:
            print(f"\n=== {model} × {backend} ===", file=sys.stderr)
            cell = run_cell(model, backend, tiers, args.input_tokens,
                            args.num_runs, args.per_cell_timeout_s, log_dir, args.tokenizer_mode)
            results["cells"][model][backend] = cell
            # Persist incrementally so a long run is recoverable
            Path(args.output).write_text(json.dumps(results, indent=2))
            chosen = cell["chosen"]
            if chosen:
                mean_ms = next(a["mean_ms"] for a in cell["attempts"] if a["status"] == "ok")
                tag = "" if (chosen["precision"], chosen["kv"]) == default_tier else f" ({chosen['precision']}+{chosen['kv']}KV)"
                print(f"  -> OK: {mean_ms:.0f} ms{tag}", file=sys.stderr)
            else:
                reason = cell["attempts"][0]["reason"]
                print(f"  -> FAIL: {reason}", file=sys.stderr)

    print(f"\nDone. Results: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
