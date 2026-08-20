"""Iterate vLLM attention backends and precisions over a list of models.

Driven by the ``vllm-backend-matrix`` skill. Produces a JSON result file that
the agent then formats into a markdown table in ``backend_compatibility.md``.

Usage:
    PYTHONNOUSERSITE=0 CUDA_HOME=/usr/local/cuda \\
    /usr/bin/python3.12 scripts/bench_vllm_backends.py \\
      --model prithivMLmods/gemma-4-E4B-it-FP8 \\
      --model RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic \\
      --backends FLASH_ATTN FLASHINFER TRITON_ATTN FLEX_ATTENTION \\
      --input_tokens 10000 100000 \\
      --output /tmp/vllm_backend_matrix.json

The script shells out to ``scripts/vllm_local.py`` once per (model, backend,
input length, precision-tier). It never runs two benchmark processes
concurrently and always captures stdout/stderr for failure diagnosis.

Precision tiers tried, in order, until one succeeds:
  Hopper / Ada (SM 8.9/9.0):  fp8 + fp8 KV  ->  fp8 + auto (BF16) KV
  Blackwell  (SM 10.x/12.x):  fp4 + fp8 KV  ->  fp4 + auto KV  ->  fp8 + fp8 KV  ->  fp8 + auto KV

If all tiers fail, the cell is recorded as a failure with the reason from the last
tier that actually ran (see ``failure_reason``) — the aggressive first tier is
usually rejected by an uninteresting dtype gate.

Input lengths are swept ascending (cheapest first). When a cell fails at a short
length for a reason that cannot depend on sequence length (a dtype or head_size
gate, a backend-selection rejection), the longer lengths are skipped and inherit
the reason — this is where most of the wall-clock saving comes from. Failures
that *are* length-sensitive (OOM, CUBLAS/illegal-address kernel crashes, missing
FlashInfer cubins, timeouts) never short-circuit, so a model that fits at 10k but
OOMs at 100k is still measured honestly at both.
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
    lora: str | None = None,
    gpu_memory_utilization: float | None = None,
) -> dict:
    """Run a single (model, backend, precision, kv) cell. Return result dict.

    Always writes a per-cell log under ``log_dir`` so the agent can diagnose
    failures without re-running.
    """
    safe_lora = re.sub(r"[^A-Za-z0-9._-]", "_", lora) if lora else "nolora"
    # input_tokens must be in the name: without it the 100k run overwrites the
    # 10k log, destroying the diagnostics for any cell that failed at 10k.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_",
                  f"{model}__{backend}__{precision}__{kv}__{input_tokens}tok__{safe_lora}")
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
    if lora:
        cmd += ["--lora", lora]
    if gpu_memory_utilization is not None:
        cmd += ["--gpu_memory_utilization", str(gpu_memory_utilization)]
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
            "length_invariant": False,
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
    #
    # Third element is `length_invariant`: True when the rejection is a static
    # dtype / head_size / capability gate that cannot change with sequence
    # length, so the caller may skip longer lengths. Keep it False for anything
    # memory- or shape-tiling-dependent (OOM, kernel crashes, cubin misses) —
    # those genuinely differ between 10k and 100k.
    patterns = [
        (r"head_size not supported", "head_size unsupported by backend", True),
        (r"kv_cache_dtype not supported", "kv_cache_dtype unsupported by backend", True),
        (r"partial multimodal token full attention not supported",
            "multimodal-prefix not supported by backend", True),
        (r"FlashInfer Internal Error: Invalid configuration",
            "FlashInfer kernel template missing for this shape", False),
        (r"AssertionError: inputs must be float16 or bfloat16",
            "FA cute kernel asserts on Q dtype (FP8 KV triggers FP8-Q quant)", True),
        (r"AssertionError: FP8 is only supported on SM\d+",
            "FA4 CuTe FP8 KV requires SM100+ (Blackwell)", True),
        (r"torch\.OutOfMemoryError: CUDA out of memory", "OOM", False),
        (r"FlexAttention does not support kv sharing",
            "FlexAttention: KV sharing not supported", True),
        (r"NotImplementedError", "not implemented", True),
        # Don't match the "FA3 does not support head_size=...: upgrading FlashAttention 3 -> 4"
        # message — that's the INFO log line right before FA4 takes over. It's not the failure.
        (r"is not valid for this configuration\. Reason: \['([^']+)'", lambda m: m.group(1), True),
        (r"RuntimeError: shape '\[[^\]]+\]' is invalid", "shape mismatch", False),
        (r"CUBLAS_STATUS_EXECUTION_FAILED",
            "CUBLAS execution failed (likely a kernel bug for this shape)", False),
        (r"cudaErrorIllegalAddress|an illegal memory access was encountered",
            "CUDA illegal memory access (kernel crash for this shape)", False),
        # A pre-quantized checkpoint only supports the precision it was built at,
        # so the lower-precision fallback tiers are unreachable for it by design.
        (r"requested precision '(\w+)' does not match model's built-in precision '(\w+)'",
            lambda m: f"checkpoint is {m.group(2)}-only (tier requested {m.group(1)})", True),
        (r"flashinfer-cubin version \([^)]+\) does not match flashinfer version",
            "flashinfer-cubin / flashinfer-python version mismatch (env bug, not a backend limit)", True),
    ]
    for pat, label, invariant in patterns:
        m = re.search(pat, text)
        if m:
            reason = label(m) if callable(label) else label
            return {"status": "fail", "reason": reason,
                    "length_invariant": invariant, "logfile": str(logfile)}
    return {"status": "fail", "reason": "unknown — see log",
            "length_invariant": False, "logfile": str(logfile)}


def run_cell(
    model: str,
    backend: str,
    tiers: list[tuple[str, str]],
    input_tokens: int,
    num_runs: int,
    timeout_s: int,
    log_dir: Path,
    tokenizer_mode: str = "auto",
    lora: str | None = None,
    gpu_memory_utilization: float | None = None,
) -> dict:
    """Try each precision tier in order until one succeeds. Return the result
    plus the (precision, kv) tuple that worked (or None if all failed)."""
    attempts = []
    builtin_precision = None
    for precision, kv in tiers:
        # A pre-quantized checkpoint rejects every precision but its own, so once
        # we've learned what it was built at, don't spend an engine init per tier
        # rediscovering that. Only the KV-dtype fallbacks remain meaningful.
        if builtin_precision is not None and precision != builtin_precision:
            attempts.append({"precision": precision, "kv": kv, "status": "fail",
                             "reason": f"skipped — checkpoint is {builtin_precision}-only",
                             "length_invariant": True})
            continue
        r = run_one(model, backend, precision, kv, input_tokens, num_runs, timeout_s, log_dir, tokenizer_mode,
                    lora=lora, gpu_memory_utilization=gpu_memory_utilization)
        attempts.append({"precision": precision, "kv": kv, **r})
        if r["status"] == "ok":
            return {"chosen": {"precision": precision, "kv": kv}, "attempts": attempts}
        m = re.match(r"checkpoint is (\w+)-only", r.get("reason", ""))
        if m:
            builtin_precision = m.group(1)
    return {"chosen": None, "attempts": attempts}


def failure_reason(cell: dict) -> str:
    """The most informative failure reason for a fully-failed cell.

    Report the *last tier that actually ran*, not the first. Tier 1 is the most
    aggressive config, so its rejection is usually the least interesting (a KV
    dtype gate); the tier that got furthest is what a reader needs. Reporting
    attempts[0] actively misleads — a cell whose fp4+auto tier OOMed at 100k
    would read as "kv_cache_dtype unsupported", implying a static incompatibility
    rather than a memory budget that a lower --gpu_memory_utilization could fix.

    Attempts rejected by the precision ladder itself are never informative: an
    fp4 checkpoint refusing an fp8 tier says nothing about the backend, so those
    are excluded whether they ran or were skipped. Otherwise a FLEX cell that
    truly OOMed at its fp4 tier reports "checkpoint is fp4-only".
    """
    informative = [a for a in cell["attempts"]
                   if not re.match(r"(skipped — )?checkpoint is \w+-only",
                                   str(a.get("reason", "")))]
    return (informative or cell["attempts"])[-1]["reason"]


def cell_is_length_invariant_failure(cell: dict) -> bool:
    """True when every precision tier failed for a sequence-length-independent
    reason, so longer input lengths are guaranteed to fail identically.

    Requires *all* attempts to be invariant: if tier 1 was rejected on a dtype
    gate but tier 2 died of OOM, a longer length could still fail differently,
    so we re-measure rather than assume.
    """
    if cell["chosen"] is not None or not cell["attempts"]:
        return False
    return all(a.get("length_invariant") for a in cell["attempts"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    help="Repeatable. HuggingFace model id (FP8 pre-quantized recommended).")
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                    help=f"Backend names. Default: {' '.join(DEFAULT_BACKENDS)}")
    ap.add_argument("--input_tokens", type=int, nargs="+", default=[10000, 100000],
                    help="One or more input lengths. Swept ascending so cheap runs come "
                         "first and length-invariant failures can short-circuit the rest.")
    ap.add_argument("--num_runs", type=int, default=3,
                    help="Timed iterations per cell. 3 is enough to stabilise the mean at "
                         "these latencies and keeps a two-length sweep affordable.")
    ap.add_argument("--no_short_circuit", action="store_true",
                    help="Measure every input length even when a shorter one failed for a "
                         "length-invariant reason. Slower; use to double-check a suspicious cell.")
    ap.add_argument("--per_cell_timeout_s", type=int, default=1800,
                    help="Per (model, backend, precision-tier) timeout. Default 30 min.")
    ap.add_argument("--output", default="/tmp/vllm_backend_matrix.json",
                    help="JSON results path. The agent reads this to write the markdown.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=None,
                    help="Pass-through to vllm_local.py. Lower it (e.g. 0.80) to retest a cell "
                         "that OOMed allocating a backend workspace — vLLM sizes the KV cache to "
                         "fill this budget, so a full budget can starve FlashInfer's workspace.")
    ap.add_argument("--log_dir", default="/tmp/vllm_backend_matrix_logs")
    ap.add_argument("--tokenizer_mode", default="auto",
                    help="Pass-through to vllm_local.py (auto/mistral/slow). Set 'mistral' for Mistral/Ministral checkpoints with only tekken.json.")
    ap.add_argument("--lora", action="append", default=[],
                    metavar="MODEL=LORA_HF_ID",
                    help="Repeatable mapping of model HF id to LoRA adapter HF id (or local path). "
                         "Adapter is loaded for every backend cell of MODEL. Models without a mapping "
                         "are benchmarked without LoRA — the JSON records `lora` per cell so the agent "
                         "can footnote no-LoRA rows in the markdown.")
    args = ap.parse_args()
    # Parse --lora into a dict
    lora_map: dict[str, str] = {}
    for kv in args.lora:
        if "=" not in kv:
            ap.error(f"--lora expects MODEL=LORA_HF_ID, got: {kv}")
        m, l = kv.split("=", 1)
        lora_map[m] = l

    # Ascending so the cheap length runs first and can short-circuit the rest.
    lengths = sorted(set(args.input_tokens))

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

    # CUDA driver version from nvidia-smi (e.g. "595.71.05") and CUDA toolkit
    # version from nvcc (e.g. "13.2"). The toolkit version is what nvcc compiles
    # against; the driver is what runs on the device. Differences here also
    # affect kernel codegen and JIT compilation paths.
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0].strip()
    except Exception:
        driver = None
    try:
        nvcc_out = subprocess.check_output(
            ["/usr/local/cuda/bin/nvcc", "--version"], text=True,
        )
        m = re.search(r"release\s+([\d.]+)", nvcc_out)
        toolkit = m.group(1) if m else None
    except Exception:
        toolkit = None
    versions["cuda-driver"] = driver
    versions["cuda-toolkit"] = toolkit

    results = {
        "gpu_name": gpu_name,
        "compute_cap": compute_cap,
        "memory": mem,
        "versions": versions,
        # Kept for backwards compatibility with prior consumers; same as versions["vllm"].
        "vllm_version": versions.get("vllm"),
        "lora_map": lora_map,
        "input_tokens": lengths,
        "num_runs": args.num_runs,
        "default_tier": default_tier,
        "fallback_tiers": tiers[1:],
        "models": args.model,
        "backends": args.backends,
        "cells": {},   # cells[model][backend][str(input_tokens)] = run_cell result
    }

    for model in args.model:
        results["cells"][model] = {}
        model_lora = lora_map.get(model)
        for backend in args.backends:
            results["cells"][model][backend] = {}
            tag = f" [lora={model_lora}]" if model_lora else ""
            invariant_fail = None
            for ntok in lengths:
                print(f"\n=== {model} × {backend} @ {ntok} tok{tag} ===", file=sys.stderr)
                if invariant_fail is not None and not args.no_short_circuit:
                    results["cells"][model][backend][str(ntok)] = {
                        "chosen": None,
                        "attempts": [],
                        "skipped": True,
                        "reason": invariant_fail,
                        "lora": model_lora,
                    }
                    Path(args.output).write_text(json.dumps(results, indent=2))
                    print(f"  -> SKIP (length-invariant failure at a shorter length): {invariant_fail}",
                          file=sys.stderr)
                    continue

                cell = run_cell(model, backend, tiers, ntok,
                                args.num_runs, args.per_cell_timeout_s, log_dir, args.tokenizer_mode,
                                lora=model_lora, gpu_memory_utilization=args.gpu_memory_utilization)
                cell["lora"] = model_lora
                results["cells"][model][backend][str(ntok)] = cell
                # Persist incrementally so a long run is recoverable
                Path(args.output).write_text(json.dumps(results, indent=2))
                chosen = cell["chosen"]
                if chosen:
                    mean_ms = next(a["mean_ms"] for a in cell["attempts"] if a["status"] == "ok")
                    ptag = "" if (chosen["precision"], chosen["kv"]) == default_tier else f" ({chosen['precision']}+{chosen['kv']}KV)"
                    print(f"  -> OK: {mean_ms:.0f} ms{ptag}", file=sys.stderr)
                else:
                    reason = failure_reason(cell)
                    print(f"  -> FAIL: {reason}", file=sys.stderr)
                    if cell_is_length_invariant_failure(cell):
                        invariant_fail = reason

    print(f"\nDone. Results: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
