"""Attribute the LoRA latency delta to specific GPU kernels from nsys reports.

Takes the two ``.sqlite`` files nsys emits for a LoRA-off / LoRA-on pair of runs
and reports, per kernel category, how much GPU time each contributes per run and
what the on-minus-off delta is. The point is to separate three candidate causes
of LoRA overhead that an end-to-end number cannot distinguish:

  1. the punica shrink/expand GEMMs themselves,
  2. the output memsets the shrink kernel's split-K accumulation forces,
  3. everything else moving (e.g. the base model falling off a fused path, or
     losing CUDA graph capture, which shows up as more launches of the *same*
     kernels rather than new ones).

Usage:
    python scripts/analyze_lora_profile.py \\
      --off ~/profiles/e2b_loraoff_100k.sqlite \\
      --on  ~/profiles/e2b_loraon_100k.sqlite \\
      --runs 3
"""

import argparse
import sqlite3
from collections import defaultdict

CATEGORIES = [
    ("lora_shrink", ("lora_shrink", "_lora_shrink_kernel")),
    ("lora_expand", ("lora_expand", "_lora_expand_kernel")),
    ("lora_moe", ("fused_moe_lora",)),
    (
        "attention",
        (
            "fmha",
            "flash_attention",
            "flash_fwd",
            "batchprefill",
            "batchdecode",
            "kernel_mha",
            "mmha",
            "merge_states",
        ),
    ),
    ("gemm", ("gemm", "cutlass", "gemv", "matmul")),
    ("norm/quant", ("norm", "quantize", "cvt_fp", "fp16_to_fp4", "block_scale", "scaled_mm")),
    ("activation", ("silu", "gelu", "_act")),
    ("elementwise/copy", ("elementwise", "vectorized_", "copy_", "cat_", "index_")),
]


def classify(name: str) -> str:
    low = name.lower()
    for label, keys in CATEGORIES:
        if any(k in low for k in keys):
            return label
    return "other"


def collect(db_path: str, runs: int):
    """Per-run GPU time and launch count per category, plus memset totals."""
    conn = sqlite3.connect(db_path)
    names = {r[0]: r[1] for r in conn.execute("SELECT id, value FROM StringIds")}

    time_by_cat = defaultdict(float)
    count_by_cat = defaultdict(int)
    per_kernel = defaultdict(lambda: [0.0, 0])

    # demangledName can be null for some kernels; fall back to shortName.
    rows = conn.execute(
        "SELECT coalesce(demangledName, shortName), end - start "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    for name_id, dur in rows:
        name = names.get(name_id, f"<{name_id}>")
        cat = classify(name)
        time_by_cat[cat] += dur
        count_by_cat[cat] += 1
        per_kernel[name][0] += dur
        per_kernel[name][1] += 1

    # Memsets are a separate CUPTI table; the shrink kernel's split-K path
    # forces an output zero_() that lands here, not in the kernel table.
    try:
        memset = conn.execute(
            "SELECT count(*), sum(end - start) FROM CUPTI_ACTIVITY_KIND_MEMSET"
        ).fetchone()
        if memset and memset[1]:
            time_by_cat["memset"] += memset[1]
            count_by_cat["memset"] += memset[0]
    except sqlite3.OperationalError:
        pass

    scale = 1e6 * runs  # ns -> ms, per run
    return (
        {k: v / scale for k, v in time_by_cat.items()},
        {k: v / runs for k, v in count_by_cat.items()},
        {k: (v[0] / scale, v[1] / runs) for k, v in per_kernel.items()},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--off", required=True, help="nsys .sqlite for the LoRA-off run")
    p.add_argument("--on", required=True, help="nsys .sqlite for the LoRA-on run")
    p.add_argument("--runs", type=int, default=3, help="timed runs captured in each")
    p.add_argument("--top", type=int, default=12, help="top individual kernels to list")
    args = p.parse_args()

    off_t, off_c, off_k = collect(args.off, args.runs)
    on_t, on_c, on_k = collect(args.on, args.runs)

    cats = sorted(
        set(off_t) | set(on_t),
        key=lambda c: -(on_t.get(c, 0) - off_t.get(c, 0)),
    )

    print(f"{'category':<18} {'off ms':>9} {'on ms':>9} {'delta ms':>9} "
          f"{'off n':>8} {'on n':>8}")
    print("-" * 68)
    for c in cats:
        o, n = off_t.get(c, 0.0), on_t.get(c, 0.0)
        print(f"{c:<18} {o:>9.2f} {n:>9.2f} {n - o:>+9.2f} "
              f"{off_c.get(c, 0):>8.0f} {on_c.get(c, 0):>8.0f}")
    print("-" * 68)
    ot, nt = sum(off_t.values()), sum(on_t.values())
    print(f"{'TOTAL GPU':<18} {ot:>9.2f} {nt:>9.2f} {nt - ot:>+9.2f} "
          f"{sum(off_c.values()):>8.0f} {sum(on_c.values()):>8.0f}")

    print(f"\nTop {args.top} kernels by on-minus-off delta (ms/run):")
    deltas = []
    for name in set(off_k) | set(on_k):
        o = off_k.get(name, (0.0, 0))
        n = on_k.get(name, (0.0, 0))
        deltas.append((n[0] - o[0], name, o, n))
    deltas.sort(reverse=True)
    for d, name, o, n in deltas[: args.top]:
        short = name if len(name) <= 62 else name[:59] + "..."
        print(f"  {d:>+8.2f}  n {o[1]:>6.0f}->{n[1]:<6.0f}  {short}")


if __name__ == "__main__":
    main()
