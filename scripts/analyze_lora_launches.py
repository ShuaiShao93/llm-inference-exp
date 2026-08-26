"""Enumerate every LoRA kernel launch in an nsys profile, grouped by shape.

Answers two questions that a modelled byte count cannot:
  1. How many shrink/expand calls does a forward pass actually issue? If vLLM
     issues one call per module rather than one per merged projection group,
     the shrink input is re-read more times than a per-group model predicts.
  2. Is all the LoRA delta inside these kernels? Anything left over is launch
     gap, memset, or a kernel the classifier missed.

Grid size is the shape fingerprint: it encodes cdiv(M, BLOCK_M) * num_slices *
num_active_loras, so distinct layer groups land in distinct grid buckets.

Usage:
    python scripts/analyze_lora_launches.py --on prof_on.sqlite [--off prof_off.sqlite] --runs 3
"""

import argparse
import sqlite3
from collections import defaultdict

LORA_HINTS = ("lora_shrink", "lora_expand", "sgmv", "bgmv", "punica")


def load(db_path):
    conn = sqlite3.connect(db_path)
    names = {r[0]: r[1] for r in conn.execute("SELECT id, value FROM StringIds")}

    def col_exists(table, col):
        return any(r[1] == col for r in conn.execute(f"PRAGMA table_info({table})"))

    kern = []
    grid_cols = all(
        col_exists("CUPTI_ACTIVITY_KIND_KERNEL", c) for c in ("gridX", "gridY", "gridZ")
    )
    sel = "demangledName, start, end" + (", gridX, gridY, gridZ" if grid_cols else "")
    for row in conn.execute(f"SELECT {sel} FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        name = names.get(row[0], "?")
        grid = row[3] * row[4] * row[5] if grid_cols else 0
        kern.append((name, row[1], row[2], grid))

    memset = []
    try:
        for row in conn.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            memset.append(row)
    except sqlite3.OperationalError:
        pass
    return kern, memset


def is_lora(name):
    low = name.lower()
    return any(h in low for h in LORA_HINTS)


def report(label, kern, memset, runs):
    lora = [k for k in kern if is_lora(k[0])]
    total_gpu = sum(e - s for _, s, e, _ in kern) / 1e6 / runs
    lora_ms = sum(e - s for _, s, e, _ in lora) / 1e6 / runs

    print(f"\n=== {label} ===")
    print(f"all kernels: {len(kern)//runs}/run, {total_gpu:.1f} ms/run")
    print(f"LoRA kernels: {len(lora)//runs}/run, {lora_ms:.1f} ms/run")

    if not lora:
        return {}

    # Bucket by (kernel kind, grid) so each distinct layer-group shape is its own row.
    buckets = defaultdict(list)
    for name, s, e, grid in lora:
        kind = "shrink" if "shrink" in name.lower() else "expand" if "expand" in name.lower() else name
        buckets[(kind, grid)].append(e - s)

    print(f"\n{'kind':<8}{'grid':>10}{'launches/run':>14}{'mean us':>10}{'ms/run':>10}{'% LoRA':>9}")
    out = {}
    for (kind, grid), durs in sorted(buckets.items(), key=lambda kv: -sum(kv[1])):
        n = len(durs) / runs
        tot = sum(durs) / 1e6 / runs
        print(f"{kind:<8}{grid:>10}{n:>14.1f}{sum(durs)/len(durs)/1e3:>10.1f}"
              f"{tot:>10.1f}{100*tot/lora_ms:>8.1f}%")
        out[(kind, grid)] = {"launches": n, "ms": tot, "mean_us": sum(durs) / len(durs) / 1e3}

    by_kind = defaultdict(lambda: [0, 0.0])
    for (kind, _), v in out.items():
        by_kind[kind][0] += v["launches"]
        by_kind[kind][1] += v["ms"]
    print()
    for kind, (n, ms) in sorted(by_kind.items()):
        print(f"  {kind:<8} {n:6.1f} launches/run  {ms:8.1f} ms/run")

    if memset:
        mb = sum(m[2] for m in memset) / 1e6 / runs
        mms = sum(m[1] - m[0] for m in memset) / 1e6 / runs
        print(f"\nmemset: {len(memset)//runs}/run, {mb:.1f} MB/run, {mms:.2f} ms/run")

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--on", required=True)
    p.add_argument("--off")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--wall-on", type=float, help="measured ms/run with LoRA, for gap accounting")
    p.add_argument("--wall-off", type=float, help="measured ms/run without LoRA")
    args = p.parse_args()

    kern_on, ms_on = load(args.on)
    on = report("LoRA ON", kern_on, ms_on, args.runs)

    gpu_on = sum(e - s for _, s, e, _ in kern_on) / 1e6 / args.runs
    lora_ms = sum(e - s for k, s, e, _ in kern_on if is_lora(k)) / 1e6 / args.runs

    if args.off:
        kern_off, ms_off = load(args.off)
        report("LoRA OFF", kern_off, ms_off, args.runs)
        gpu_off = sum(e - s for _, s, e, _ in kern_off) / 1e6 / args.runs
        delta = gpu_on - gpu_off
        print(f"\n--- attribution ---")
        print(f"GPU time delta on-off : {delta:8.1f} ms/run")
        print(f"LoRA kernels           : {lora_ms:8.1f} ms/run  ({100*lora_ms/delta:.0f}% of delta)")
        print(f"unaccounted (other kernels grew / shrank): {delta - lora_ms:8.1f} ms/run")

    if args.wall_on and args.wall_off:
        wall_delta = args.wall_on - args.wall_off
        print(f"\nwall delta            : {wall_delta:8.1f} ms/run")
        print(f"LoRA kernels           : {lora_ms:8.1f} ms/run  ({100*lora_ms/wall_delta:.0f}% of wall delta)")
        print(f"outside LoRA kernels   : {wall_delta - lora_ms:8.1f} ms/run")


if __name__ == "__main__":
    main()
