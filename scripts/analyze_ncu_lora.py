"""Weight per-launch ncu DRAM bytes by real launch counts to get full-pass traffic.

ncu can only afford to profile a subset of launches, and the subset it captures is
biased by launch order -- so a straight average over captured launches is not the
per-pass number. This script buckets captured launches by launch geometry, then
multiplies each bucket by how many times that geometry actually runs per forward
pass (from nsys), giving measured traffic for the whole pass.

Expand geometry separates cleanly: gridX = cdiv(M, BLOCK_M) * cdiv(MAX_N, BLOCK_N)
and gridY = num_slices, so every distinct output width is its own bucket. Shrink
geometry does not: gridX is cdiv(M, BLOCK_M) regardless of the reduction width, so
all shrink shapes collide. Measured bytes recover the reduction width instead --
shrink moves about ``num_slices * M * K * 2``, so ``bytes / (num_slices * M * 2)``
identifies K, and launches are assigned to the nearest expected K.

Usage:
    python scripts/analyze_ncu_lora.py --csv ncu_full.csv --tokens 100000 --peak-bw 864
"""

import argparse
import csv
import io
from collections import defaultdict

BYTES = "dram__bytes.sum"
DUR = "gpu__time_duration.sum"
PCT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
L2 = "lts__t_sector_hit_rate.pct"
GX, GY, GZ = "launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z"

# Launches per forward pass, from nsys. Expand keys are (gridX, num_slices);
# shrink keys are (num_slices, reduction width K) since shrink's grid hides K.
EXPAND_COUNTS = {
    (150048, 2): 20,    # gate_up, layers 15-34 (FFN 12288)
    (75024, 2): 15,     # gate_up, layers 0-14  (FFN 6144)
    (18756, 1): 105,    # o_proj + down_proj + per_layer_projection, out 1536
    (25008, 3): 28,     # qkv, q=2048
    (50016, 3): 7,      # qkv, q=4096 (every 5th layer)
    (3126, 1): 35,      # per_layer_input_gate, out 256
    (109410, 1): 1,     # per_layer_model_projection, out 8960
}
SHRINK_COUNTS = {
    (3, 1536): 35,      # qkv
    (2, 1536): 35,      # gate_up
    (1, 2048): 28,      # o_proj, narrow attention
    (1, 4096): 7,       # o_proj, wide attention
    (1, 6144): 15,      # down_proj, layers 0-14
    (1, 12288): 20,     # down_proj, layers 15-34
    (1, 1536): 36,      # per_layer_input_gate (35) + per_layer_model_projection (1)
    (1, 256): 35,       # per_layer_projection
}


def load(path):
    raw = open(path).read().splitlines()
    try:
        start = next(i for i, l in enumerate(raw) if l.lstrip('"').startswith("ID"))
    except StopIteration:
        return {}
    launches = defaultdict(dict)
    for r in csv.DictReader(io.StringIO("\n".join(raw[start:]))):
        try:
            v = float(r["Metric Value"].replace(",", ""))
        except (ValueError, KeyError, TypeError):
            continue
        name = r.get("Kernel Name", "")
        kind = "shrink" if "shrink" in name else "expand" if "expand" in name else name
        launches[(r["ID"], kind)][r["Metric Name"]] = v
    return launches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--tokens", type=int, default=100000)
    p.add_argument("--peak-bw", type=float, default=864.0, help="GB/s")
    args = p.parse_args()

    launches = load(args.csv)
    if not launches:
        print("no launches parsed - check the CSV header detection")
        return
    M = args.tokens

    buckets = defaultdict(list)
    for (_, kind), m in launches.items():
        if BYTES not in m or DUR not in m:
            continue
        gx, gy = int(m.get(GX, 0)), int(m.get(GY, 0))
        if kind == "shrink":
            # Recover K from measured bytes, then snap to the nearest expected K.
            implied = m[BYTES] / (gy * M * 2)
            ks = [k for ns, k in SHRINK_COUNTS if ns == gy]
            key = (kind, gy, min(ks, key=lambda k: abs(k - implied)) if ks else 0)
        else:
            key = (kind, gy, gx)
        buckets[key].append(m)

    print(f"{'kind':<7}{'sl':>3}{'K or gridX':>11}{'seen':>5}{'n/pass':>7}"
          f"{'MB ea':>8}{'ms ea':>8}{'GB/s':>6}{'DRAM%':>7}{'SM%':>6}{'L2%':>6}{'GB/pass':>9}")
    tot_gb = tot_ms = 0.0
    missing = []
    for (kind, gy, key), ms in sorted(buckets.items(), key=lambda kv: kv[0]):
        n = len(ms)
        mb = sum(x[BYTES] for x in ms) / n / 1e6
        dur = sum(x[DUR] for x in ms) / n / 1e6            # ns -> ms
        cnt = (SHRINK_COUNTS if kind == "shrink" else EXPAND_COUNTS).get(
            (gy, key) if kind == "shrink" else (key, gy), 0)
        if not cnt:
            missing.append((kind, gy, key, n))
        gb = mb * cnt / 1e3
        tot_gb += gb
        tot_ms += dur * cnt
        print(f"{kind:<7}{gy:>3}{key:>11}{n:>5}{cnt:>7}{mb:>8.1f}{dur:>8.2f}"
              f"{(mb/1e3)/(dur/1e3):>6.0f}{sum(x.get(PCT,0) for x in ms)/n:>7.1f}"
              f"{sum(x.get(SM,0) for x in ms)/n:>6.1f}"
              f"{sum(x.get(L2,0) for x in ms)/n:>6.1f}{gb:>9.1f}")

    covered = sum(
        c for k, c in list(SHRINK_COUNTS.items())
        if ("shrink", k[0], k[1]) in buckets
    ) + sum(
        c for k, c in list(EXPAND_COUNTS.items())
        if ("expand", k[1], k[0]) in buckets
    )
    total_launches = sum(SHRINK_COUNTS.values()) + sum(EXPAND_COUNTS.values())
    print(f"\nattributed by shape: {covered}/{total_launches} launches per pass")
    if covered < total_launches:
        print("  uncovered buckets are excluded from the shape-weighted total")
    if missing:
        print(f"  unexpected geometries (no launch count): {missing}")
    print(f"  shape-weighted: {tot_gb:.1f} GB over {tot_ms:.0f} ms "
          f"= {tot_gb/(tot_ms/1e3):.0f} GB/s "
          f"= {100*tot_gb/(tot_ms/1e3)/args.peak_bw:.1f}% of {args.peak_bw:.0f} GB/s peak")

    # Cross-check that needs no shape attribution. ncu captures a contiguous
    # window of launches, so its composition already matches a forward pass; just
    # rescale the captured totals to one pass. Immune to byte-snapping mistakes,
    # which matters because shrink's grid hides K and some shapes measure alike.
    seen_gb = sum(x[BYTES] for ms in buckets.values() for x in ms) / 1e9
    seen_ms = sum(x[DUR] for ms in buckets.values() for x in ms) / 1e6
    seen_n = sum(len(ms) for ms in buckets.values())
    scale = total_launches / seen_n
    print(f"\ncaptured {seen_n} launches = {seen_n/total_launches:.2f} passes")
    print(f"  rescaled to one pass: {seen_gb*scale:.1f} GB over {seen_ms*scale:.0f} ms "
          f"= {seen_gb/(seen_ms/1e3):.0f} GB/s "
          f"= {100*seen_gb/(seen_ms/1e3)/args.peak_bw:.1f}% of {args.peak_bw:.0f} GB/s peak")


if __name__ == "__main__":
    main()
