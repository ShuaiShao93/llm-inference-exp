"""Model vLLM's punica shrink/expand DRAM traffic from the real per-layer shapes.

A uniform-shape model of Gemma 4 E2B undercounts LoRA traffic by ~1.8x, because
the stack is heterogeneous in two ways that a single `intermediate_size` /
`num_attention_heads` reading hides:

  * layers 15-34 have a 2x wider FFN than layers 0-14
  * every 5th layer has 2x the attention width (full attention vs the rest)

and because vLLM issues LoRA for six module groups per layer, not four. The two
extra groups are Gemma's per-layer-input machinery, which the adapter does not
target at all -- vLLM allocates zero weights for them and runs the kernels anyway.

Traffic per group follows the launch geometry, not the algorithm:
  shrink  grid=(cdiv(M,BLOCK_M), num_slices, loras) -- each slice program re-reads
          the whole input tile, so the input is read num_slices times.
  expand  grid=(cdiv(M,BLOCK_M)*cdiv(MAX_N,BLOCK_N), num_slices, loras) -- the
          intermediate is read once per slice and the output is read-modify-written
          because add_inputs=True accumulates onto the base projection.

Usage:
    python scripts/model_lora_traffic.py --tokens 100000 [--peak-bw 864 --ms 950]
"""

import argparse

HIDDEN = 1536
LAYERS = 35
WIDE_FFN_FROM = 15          # layers >= this have the 2x FFN
NARROW_FFN, WIDE_FFN = 6144, 12288
FULL_ATTN_EVERY = 5         # layers 4, 9, 14, ... have 2x attention width
Q_NARROW, KV_NARROW = 2048, 256
KV_LORA_LAYERS = 15         # adapter carries k/v LoRA only on layers 0-14
PER_LAYER_INPUT = 256       # hidden_size_per_layer_input


def groups(layer):
    """(name, shrink_K, [expand slice widths], targeted_by_adapter) per group."""
    wide_attn = (layer + 1) % FULL_ATTN_EVERY == 0
    q = Q_NARROW * (2 if wide_attn else 1)
    kv = KV_NARROW * (2 if wide_attn else 1)
    ffn = WIDE_FFN if layer >= WIDE_FFN_FROM else NARROW_FFN
    return [
        ("qkv", HIDDEN, [q, kv, kv], True),
        ("o_proj", q, [HIDDEN], True),
        ("gate_up", HIDDEN, [ffn, ffn], True),
        ("down_proj", ffn, [HIDDEN], True),
        # Gemma per-layer-input machinery: present in the model, absent from the
        # adapter, still launched with zero weights.
        ("per_layer_input_gate", HIDDEN, [PER_LAYER_INPUT], False),
        ("per_layer_projection", PER_LAYER_INPUT, [HIDDEN], False),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=100000)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--peak-bw", type=float, default=864.0, help="GB/s")
    p.add_argument("--ms", type=float, help="measured LoRA kernel ms/run, for utilization")
    args = p.parse_args()

    M, r = args.tokens, args.rank
    rows, launches = {}, {"shrink": 0, "expand": 0}
    waste = {"untargeted": 0, "dead_kv_slices": 0, "shrink_reread": 0}

    def add(name, k_in, slices, targeted, layer=None):
        ns = len(slices)
        # shrink: input read once per slice, rank-r intermediate written in fp32
        s_in = ns * M * k_in * 2
        s_bytes = s_in + ns * M * r * 4
        # expand: intermediate read once per slice, output read-modify-written
        e_bytes = ns * M * r * 4 + 2 * M * sum(slices) * 2
        d = rows.setdefault(name, {"n": 0, "shrink": 0, "expand": 0})
        d["n"] += 1
        d["shrink"] += s_bytes
        d["expand"] += e_bytes
        launches["shrink"] += 1
        launches["expand"] += 1
        if not targeted:
            waste["untargeted"] += s_bytes + e_bytes
        # k/v slices past the adapter's coverage carry zero weights but full traffic
        if name == "qkv" and layer is not None and layer >= KV_LORA_LAYERS:
            waste["dead_kv_slices"] += (
                2 * M * k_in * 2 + 2 * M * r * 4        # 2 of 3 shrink slices
                + 2 * M * r * 4 + 2 * M * sum(slices[1:]) * 2
            )
        waste["shrink_reread"] += (ns - 1) * M * k_in * 2

    for layer in range(LAYERS):
        for name, k_in, slices, targeted in groups(layer):
            add(name, k_in, slices, targeted, layer)
    # one per-forward-pass projection producing all 35 layers' per-layer inputs
    add("per_layer_model_projection", HIDDEN, [PER_LAYER_INPUT * LAYERS], False)

    tot_s = sum(d["shrink"] for d in rows.values())
    tot_e = sum(d["expand"] for d in rows.values())
    tot = tot_s + tot_e

    print(f"M={M} rank={r}  {LAYERS} layers\n")
    print(f"{'group':<28}{'n':>4}{'shrink GB':>11}{'expand GB':>11}{'GB':>9}{'% LoRA':>8}")
    for name, d in sorted(rows.items(), key=lambda kv: -(kv[1]["shrink"] + kv[1]["expand"])):
        g = (d["shrink"] + d["expand"]) / 1e9
        print(f"{name:<28}{d['n']:>4}{d['shrink']/1e9:>11.1f}{d['expand']/1e9:>11.1f}"
              f"{g:>9.1f}{100*g*1e9/tot:>7.1f}%")
    print(f"\n{'total':<28}{launches['shrink']+launches['expand']:>4}"
          f"{tot_s/1e9:>11.1f}{tot_e/1e9:>11.1f}{tot/1e9:>9.1f}")
    print(f"launches: {launches['shrink']} shrink + {launches['expand']} expand")
    print(f"shrink {100*tot_s/tot:.0f}% / expand {100*tot_e/tot:.0f}% of traffic")

    print("\nrecoverable traffic:")
    for k, v in waste.items():
        print(f"  {k:<20}{v/1e9:>8.1f} GB  {100*v/tot:>5.1f}% of LoRA traffic")

    if args.ms:
        gbs = (tot / 1e9) / (args.ms / 1e3)
        print(f"\nmeasured {args.ms:.0f} ms/run -> {gbs:.0f} GB/s = "
              f"{100*gbs/args.peak_bw:.1f}% of {args.peak_bw:.0f} GB/s peak")


if __name__ == "__main__":
    main()
