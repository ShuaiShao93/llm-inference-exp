"""Launch punica shrink/expand once per real E2B layer-group shape, for ncu.

The earlier ncu pass only captured qkv-shaped launches, but down_proj is 43% of
shrink traffic and gate_up expand is 56% of all LoRA traffic — so the achieved
bandwidth of the shapes that actually dominate was never measured. This driver
issues exactly one shrink and one expand per named group so ncu can target them
with a small --launch-count.

Slice widths are per-slice, not uniform: a merged QKV under GQA is
(2048, 256, 256), not three equal projections. Modelling it as uniform
overstates expand's output width by ~1.6x.

Usage:
    python scripts/ncu_punica_shapes.py --group qkv_kv --tokens 100000 --tile best
"""

import argparse

import torch

from vllm.lora.ops.triton_ops import lora_expand_op, lora_shrink_op

HIDDEN = 1536

# name -> (shrink input width, [expand output width per slice])
GROUPS = {
    "qkv_kv": (HIDDEN, [2048, 256, 256]),   # layers with their own k/v
    "qkv_q": (HIDDEN, [2048]),              # layers that share k/v upstream
    "gate_up": (HIDDEN, [6144, 6144]),
    "o_proj": (2048, [HIDDEN]),
    "down_proj": (6144, [HIDDEN]),
}

# Winners from the L40S tile sweep. split_k=1 removes the atomic accumulation
# pass and the output memset it forces.
TILES = {
    "best": (
        dict(block_m=128, block_n=16, block_k=128, split_k=1),
        dict(block_m=64, block_n=128, block_k=32),
    ),
    "default": (
        dict(block_m=32, block_n=16, block_k=32, split_k=8),
        dict(block_m=64, block_n=64, block_k=32),
    ),
}
BASE = dict(num_warps=4, num_ctas=1, group_size_m=8, num_stages=2, max_nreg=None)


def metadata(m, device):
    return {
        "token_lora_mapping": torch.zeros(m, dtype=torch.int32, device=device),
        "token_indices_sorted_by_lora_ids": torch.arange(m, dtype=torch.int32, device=device),
        "num_tokens_per_lora": torch.tensor([m, 0], dtype=torch.int32, device=device),
        "lora_token_start_loc": torch.tensor([0, m, m], dtype=torch.int32, device=device),
        "lora_ids": torch.tensor([0, -1], dtype=torch.int32, device=device),
        "no_lora_flag_cpu": torch.tensor([False], dtype=torch.bool, device="cpu"),
        "num_active_loras": torch.tensor([1], dtype=torch.int32, device="cpu"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=sorted(GROUPS))
    p.add_argument("--tokens", type=int, default=100000)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--tile", default="best", choices=sorted(TILES))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=3)
    args = p.parse_args()

    dev, dt = torch.device("cuda"), torch.bfloat16
    kin, outs = GROUPS[args.group]
    ns, m, r = len(outs), args.tokens, args.rank
    s_cfg, e_cfg = TILES[args.tile]
    s_cfg, e_cfg = {**s_cfg, **BASE}, {**e_cfg, **BASE}
    lora_shrink_op.get_lora_op_configs = lambda *a, **k: s_cfg
    lora_expand_op.get_lora_op_configs = lambda *a, **k: e_cfg

    meta = metadata(m, dev)
    act = torch.randn(m, kin, dtype=dt, device=dev)
    a_w = [torch.randn(1, r, kin, dtype=dt, device=dev) for _ in range(ns)]
    inter = torch.zeros(ns, m, r, dtype=torch.float32, device=dev)
    b_w = [torch.randn(1, o, r, dtype=dt, device=dev) for o in outs]
    out = torch.zeros(m, sum(outs), dtype=dt, device=dev)

    def shrink():
        lora_shrink_op._lora_shrink(act, a_w, inter, scaling=1.0, **meta)

    def expand():
        lora_expand_op._lora_expand(inter, b_w, out, offset_start=0, add_inputs=True, **meta)

    for _ in range(args.warmup):
        shrink()
        expand()
    torch.cuda.synchronize()

    for _ in range(args.iters):
        shrink()
        expand()
    torch.cuda.synchronize()

    # Bytes a perfect implementation would move: read the activation once, write
    # the rank-16 intermediate; then read it back and read-modify-write the
    # output because add_inputs=True accumulates onto the base projection.
    s_floor = m * kin * 2 + ns * m * r * 4
    e_floor = ns * m * r * 4 + 2 * m * sum(outs) * 2
    print(f"OK group={args.group} tile={args.tile} M={m} K={kin} slices={outs} "
          f"shrink_floor={s_floor/1e6:.1f}MB expand_floor={e_floor/1e6:.1f}MB")


if __name__ == "__main__":
    main()
