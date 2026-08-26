"""Compare vLLM's punica LoRA kernels against a plain cuBLAS GEMM of the same shape.

Punica exists to serve many adapters at once: it gathers each token's row via
``token_indices_sorted_by_lora_ids`` so rows belonging to different adapters can
be batched into one launch. With a single always-on adapter that gather is pure
overhead, but the kernel still pays for it. This isolates how much.

For each op we time the punica kernel (shipped default config and the best tile
config found by the sweep) against torch.matmul on identical shapes, and report
achieved DRAM bandwidth so the numbers can be read against the card's peak.
"""

import argparse
import statistics

import torch

from vllm.lora.ops.triton_ops import lora_expand_op, lora_shrink_op


def meta(num_tokens, device):
    return {
        "token_lora_mapping": torch.zeros(num_tokens, dtype=torch.int32, device=device),
        "token_indices_sorted_by_lora_ids": torch.arange(
            num_tokens, dtype=torch.int32, device=device
        ),
        "num_tokens_per_lora": torch.tensor([num_tokens, 0], dtype=torch.int32, device=device),
        "lora_token_start_loc": torch.tensor(
            [0, num_tokens, num_tokens], dtype=torch.int32, device=device
        ),
        "lora_ids": torch.tensor([0, -1], dtype=torch.int32, device=device),
        "no_lora_flag_cpu": torch.tensor([False], dtype=torch.bool, device="cpu"),
        "num_active_loras": torch.tensor([1], dtype=torch.int32, device="cpu"),
    }


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    return statistics.median(s)


def force(module, cfg):
    orig = module.get_lora_op_configs
    module.get_lora_op_configs = lambda *a, **kw: cfg
    return orig


SHRINK_DEFAULT = dict(block_m=32, block_n=16, block_k=32, split_k=8,
                      num_warps=4, num_ctas=1, group_size_m=8, num_stages=2, max_nreg=None)
SHRINK_BEST = dict(block_m=128, block_n=16, block_k=128, split_k=1,
                   num_warps=4, num_ctas=1, group_size_m=8, num_stages=2, max_nreg=None)
EXPAND_DEFAULT = dict(block_m=64, block_n=64, block_k=32,
                      num_warps=4, num_ctas=1, num_stages=2, max_nreg=None)
EXPAND_BEST = dict(block_m=64, block_n=128, block_k=32,
                   num_warps=4, num_ctas=1, num_stages=2, max_nreg=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=100000)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--peak-gbps", type=float, default=864.0)
    # (label, in_hidden, out_hidden, num_slices) mirroring E2B's real layers
    p.add_argument("--iters", type=int, default=20)
    a = p.parse_args()

    dev = torch.device("cuda")
    dt = torch.bfloat16
    M, r = a.tokens, a.rank
    print(f"GPU: {torch.cuda.get_device_name()}  peak {a.peak_gbps:.0f} GB/s")
    print(f"tokens={M} rank={r}\n")

    LAYERS = [
        ("qkv",  1536, 2048, 3),
        ("gate_up", 1536, 6144, 2),
        ("o_proj", 2048, 1536, 1),
        ("down_proj", 6144, 1536, 1),
    ]

    def bw(bytes_moved, ms):
        return bytes_moved / (ms * 1e-3) / 1e9

    hdr = f"{'layer':<11}{'op':<8}{'variant':<10}{'ms':>9}{'GB/s':>9}{'%peak':>7}{'vs cuBLAS':>11}"
    print(hdr); print("-" * len(hdr))

    for name, kin, kout, ns in LAYERS:
        md = meta(M, dev)

        # ---- shrink: [M, kin] @ [kin, r] per slice -> [ns, M, r] fp32
        x = torch.randn(M, kin, dtype=dt, device=dev)
        ws = [torch.randn(1, r, kin, dtype=dt, device=dev) for _ in range(ns)]
        so = torch.zeros(ns, M, r, dtype=torch.float32, device=dev)
        wc = torch.randn(kin, r * ns, dtype=dt, device=dev)

        s_bytes = M * kin * 2 + ns * M * r * 4
        cub = timed(lambda: torch.matmul(x, wc), iters=a.iters)
        print(f"{name:<11}{'shrink':<8}{'cuBLAS':<10}{cub:9.3f}{bw(s_bytes, cub):9.1f}"
              f"{100*bw(s_bytes,cub)/a.peak_gbps:7.1f}{'1.00x':>11}")
        for lbl, cfg in (("default", SHRINK_DEFAULT), ("best-tile", SHRINK_BEST)):
            orig = force(lora_shrink_op, cfg)
            try:
                t = timed(lambda: lora_shrink_op._lora_shrink(x, ws, so, scaling=1.0, **md),
                          iters=a.iters)
                print(f"{'':<11}{'':<8}{lbl:<10}{t:9.3f}{bw(s_bytes,t):9.1f}"
                      f"{100*bw(s_bytes,t)/a.peak_gbps:7.1f}{t/cub:10.2f}x")
            except Exception as e:
                print(f"{'':<11}{'':<8}{lbl:<10}  FAIL {type(e).__name__}")
            finally:
                lora_shrink_op.get_lora_op_configs = orig
        del x, ws, so, wc
        torch.cuda.empty_cache()

        # ---- expand: [ns, M, r] @ [r, kout] per slice -> [M, kout*ns], accumulating
        d = torch.randn(ns, M, r, dtype=torch.float32, device=dev)
        we = [torch.randn(1, kout, r, dtype=dt, device=dev) for _ in range(ns)]
        eo = torch.zeros(M, kout * ns, dtype=dt, device=dev)
        wl = [torch.randn(r, kout, dtype=dt, device=dev) for _ in range(ns)]
        dl = [d[i].to(dt) for i in range(ns)]

        # add_inputs=True reads the output back before writing, so it counts twice
        e_bytes = ns * M * r * 4 + 2 * M * kout * ns * 2

        def cublas_expand():
            for i in range(ns):
                eo[:, i * kout:(i + 1) * kout] += torch.matmul(dl[i], wl[i])

        cub = timed(cublas_expand, iters=a.iters)
        print(f"{name:<11}{'expand':<8}{'cuBLAS':<10}{cub:9.3f}{bw(e_bytes,cub):9.1f}"
              f"{100*bw(e_bytes,cub)/a.peak_gbps:7.1f}{'1.00x':>11}")
        for lbl, cfg in (("default", EXPAND_DEFAULT), ("best-tile", EXPAND_BEST)):
            orig = force(lora_expand_op, cfg)
            try:
                t = timed(lambda: lora_expand_op._lora_expand(
                    d, we, eo, offset_start=0, add_inputs=True, **md), iters=a.iters)
                print(f"{'':<11}{'':<8}{lbl:<10}{t:9.3f}{bw(e_bytes,t):9.1f}"
                      f"{100*bw(e_bytes,t)/a.peak_gbps:7.1f}{t/cub:10.2f}x")
            except Exception as e:
                print(f"{'':<11}{'':<8}{lbl:<10}  FAIL {type(e).__name__}")
            finally:
                lora_expand_op.get_lora_op_configs = orig
        del d, we, eo, wl, dl
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
