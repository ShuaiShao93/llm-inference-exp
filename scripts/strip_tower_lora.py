"""Strip multimodal-tower LoRA weights from a PEFT adapter directory.

Useful when an HF adapter was trained with the full multimodal model loaded
(unsloth often does this with ``unsloth_fixed=true``) so the safetensors
file contains entries for ``vision_tower.*`` / ``audio_tower.*`` /
``multi_modal_projector.*`` modules. vLLM only applies LoRA to the language
tower of multimodal models, so it rejects the adapter with::

    expected target modules in {q_proj, k_proj, ...}
    but received ['vision_tower.encoder.layers.0.self_attn.q_proj.linear', ...]

This script:

  1. Reads ``<src>/adapter_model.safetensors``.
  2. Drops every tensor whose key contains any of the configured prefixes
     (default: vision_tower / audio_tower / multi_modal_projector).
  3. Writes the surviving tensors to ``<dst>/adapter_model.safetensors``.
  4. Copies ``<src>/adapter_config.json`` to ``<dst>`` with ``exclude_modules``
     patched to explicitly list the dropped prefixes (so future PEFT loads
     also skip them).

Usage:

    /usr/bin/python3.12 scripts/strip_tower_lora.py \\
        --src ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev> \\
        --dst ~/model_ckpt/synthetic-loras/<adapter-name>-stripped
"""

import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Source adapter directory (PEFT layout).")
    ap.add_argument("--dst", required=True, help="Destination directory (created if absent).")
    ap.add_argument(
        "--exclude-prefixes",
        nargs="+",
        default=["vision_tower", "audio_tower", "multi_modal_projector"],
        help="Substring prefixes to drop. Default covers the standard "
             "multimodal tower module names.",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    in_st = src / "adapter_model.safetensors"
    out_st = dst / "adapter_model.safetensors"
    kept: dict = {}
    dropped = 0
    with safe_open(in_st, framework="pt", device="cpu") as f:
        for k in f.keys():
            if any(p in k for p in args.exclude_prefixes):
                dropped += 1
                continue
            kept[k] = f.get_tensor(k)
    save_file(kept, out_st)
    print(f"kept {len(kept)} tensors, dropped {dropped}", file=sys.stderr)

    cfg = json.load(open(src / "adapter_config.json"))
    cfg["exclude_modules"] = [f"{p}.*" for p in args.exclude_prefixes]
    json.dump(cfg, open(dst / "adapter_config.json", "w"), indent=2)
    print(f"wrote stripped adapter to {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
