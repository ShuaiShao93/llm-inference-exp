"""Build a synthetic LoRA adapter with random weights at a fixed (r, alpha)
shape that targets the standard 7 projection modules.

Used by the ``vllm-backend-matrix`` skill when no existing HF adapter matches
the desired (r=16, alpha=16) shape for a given base model. The weights are
random so don't expect coherent outputs, but the *shape* is correct so vLLM
exercises its real LoRA dispatch path with the right rank.

Saves a PEFT-compatible adapter directory (adapter_config.json +
adapter_model.safetensors) that vLLM can load via LoRARequest(lora_path=...).

Usage:
    /usr/bin/python3.12 scripts/build_synthetic_lora.py \\
        --base RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic \\
        --out ~/model_ckpt/synthetic-loras/llama-3.2-3b-r16

For the Gemma 4 multimodal family, the script automatically excludes
vision_tower / audio_tower / multi_modal_projector modules so vLLM (which
loads only the language tower for LoRA) accepts the adapter.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Base model HF id or local path.")
    ap.add_argument("--out", required=True, help="Output directory for the adapter.")
    ap.add_argument("--r", type=int, default=16, help="LoRA rank (default 16).")
    ap.add_argument("--alpha", type=int, default=16, help="LoRA alpha (default 16).")
    ap.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
        help="Target module suffixes. Default covers attention QKV+O and MLP up/gate/down.",
    )
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # CPU-only load is enough — we don't run inference, just attach LoRA and
    # save. Skip the slow GPU load and use bfloat16 to keep memory modest.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"Loading base model: {args.base}", file=sys.stderr)
    cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=False)
    # Multimodal exclusions: Gemma 4 family ships a Gemma4ForConditionalGeneration
    # with vision_tower + audio_tower. vLLM doesn't apply LoRA there, so the
    # adapter must explicitly skip them.
    arch = (cfg.architectures or [""])[0].lower()
    exclude_modules: list[str] = []
    if "conditional" in arch or "gemma4" in arch or "gemma3" in arch:
        exclude_modules = ["vision_tower.*", "audio_tower.*", "multi_modal_projector.*"]
        print(f"Multimodal architecture detected ({arch}); excluding {exclude_modules}",
              file=sys.stderr)

    # ``device_map=None`` keeps everything on CPU; we only need shapes.
    # AutoModelForCausalLM doesn't know how to load multimodal ForConditionalGeneration
    # heads (Mistral3 / Gemma4 vision-language models); fall back to AutoModel which
    # accepts any architecture. The language LoRA still attaches to the language
    # tower because get_peft_model searches by module-name suffix.
    # Multimodal CausalLM models register under AutoModelForImageTextToText
    # rather than AutoModelForCausalLM. Try the causal-LM factory first; on
    # ``Unrecognized configuration class`` fall back to the image-text-to-text
    # factory so we still get prepare_inputs_for_generation (which PEFT's
    # CAUSAL_LM task type needs).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None,
        )
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText
        print("AutoModelForCausalLM rejected this arch; falling back to AutoModelForImageTextToText",
              file=sys.stderr)
        model = AutoModelForImageTextToText.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None,
        )
    lora_kwargs = dict(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=args.target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if exclude_modules:
        lora_kwargs["exclude_modules"] = exclude_modules

    lora_cfg = LoraConfig(**lora_kwargs)
    peft_model = get_peft_model(model, lora_cfg)
    print("Adapter modules instantiated:", file=sys.stderr)
    peft_model.print_trainable_parameters()

    peft_model.save_pretrained(out)
    print(f"\nSaved synthetic LoRA to: {out}", file=sys.stderr)
    print(f"  r={args.r}  alpha={args.alpha}", file=sys.stderr)
    print(f"  target_modules={args.target_modules}", file=sys.stderr)
    if exclude_modules:
        print(f"  exclude_modules={exclude_modules}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
