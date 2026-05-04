"""
Quantize a BF16 HuggingFace model and export an HF-format checkpoint that
TRT-LLM's PyTorch backend can load directly (with hf_quant_config.json).

This mirrors the approach from the official NVIDIA Model-Optimizer:
  https://nvidia.github.io/TensorRT-LLM/features/quantization.html
  https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq

Supported formats (--qformat):
  fp8      FP8 weight + activation quantization
  fp4      NVFP4 weight + activation quantization (recommended for Blackwell)

KV cache quantization is NOT baked into the checkpoint; set it at inference
time via --kv_cache_precision on trtllm_local.py (fp8 works on all supported
GPUs; NVFP4 KV requires the weight checkpoint to also be FP8 per the docs).

Usage:
  # Quantize to FP4 (NVFP4) — default model is Llama-3.2-3B-Instruct
  python scripts/quantize_trtllm.py \\
      --qformat fp4 \\
      --output_dir /tmp/llama3b_fp4

  # Quantize to FP8
  python scripts/quantize_trtllm.py \\
      --qformat fp8 \\
      --output_dir /tmp/llama3b_fp8

  # Benchmark the result
  python scripts/trtllm_local.py \\
      --model /tmp/llama3b_fp4 \\
      --precision fp4 \\
      --kv_cache_precision fp8
"""

import argparse
import copy
import os
import sys


# Maps user-facing --qformat names to modelopt config attribute names
_QFORMAT_TO_MTQ_CFG = {
    "fp8": "FP8_DEFAULT_CFG",
    "fp4": "NVFP4_DEFAULT_CFG",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a BF16 HF model to a TRT-LLM-compatible checkpoint")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model ID or local path to a BF16 checkpoint",
    )
    parser.add_argument(
        "--qformat",
        default="fp4",
        choices=list(_QFORMAT_TO_MTQ_CFG),
        help="Quantization format: fp8 or fp4 (default: fp4)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write the quantized HF checkpoint",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=64,
        help="Number of calibration samples (default: 64)",
    )
    parser.add_argument(
        "--calib_max_seq_length",
        type=int,
        default=512,
        help="Max token length per calibration sample (default: 512)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Calibration batch size (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        sys.exit("nvidia-modelopt not installed. Run: pip install 'nvidia-modelopt[torch]~=0.37.0'")

    import torch
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tensorrt_llm.quantization.quantize_by_modelopt import get_calib_dataloader

    cfg_name = _QFORMAT_TO_MTQ_CFG[args.qformat]
    quant_cfg = getattr(mtq, cfg_name)

    print(f"Model:      {args.model}")
    print(f"Format:     {args.qformat} ({cfg_name})")
    print(f"Output dir: {args.output_dir}")
    print(f"Calib size: {args.calib_size} samples × {args.calib_max_seq_length} tokens")
    print()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=False,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading calibration data (cnn_dailymail)...")
    calib_dataloader = get_calib_dataloader(
        dataset_name_or_dir="cnn_dailymail",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        calib_size=args.calib_size,
        block_size=args.calib_max_seq_length,
        device="cuda",
    )

    print(f"Quantizing ({args.qformat})...")
    # Deep copy so the original config dict is not mutated
    cfg = copy.deepcopy(quant_cfg)

    def calibrate():
        for batch in calib_dataloader:
            with torch.no_grad():
                model(**batch)

    mtq.quantize(model, cfg, calibrate)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Exporting to {args.output_dir}...")
    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone. Checkpoint written to: {args.output_dir}")
    print(f"\nTo benchmark with trtllm_local.py:")
    kv = "fp8" if args.qformat == "fp8" else "auto"
    print(f"  python scripts/trtllm_local.py \\")
    print(f"      --model {args.output_dir} \\")
    print(f"      --precision {args.qformat} \\")
    print(f"      --kv_cache_precision {kv}")


if __name__ == "__main__":
    main()
