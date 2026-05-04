# Benchmarking with vllm_local.py

## Usage

```bash
python3 scripts/vllm_local.py \
  --model <hf-model-id-or-local-path> \
  --precision <fp4|fp8|bf16|auto> \
  --input_tokens <N> \
  [--attention_backend FLASHINFER|FLASH_ATTN] \
  [--kv_cache_precision auto|fp8] \
  [--num_runs 5]
```

## Rules

- **Never run two benchmark processes at the same time.** They share the GPU; concurrent runs corrupt results and can OOM.
- Always wait for the previous run to fully exit before starting the next. Check with `nvidia-smi` — free memory should be near the full GPU capacity.
- If a process hangs or errors, kill it explicitly before the next run.

## Defaults

- `--attention_backend FLASHINFER` — fastest on Blackwell (SM100); auto-activates TRTLLM prefill kernels.
- `--kv_cache_precision auto` — keeps KV cache in bf16, required for TRTLLM path.

## Known gotchas

- `FLASH_ATTN` does not support `--kv_cache_precision fp8`; it will error at startup.
- `FLASHINFER + fp8 KV cache` bypasses TRTLLM and falls back to native FlashInfer prefill, which is ~27% slower on Blackwell. Prefer `auto` KV cache unless fp8 KV is specifically under test.
- fp8 KV cache with forced TRTLLM (`use_trtllm_attention=True`) is not yet stable.
