# Guidelines

## Docs

- Keep every doc file short. Each file should fit on one screen.
- One file per topic. Don't merge unrelated guidelines.
- Use bullet points, not prose paragraphs.
- No filler ("Note that…", "It is important to…"). State facts directly.
- Update the relevant doc when a finding changes the recommended approach.

## Code

- Commit messages explain *why*, not what.
- One logical change per commit.

## Benchmarking with vllm_local.py

```bash
python3 scripts/vllm_local.py \
  --model <hf-model-id-or-local-path> \
  --precision <fp4|fp8|bf16|auto> \
  --input_tokens <N> \
  [--attention_backend FLASHINFER|FLASH_ATTN] \
  [--kv_cache_precision auto|fp8] \
  [--num_runs 5]
```

**Never run two benchmark processes at the same time.** They share the GPU; concurrent runs corrupt results and can OOM. Always confirm `nvidia-smi` shows near-full free memory before starting.

### Defaults

- `--attention_backend FLASHINFER` — fastest on Blackwell (SM100); auto-activates TRTLLM prefill kernels.
- `--kv_cache_precision auto` — keeps KV cache in bf16, required for TRTLLM path.

### Known gotchas

- `FLASH_ATTN` does not support `--kv_cache_precision fp8`; errors at startup.
- `FLASHINFER + fp8 KV cache` bypasses TRTLLM, falling back to native FlashInfer prefill (~27% slower on Blackwell). Prefer `auto` KV cache.
- fp8 KV cache with forced TRTLLM (`use_trtllm_attention=True`) is not yet stable.
