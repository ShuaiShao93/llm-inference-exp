# Guidelines

## Docs

- This guideline doc should be concise and most of the bullets should be 1 sentence.

## Code

- Do not commit unless you are asked to.

## Benchmarking with vllm_local.py

- Always pick the most optimal default arguments unless you are asked to test a specific config.
- Never run two benchmark processes at the same time. They share the GPU; concurrent runs corrupt results and can OOM.

