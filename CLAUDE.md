# Guidelines

## Docs

- This guideline doc should be concise and most of the bullets should be 1 sentence.
- When executing the skills, if anything is found to need updates, ask if we should update it.
- For any complex flow that takes long to figure out what to do, ask if we should add a skill and/or a script for it.
- For skills, avoid too specific steps/details that are spefiic to a particular model/env or that may be changed after a dependency upgrade.

## Code

- Do not commit unless you are asked to.

## Benchmarking

- scripts/ directory has scripts for running benchmark
- trtllm must be run in a miniconda env (`~/miniconda3/envs/trtllm/bin/python`), but vllm can be run in the system env (`/usr/bin/python3.12`)
- Always pick the most optimal default arguments unless you are asked to test a specific config.
- Never run two benchmark processes at the same time. They share the GPU; concurrent runs corrupt results and can OOM.
- If a pre-quantized checkpoint at the desired precision is not available on HuggingFace, use `scripts/quantize_trtllm.py` to create one from a BF16 model before benchmarking.


