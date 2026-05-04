# Guidelines

## Docs

- This guideline doc should be concise and most of the bullets should be 1 sentence.
- When executing the skills, if anything is found to need updates, suggest updating it.
- For any complex flow that takes long to figure out what to do, suggest adding a skill for it.

## Code

- Do not commit unless you are asked to.

## Benchmarking

- scripts/ directory has scripts for running benchmark
- Always pick the most optimal default arguments unless you are asked to test a specific config.
- Never run two benchmark processes at the same time. They share the GPU; concurrent runs corrupt results and can OOM.


