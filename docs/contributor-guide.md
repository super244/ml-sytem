# Contributor Guide

## Development Principles

- keep data, training, inference, and evaluation aligned on shared schemas
- prefer config-driven extensions over one-off scripts
- add manifests, cards, and summaries for new workflows
- keep local-first usability intact

## Common Extension Paths

- add a new synthetic family under `data/synthesis/families/`
- add a new public adapter in `data/public/registry.yaml`
- add a new training profile in `training/configs/profiles/`
- add a new prompt preset in `inference/configs/prompt_presets.yaml`
- add a new benchmark in `evaluation/benchmarks/registry.yaml`

## Validation Expectations

Before opening a change, run the highest-value checks available in your environment:

```bash
python3 -m pytest
python3 -m compileall ai_factory data training inference evaluation
cd frontend && npm run typecheck
```
