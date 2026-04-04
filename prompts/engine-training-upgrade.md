# Engine And Training Upgrade Prompt

Upgrade the AI-Factory engine-adjacent training stack so local workflows, fine-tuning scripts, packaging, and performance-sensitive paths are as strong as they can reasonably be in this repository.

## Primary Goal

Improve training quality, reproducibility, and performance without breaking the surrounding platform contracts.

## Read This First

- `prompts/shared-repo-context.md`
- `training/README.md`
- `training/src/config.py`
- `training/src/trainer.py`
- `training/src/workflows.py`
- `training/src/modeling.py`
- `training/src/packaging.py`
- `training/scripts/`
- `ai_factory_titan/README.md`
- `ai_factory_titan/src/`

## Scope

- Training workflows, checkpointing, packaging, comparison, and tracking
- Engine-adjacent optimizations that improve runtime realism
- Better defaults for QLoRA/fine-tuning/export paths
- Places where the Rust Titan engine can support training or inference-adjacent workloads

## Definition Of Done

- Training scripts remain coherent and config-driven.
- Performance-sensitive code paths have a documented reason for their structure.
- Packaging and export outputs stay compatible with the wider AI-Factory system.
- Tests or smoke checks cover the touched workflow.
