# Training System

The training system is built around composed profiles rather than one-off monolithic YAML files.

## Config Components

- `components/models/`
- `components/adapters/`
- `components/data/`
- `components/runtime/`
- `components/logging/`
- `components/packaging/`
- `profiles/`

Profiles combine these pieces to define the full experiment.

## Supported Modes

- instruction tuning
- chain-of-thought supervision
- curriculum learning
- difficulty-aware weighting
- failure replay augmentation
- verification-oriented weighting
- dry-run validation
- resume from checkpoints
- adapter packaging and merged export

## Default Base Model

The default runnable path uses `Qwen2.5-Math-1.5B-Instruct`. Larger 7B templates are included for scale-up scenarios, but not required for local validation.

## Runtime Readiness

The repo includes:

- local hybrid runtime settings
- scale-up runtime settings
- Accelerate-ready config
- DeepSpeed Zero-2 example config

## Main Commands

```bash
python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python3 -m training.train --config training/configs/profiles/verifier_augmented.yaml
python3 training/scripts/package_adapter.py --run-dir artifacts/runs/<run_id>
python3 training/scripts/export_merged_model.py --run-dir artifacts/runs/<run_id>
python3 training/scripts/compare_runs.py --run-a artifacts/runs/<run_a> --run-b artifacts/runs/<run_b>
```
