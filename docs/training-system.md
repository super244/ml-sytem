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
- scratch causal LM pretraining
- chain-of-thought supervision
- plain-text next-token training over math documents
- curriculum learning
- difficulty-aware weighting
- failure replay augmentation
- verification-oriented weighting
- dry-run validation
- resume from checkpoints
- adapter packaging and merged export

## Default Base Model

The default runnable path uses `Qwen2.5-Math-1.5B-Instruct`. Larger 7B templates are included for scale-up scenarios, but not required for local validation.

For actual from-scratch training, the repo now includes a dedicated 2B-profile path:

- `training/configs/profiles/pretraining.yaml`
- `training/configs/components/models/qwen2_scratch_2b.yaml`
- `training/configs/components/data/pretraining_text_4k.yaml`

The architecture YAML is the place where you specify model scale. The current default is a 24-layer Qwen2-style decoder with `hidden_size=2560`, `intermediate_size=6912`, `num_attention_heads=20`, `num_key_value_heads=10`, and `vocab_size=50257`, which instantiates to about 2.00B parameters.

You can now specify the model size directly through `target_parameters`, for example:

```yaml
target_parameters: 2b
```

Use this helper to preview the resolved architecture:

```bash
python3 training/scripts/plan_model_scale.py --target-parameters 2b
```

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
python3 training/scripts/train_tokenizer.py --config training/configs/profiles/pretraining.yaml --output-dir artifacts/tokenizers/qwen2_math_2b
python3 -m training.train --config training/configs/profiles/pretraining.yaml --dry-run --validate-model-load
python3 -m training.train --config training/configs/profiles/pretraining.yaml
python3 training/scripts/package_adapter.py --run-dir artifacts/runs/<run_id>
python3 training/scripts/export_merged_model.py --run-dir artifacts/runs/<run_id>
python3 training/scripts/compare_runs.py --run-a artifacts/runs/<run_a> --run-b artifacts/runs/<run_b>
```
