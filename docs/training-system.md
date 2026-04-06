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

## Operator Entry Points

The recommended first step for a new run is to use the platform-specific bootstrap script:

- Linux cloud GPU hosts: `bash scripts/start-linux.sh`
- Apple Silicon local hosts: `bash scripts/start-mac.sh`

Those scripts are designed to install dependencies, prepare tokenizer assets, and hand off to the selected training profile with minimal manual setup.

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

## Model Scale Ladder

The default runnable path uses the registry-backed baseline entry, while scratch work is organized around a canonical scale ladder instead of a single hard-coded size.

The repository now includes dedicated scratch templates for:

- `1b`
- `2b`
- `4b`
- `9b`
- `12b`
- `20b`
- `27b`
- `30b`
- `70b`
- `120b`

For actual from-scratch training, the repo includes a dedicated 2B-profile path:

- `training/configs/profiles/pretraining.yaml`
- `training/configs/components/models/qwen2_scratch_2b.yaml`
- `training/configs/components/data/pretraining_text_4k.yaml`

The architecture YAML is the place where you specify model scale. The current default is a 24-layer Qwen2-style decoder with `hidden_size=2560`, `intermediate_size=6912`, `num_attention_heads=20`, `num_key_value_heads=10`, and `vocab_size=50257`, which instantiates to about 2.00B parameters.

You can specify the model size directly through `target_parameters`, for example `2b`, or switch to the matching template in `training/configs/components/models/` for the size you want.

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
- Titan runtime hardware reporting for CUDA, Metal, and CPU fallback

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

The data preparation path is optimized to reduce repeated work during tokenization and corpus assembly, so the preprocessing steps behind these commands should complete faster on large datasets.
