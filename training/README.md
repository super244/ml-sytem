# Training Layer

The training layer is a composed experiment engine for math-specialist adaptation and scratch-model pretraining. It supports local QLoRA-first iteration, curriculum and failure-aware weighting, scratch initialization from an explicit architecture config, tokenizer training, optional full-precision export, and run packaging for downstream inference and evaluation.
It now also performs stricter config validation, emits richer run reports/manifests, can resume from the latest checkpoint on demand, and plugs cleanly into the Linux cloud and macOS local bootstrap scripts.

## Core Modules

- `train.py`: main CLI entry point with ultimate harness integration.
- `src/config.py`: composed experiment configuration and profile loading.
- `src/data.py`: dataset loading, prompt formatting, weighting, and curriculum ordering.
- `src/collators.py`: weighted data collator support.
- `src/modeling.py`: model/tokenizer loading, target-parameter scratch-model construction, LoRA/QLoRA prep, merged export helpers.
- `src/scaling.py`: parameter-budget parsing and qwen2 architecture planning.
- `src/trainer.py`: weighted-loss trainer integration with ultimate harness support.
- `src/validation.py`: dry-run tokenization and data validation.
- `src/analysis.py`: parameter reports, dataset diagnostics, run summaries.
- `src/packaging.py`: manifest writing, publication, and serving-oriented packaging.
- `src/comparison.py`: run-vs-run comparison helpers and report generation.
- `src/environment.py`: reproducibility snapshots for Python, platform, packages, files, and runtime context.
- `src/tracking.py`: optional tracker adapters plus always-on local tracking artifacts.
- `src/optimization.py`: hardware detection and backend-optimized configuration (Metal/CUDA/CPU).
- `src/ultimate_harness.py`: ultimate training harness with automatic hardware-aware optimization.
- `scripts/train_tokenizer.py`: local BPE tokenizer training over the configured corpus.
- `scripts/plan_model_scale.py`: plan a scratch architecture from a target parameter budget.

Bootstrap-first entry points:

- `scripts/start-linux.sh`: Linux cloud GPU startup, dependency install, CUDA checks, tokenizer setup, and training launch.
- `scripts/start-mac.sh`: Apple Silicon local startup, dependency install, Metal-aware checks, tokenizer setup, and training launch.
- `training/configs/profiles/local_metal.yaml`: Apple Silicon-friendly local profile that disables Linux-only quantization defaults.

## Config Layout

- `configs/components/models/`
- `configs/components/adapters/`
- `configs/components/data/`
- `configs/components/runtime/`
- `configs/components/logging/`
- `configs/components/tracking/`
- `configs/components/packaging/`
- `configs/profiles/`

Legacy top-level config names are retained as wrappers for compatibility, but the primary entry points are the profile configs.

The parameter budget for a scratch run now lives in the model component YAML. The canonical ladder is:

- `training/configs/components/models/qwen2_scratch_1b.yaml`
- `training/configs/components/models/qwen2_scratch_2b.yaml`
- `training/configs/components/models/qwen2_scratch_4b.yaml`
- `training/configs/components/models/qwen2_scratch_9b.yaml`
- `training/configs/components/models/qwen2_scratch_12b.yaml`
- `training/configs/components/models/qwen2_scratch_20b.yaml`
- `training/configs/components/models/qwen2_scratch_27b.yaml`
- `training/configs/components/models/qwen2_scratch_30b.yaml`
- `training/configs/components/models/qwen2_scratch_70b.yaml`
- `training/configs/components/models/qwen2_scratch_120b.yaml`

For the default pretraining path, `training/configs/components/models/qwen2_scratch_2b.yaml` uses `target_parameters: 2b` and resolves to the 24-layer / 2560-hidden / ~2.00B-parameter architecture automatically.

## Named Profiles

- `baseline_qlora`
- `math_specialist`
- `failure_aware`
- `verifier_augmented`
- `long_context`
- `fast_dev`
- `full_finetune`
- `continual_learning`
- `multitask_learning`
- `pretraining`
- `m5_max_ultimate` - Apple Silicon M5 Max with Metal Performance Shaders
- `cuda_ultimate_a100` - NVIDIA A100 with Tensor Cores and BF16
- `cuda_ultimate_h100` - NVIDIA H100 with Transformer Engine and FP8

## Ultimate Optimization

The training layer now includes ultimate hardware-aware optimization:

- **Automatic Hardware Detection**: Detects Apple Silicon (M1/M2/M3/M4/M5) and NVIDIA GPUs (A100/H100/RTX)
- **Backend Selection**: Automatically selects Metal for Apple Silicon, CUDA for NVIDIA
- **Mixed Precision**: BF16/FP16 for CUDA, optimized FP32 for Metal
- **Fused Kernels**: RMSNorm+SiLU fusion, FlashAttention, optimized memory layouts
- **Zero-Copy Memory**: Unified memory path on Apple Silicon

### Hardware Detection

```bash
python -m training.src.optimization
ai-factory optimize detect
```

### Benchmarking

```bash
python -c "from training.src.ultimate_harness import quick_benchmark; quick_benchmark()"
ai-factory optimize benchmark
```

### Training with Ultimate Profiles

```bash
# Apple Silicon M5 Max
python -m training.train --config training/configs/profiles/m5_max_ultimate.yaml

# NVIDIA A100
python -m training.train --config training/configs/profiles/cuda_ultimate_a100.yaml

# NVIDIA H100
python -m training.train --config training/configs/profiles/cuda_ultimate_h100.yaml
```

## Example Commands

```bash
python3 -m ai_factory.cli train-preflight --config training/configs/profiles/failure_aware.yaml
python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --resume-from-latest-checkpoint
python3 -m training.train --config training/configs/profiles/math_specialist.yaml
python3 training/scripts/plan_model_scale.py --target-parameters 2b
python3 training/scripts/train_tokenizer.py --config training/configs/profiles/pretraining.yaml --output-dir artifacts/tokenizers/qwen2_math_2b
python3 -m training.train --config training/configs/profiles/pretraining.yaml --dry-run --validate-model-load
python3 -m training.train --config training/configs/profiles/pretraining.yaml
python3 training/scripts/export_merged_model.py --run-dir artifacts/runs/<run_id>
python3 training/scripts/compare_runs.py --left artifacts/runs/<run_a> --right artifacts/runs/<run_b> --markdown-output artifacts/runs/comparison.md
```

For the scratch-pretraining path:

- `training/configs/profiles/pretraining.yaml` is now a real scratch run, not a mislabeled fine-tune.
- `training/configs/components/data/pretraining_text_4k.yaml` switches the data objective to causal next-token modeling over plain math documents instead of masked instruction tuning.
- `training/configs/components/models/qwen2_scratch_2b.yaml` is the default scratch template; the matching sibling files cover the rest of the ladder.
- `training/scripts/train_tokenizer.py` can build a local tokenizer that matches the configured vocab budget before the first full run.
- `training/scripts/plan_model_scale.py` lets you preview the resolved qwen2 architecture before you commit to a run.

Every run now writes:

- `manifests/config_snapshot.json`
- `manifests/config_report.json`
- `manifests/validation_report.json`
- `manifests/environment_snapshot.json`
- `manifests/tracking_context.json`
- `logs/tracking_events.jsonl`
- `metrics/tracking_summary.json`

The new `ai-factory train-preflight --config <profile>` command validates artifacts, disk headroom, tokenizer readiness, model source resolution, CUDA visibility, and the active attention backend before a real launch. For scratch pretraining, a real run now requires the local tokenizer artifact referenced by `model.tokenizer_path`; dry-runs still allow a fallback tokenizer so you can validate the rest of the path first.

On CPU-only machines, `python -m training.train --config <profile> --dry-run` now builds `transformers.TrainingArguments` in CPU mode automatically so config validation and artifact checks succeed without a CUDA device. Mixed-precision flags from the profile are still honored automatically when the same config is launched on a CUDA host.

The corpus preparation path is also tuned for less waiting on larger datasets, so tokenization and split generation should feel faster than the previous single-pass flow.
