# Training Layer

The training layer is a composed experiment engine for math-specialist adaptation and scratch-model pretraining. It supports local QLoRA-first iteration, curriculum and failure-aware weighting, scratch initialization from an explicit architecture config, tokenizer training, optional full-precision export, and run packaging for downstream inference and evaluation.
It now also performs stricter config validation, emits richer run reports/manifests, and can resume from the latest checkpoint on demand.

## Core Modules

- `train.py`: main CLI entry point.
- `src/config.py`: composed experiment configuration and profile loading.
- `src/data.py`: dataset loading, prompt formatting, weighting, and curriculum ordering.
- `src/collators.py`: weighted data collator support.
- `src/modeling.py`: model/tokenizer loading, target-parameter scratch-model construction, LoRA/QLoRA prep, merged export helpers.
- `src/scaling.py`: parameter-budget parsing and qwen2 architecture planning.
- `src/trainer.py`: weighted-loss trainer integration.
- `src/validation.py`: dry-run tokenization and data validation.
- `src/analysis.py`: parameter reports, dataset diagnostics, run summaries.
- `src/packaging.py`: manifest writing, publication, and serving-oriented packaging.
- `src/comparison.py`: run-vs-run comparison helpers and report generation.
- `src/environment.py`: reproducibility snapshots for Python, platform, packages, files, and runtime context.
- `src/tracking.py`: optional tracker adapters plus always-on local tracking artifacts.
- `scripts/train_tokenizer.py`: local BPE tokenizer training over the configured corpus.
- `scripts/plan_model_scale.py`: plan a scratch architecture from a target parameter budget.

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

## Example Commands

```bash
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
