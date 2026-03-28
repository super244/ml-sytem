# Training Layer

The training layer is a composed experiment engine for math-specialist adaptation. It supports local QLoRA-first iteration, curriculum and failure-aware weighting, optional full-precision export, and run packaging for downstream inference and evaluation.

## Core Modules

- `train.py`: main CLI entry point.
- `src/config.py`: composed experiment configuration and profile loading.
- `src/data.py`: dataset loading, prompt formatting, weighting, and curriculum ordering.
- `src/collators.py`: weighted data collator support.
- `src/modeling.py`: model/tokenizer loading, LoRA/QLoRA prep, merged export helpers.
- `src/trainer.py`: weighted-loss trainer integration.
- `src/validation.py`: dry-run tokenization and data validation.
- `src/analysis.py`: parameter reports, dataset diagnostics, run summaries.
- `src/packaging.py`: manifest writing, publication, and serving-oriented packaging.
- `src/comparison.py`: run-vs-run comparison helpers.
- `src/environment.py`: reproducibility snapshots for Python, platform, packages, files, and runtime context.
- `src/tracking.py`: optional tracker adapters plus always-on local tracking artifacts.

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

## Named Profiles

- `baseline_qlora`
- `calculus_specialist`
- `curriculum_specialist`
- `failure_aware`
- `verifier_augmented`
- `long_context`
- `fast_iteration_small_model`

## Example Commands

```bash
python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python3 -m training.train --config training/configs/profiles/calculus_specialist.yaml
python3 training/scripts/export_merged_model.py --run-dir artifacts/runs/<run_id>
python3 training/scripts/compare_runs.py --run-a artifacts/runs/<run_a> --run-b artifacts/runs/<run_b>
```

Every run now writes:

- `manifests/config_snapshot.json`
- `manifests/environment_snapshot.json`
- `manifests/tracking_context.json`
- `logs/tracking_events.jsonl`
- `metrics/tracking_summary.json`
