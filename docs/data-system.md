# Data System

AI-Factory uses a canonical `v2` record schema so synthetic, public, replay, and benchmark data can move through the same tooling.

## Schema Highlights

- stable `id`
- `question`, `solution`, `final_answer`
- `difficulty`, `topic`, `subtopic`
- `reasoning_style`
- typed `step_checks`
- `quality_score`
- `failure_case`
- contamination metadata
- source lineage
- generator metadata

## Dataset Families

- `custom_derivative_mastery`
- `custom_integral_arena`
- `custom_limits_series_lab`
- `custom_multivariable_studio`
- `custom_odes_optimization_lab`
- `custom_olympiad_reasoning_studio`

## Public Adapters

The public registry currently tracks five calculus-oriented dataset families through a shared adapter interface. Each registry entry can declare:

- loader path and split
- question/solution/topic field mapping
- filtering rules
- reasoning style
- usage intent
- default weighting
- benchmark tags

## Processed Corpus Inputs

`data/prepare_dataset.py` accepts composable source specs in `data/configs/processing.yaml`. In addition to legacy local file paths and globs, sources can be declared as `local`, `huggingface`, `s3`, or `web`, and nested under `composite` groups with per-source `sample_ratio` and `version` metadata.

## Cache Integration

The `ai_factory/core/cache/` layer provides:
- LRU eviction for bounded memory
- Disk-backed persistence for large datasets
- Automatic invalidation on source changes
- Cross-process cache sharing via Redis

## Security Considerations

Sensitive data in datasets is handled by `ai_factory/core/security/`:
- Credential redaction before training
- Encrypted storage for PII-containing datasets
- Access control via `ai_factory/core/security/config.py`

## Derived Packs

The corpus builder emits these fixed pack ids:

- `core_train_mix`
- `calculus_hard_pack`
- `olympiad_reasoning_pack`
- `failure_replay_pack`
- `verification_pack`
- `benchmark_holdout_pack`

## Build Outputs

`data/prepare_dataset.py` writes:

- processed train/eval/test splits
- `manifest.json`
- `card.md`
- `size_report.md`
- `pack_summary.json`
- per-pack manifests and cards under `data/processed/packs/`
- source summaries, version metadata, and optional source warnings in the manifest metadata

## Key Commands

```bash
python3 data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
python3 data/public/normalize_public_datasets.py --registry data/public/registry.yaml
python3 data/prepare_dataset.py --config data/configs/processing.yaml
python3 data/tools/validate_dataset.py --input data/processed/train.jsonl
python3 data/tools/build_benchmark_pack.py --input data/processed/test.jsonl --output /tmp/benchmark_pack
```
