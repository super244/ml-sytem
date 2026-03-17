# Data Layer

The data layer turns heterogeneous math corpora into a single research-grade schema and a set of reproducible packs. It is built around canonical `v2` records, lineage tracking, dedupe, contamination checks, quality scoring, and derived benchmark/training slices.

## Main Packages

- `adapters/`: public dataset registry and normalization logic
- `builders/`: processed corpus assembly and derived-pack construction
- `quality/`: difficulty estimation, scoring, contamination, and mining
- `reports/`: dataset cards and size reports
- `synthesis/`: synthetic families and generator registry
- `tools/`: preview, export, validate, audit, and benchmark-pack utilities

## Supported Dataset Families

- `custom_derivative_mastery`
- `custom_integral_arena`
- `custom_limits_series_lab`
- `custom_multivariable_studio`
- `custom_odes_optimization_lab`
- `custom_olympiad_reasoning_studio`

## Core Commands

```bash
python3 data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
python3 data/public/normalize_public_datasets.py --registry data/public/registry.yaml
python3 data/prepare_dataset.py --config data/configs/processing.yaml
python3 data/tools/audit_dataset.py --input data/processed/normalized_all.jsonl
python3 data/tools/export_subset.py --input data/processed/train.jsonl --output /tmp/train_subset.jsonl --limit 128
python3 data/mine_failure_cases.py --input evaluation/results/latest/per_example.jsonl --output data/raw/failure_cases.jsonl
```
