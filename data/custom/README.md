# Custom Dataset Packs

This directory is populated by `data/generator/generate_calculus_datasets.py`. The generator materializes six custom corpora, each targeting roughly 3 MB, to make the repository feel like a real specialist ML workspace instead of a thin scaffold.

Generated packs:

- `custom_derivative_mastery.jsonl`
- `custom_integral_arena.jsonl`
- `custom_limits_series_lab.jsonl`
- `custom_multivariable_studio.jsonl`
- `custom_odes_optimization_lab.jsonl`
- `custom_olympiad_reasoning_studio.jsonl`

Each record follows the canonical schema and includes:

- `schema_version`
- `id`
- `question`
- `solution`
- `final_answer`
- `difficulty`
- `topic`
- `reasoning_style`
- `source`
- `step_checks`
- `lineage`
- `generator`
- `metadata`
