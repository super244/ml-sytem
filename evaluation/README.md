# Evaluation Layer

The evaluation layer turns the model into a benchmarkable system rather than a purely interactive demo. It supports explicit benchmark manifests, model-vs-model comparison, verifier analysis, slice reporting, and regression-style reporting artifacts.

## Core Modules

- `evaluate.py`: benchmark runner and artifact writer.
- `benchmark_registry.py`: benchmark discovery and registry loading.
- `metrics.py`: answer extraction, parsing, verification, and score computation.
- `reporting.py`: JSON, JSONL, Markdown, and leaderboard reporting.
- `error_taxonomy.py`: reasoning and formatting failure labels.
- `analysis/analyze_failures.py`: failure clustering and drill-down reports.

## Included Benchmark Configs

- `configs/base_smoke.yaml`
- `configs/base_vs_finetuned.yaml`
- `configs/source_ablation_calculus_only.yaml`
- `configs/curriculum_ablation.yaml`
- `configs/verifier_on_off.yaml`
- `configs/run_vs_run_template.yaml`

## Metrics

- final-answer accuracy
- parse rate
- step correctness
- verifier agreement
- formatting failure rate
- arithmetic slip rate
- no-answer rate
- candidate agreement
- latency
- approximate prompt/completion token and cost metadata

## Example Commands

```bash
python3 -m evaluation.evaluate --config evaluation/configs/base_smoke.yaml
python3 -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
python3 -m evaluation.evaluate --config evaluation/configs/verifier_on_off.yaml
python3 evaluation/analysis/analyze_failures.py --input evaluation/results/latest/per_example.jsonl --output evaluation/results/latest/failure_analysis.json
```

`base_smoke.yaml` works before any adapters are trained. `base_vs_finetuned.yaml` requires a packaged adapter at `artifacts/models/atlas-math-failure-aware/latest`.
