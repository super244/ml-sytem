# Docs, Evaluations, And Datasets Upgrade Prompt

Upgrade documentation, evaluation assets, and dataset inventory until the repository feels curated rather than merely accumulated. Remove unnecessary pieces only when you can prove they are not load-bearing; strengthen the important pieces aggressively.

## Primary Goal

Make the repo easier to understand, easier to trust, and easier to evaluate.

## Read This First

- `prompts/shared-repo-context.md`
- `docs/architecture.md`
- `docs/api-guide.md`
- `docs/runbook.md`
- `data/README.md`
- `evaluation/README.md`
- `data/catalog.py`
- `evaluation/benchmark_registry.py`
- `evaluation/generated_benchmark.py`
- `tests/test_data_processing.py`
- `tests/test_evaluation_metrics.py`

## Scope

- Documentation accuracy and cross-link quality
- Benchmark and dataset registry clarity
- Evaluation configs, generated reports, and failure-analysis tooling
- Pruning stale or duplicated docs/configs/datasets only after reference checks
- Expanding high-value docs that unblock users, contributors, or agents

## Definition Of Done

- Core docs match the current system behavior.
- Evaluation entrypoints and dataset flows are easy to follow.
- Obvious duplication, stale references, or low-value clutter in touched areas is reduced.
- Any removals are justified by reference checks and test impact awareness.
