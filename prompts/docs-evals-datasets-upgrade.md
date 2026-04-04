# Docs, Evaluations, And Datasets Upgrade Prompt

Upgrade the repository's documentation, evaluation frameworks, and dataset catalogs to rigorous production standards. Your goal is to curate these assets so the repository feels trustworthy, comprehensible, and scientifically reproducible.

## Primary Goal

Elevate documentation to act as the ultimate source of truth, make evaluation pipelines bulletproof and deterministic, and establish a dataset inventory that enforces strict lineage tracking. Eliminate ambiguity. 

## Read This First (Mandatory Ingestion)

- `prompts/shared-repo-context.md`
- `docs/architecture.md` (System blueprint)
- `docs/api-guide.md` (OpenAPI contract constraints)
- `docs/runbook.md` (Operational procedures)
- `data/README.md` (Dataset pipeline architecture)
- `evaluation/README.md` (Evaluation harness architecture)
- `data/catalog.py` (Dataset registry and schemas)
- `evaluation/benchmark_registry.py` (Evaluation scenarios)
- `evaluation/generated_benchmark.py` (Dynamic evaluation engine)
- `tests/test_data_processing.py`
- `tests/test_evaluation_metrics.py`

## Scope & Execution Directives

1. **Documentation Hygiene**:
   - Ensure `architecture.md` and `api-guide.md` accurately reflect the *current* codebase. Cross-link heavily.
   - Use Markdown linters to enforce uniform formatting.
   - Expand high-value docs that unblock users, agents, and contributors (e.g., precise setup instructions, troubleshooting matrices).
2. **Dataset Registry Quality (The `data/` module)**:
   - Enforce explicit Pydantic validation on all dataset catalog entries.
   - Datasets must carry cryptographic hashes (SHA-256) and explicit lineage data.
   - Clean up stale, unreferenced JSON files or dead loading scripts only after exhaustive `grep_search`.
3. **Evaluation Rigor (The `evaluation/` module)**:
   - Ensure the evaluation harness (MMLU, HumanEval, GSM8K) handles failures gracefully. An evaluator should not crash if an LLM outputs malformed JSON.
   - Introduce strict metrics typing (e.g., separating continuous scores from discrete pass/fail boolean states).
   - Generate detailed failure analysis artifacts (JSONL) on every evaluation run to unblock error-driven retraining.

## Definition Of Done

- The core documentation seamlessly matches system behavior, verified by testing the documented CLI/API commands.
- The evaluation entry points and dataset ingestion flows are completely deterministic, strictly typed, and cleanly structured.
- Obvious duplication, stale references, and confusingly named benchmarks are resolved.
- Deletions of legacy data are fully justified by reference checks.