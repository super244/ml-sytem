# Evaluation System

Evaluation is benchmark-first. The system treats held-out packs, verification slices, and run comparisons as explicit artifacts rather than ad hoc notebook outputs.

## Benchmark Registry

`evaluation/benchmarks/registry.yaml` declares benchmark ids, titles, paths, descriptions, and tags. The default registry includes:

- `benchmark_holdout`
- `core_eval`
- `calculus_hard`
- `verification_suite`

## Reported Metrics

- final-answer accuracy
- parse rate
- no-answer rate
- step correctness
- verifier agreement
- formatting failures
- arithmetic slips
- latency
- candidate agreement
- approximate prompt/completion token and cost metrics

## Output Artifacts

- `summary.json`
- `summary.md`
- `leaderboard.json`
- `per_example.jsonl`

## Analysis Views

- by topic
- by difficulty
- by source
- by pack
- by generator family
- failure taxonomy
- win-case extraction
- pairwise run comparison
