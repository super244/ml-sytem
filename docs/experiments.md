# Experiment Design

AI-Factory is set up to support fast local ablations and more serious specialist runs without changing the operating model. The intended experiment ladder is:

1. Establish a prompt-engineered base-model baseline.
2. Validate the data build and tokenizer path with `--dry-run`.
3. Train a strong local QLoRA baseline.
4. Specialize into curriculum, failure-aware, or verifier-augmented variants.
5. Compare the resulting runs against the same benchmark registry.

## Recommended Profiles

- `baseline_qlora`: default local baseline.
- `calculus_specialist`: emphasizes hard calculus sources and specialist prompts.
- `curriculum_specialist`: orders examples by difficulty progression.
- `failure_aware`: amplifies mined failure cases and replay packs.
- `verifier_augmented`: focuses on examples with `step_checks` and verification anchors.
- `long_context`: increases maximum context and adapter settings for longer derivations.
- `fast_iteration_small_model`: uses a smaller base model for cheap iteration.

## Suggested Ablation Ladder

1. Prompt-only base inference.
2. `baseline_qlora`.
3. `calculus_specialist`.
4. `curriculum_specialist`.
5. `failure_aware`.
6. `verifier_augmented`.
7. `long_context` if latency and memory budgets permit.

## What To Measure

- final-answer accuracy
- parse rate and no-answer rate
- step correctness against typed `step_checks`
- verifier agreement and arithmetic slip rate
- solve rate on hard calculus and olympiad slices
- latency and candidate agreement for self-consistency runs
- qualitative win cases and stable failure clusters

## Failure-Driven Loop

1. Run evaluation on `benchmark_holdout` and `verification_suite`.
2. Inspect `summary.md`, `leaderboard.json`, and `per_example.jsonl`.
3. Mine failures with `data/mine_failure_cases.py`.
4. Rebuild the processed corpus with updated failure logs.
5. Retrain with the `failure_aware` or `verifier_augmented` profile.
6. Compare the new run to the prior run with `training/scripts/compare_runs.py`.

## Artifact Expectations

Every serious experiment should leave behind:

- a frozen config snapshot
- a run manifest
- JSONL metrics
- dataset and parameter reports
- a Markdown run summary
- either a packaged adapter or a merged export for serving

The experiment playbook in `docs/experiment-playbook.md` expands this into concrete recipes.
