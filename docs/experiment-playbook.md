# Experiment Playbook

## Recommended Local Ladder

1. Generate and normalize data.
2. Build the processed corpus and verify manifests.
3. Run `baseline_qlora` in dry-run mode.
4. Train `baseline_qlora`.
5. Evaluate on `benchmark_holdout` and `verification_suite`.
6. Mine failures.
7. Retrain with `failure_aware` or `verifier_augmented`.
8. Compare runs and inspect win cases.

## Suggested Questions

- Did the fine-tuned model improve on hard calculus without regressing on simpler slices?
- Did verifier agreement improve, or only answer formatting?
- Which sources and synthetic families produce the highest-value examples?
- Does curriculum ordering improve stability or just latency?

## Useful Pairings

- `baseline_qlora` vs `calculus_specialist`
- `curriculum_specialist` vs `baseline_qlora`
- `failure_aware` vs latest strong baseline
- `verifier_augmented` with and without verification at inference time
