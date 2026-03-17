# Reproducibility

Atlas Math Lab uses manifests and config snapshots to make runs easier to replay and compare.

## Reproducibility Anchors

- config files and config hashes
- build ids
- git SHA capture when available
- dataset manifests with file hashes and row counts
- run manifests and Markdown summaries
- JSONL metrics and telemetry

## Repeatability Tips

- keep generated data configs under version control
- prefer dry-run before expensive training
- preserve `artifacts/runs/<run_id>/manifest.json`
- avoid mutating benchmark files in place
- refresh notebooks from the builder rather than editing them by hand
