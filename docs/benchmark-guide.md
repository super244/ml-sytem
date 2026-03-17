# Benchmark Guide

Benchmarks are declared explicitly in `evaluation/benchmarks/registry.yaml` and should represent held-out or audit-oriented slices rather than ad hoc training leftovers.

## Current Benchmarks

- `benchmark_holdout`: canonical held-out pack
- `core_eval`: default eval split
- `calculus_hard`: hard calculus-focused slice
- `verification_suite`: step-check-rich verification slice

## Adding A Benchmark

1. Build or curate the dataset as canonical `v2` records.
2. Write it to a stable JSONL path.
3. Add an entry to the benchmark registry.
4. Optionally add a dedicated evaluation config if the benchmark deserves a recurring report.

## Good Benchmark Hygiene

- keep benchmark records out of training sources or guard them with contamination checks
- preserve source lineage
- document why the slice exists
- prefer stable manifests and cards for long-lived slices
