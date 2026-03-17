# Troubleshooting

## `datasets` import or download failures

Public dataset normalization depends on Hugging Face `datasets` and network access. If those are unavailable, skip normalization and continue with the local synthetic corpora.

## Dry-run passes but training fails on model load

Use dry-run to validate configs and tokenization first. If full training still fails, confirm local model availability, adapter settings, and VRAM budget.

## Empty benchmark results

Check that the referenced benchmark JSONL exists and that the path in `evaluation/benchmarks/registry.yaml` matches the latest processed outputs.

## Frontend cannot reach the API

Ensure `NEXT_PUBLIC_API_BASE_URL` points to the running FastAPI server and verify `/v1/health` responds.
