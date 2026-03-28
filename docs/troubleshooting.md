# Troubleshooting

## Orchestration Runtime

- If `ai-factory status` looks stale, inspect `ai-factory events <instance-id>` and `ai-factory tasks <instance-id>` to see the task-level state and most recent heartbeat.
- If a task is stuck in `retry_waiting`, the backoff window has not elapsed yet. Use `ai-factory watch <instance-id>` to monitor the run or retry manually with `ai-factory retry <instance-id>`.
- If a task is marked `dead_lettered`, inspect `artifacts/instances/<instance-id>/logs/` plus `artifacts/control_plane/events.jsonl` before retrying.
- If heartbeats stop updating for a running task, verify the child process is still alive and that `artifacts/control_plane/control_plane.db` is writable.

## `datasets` import or download failures

Public dataset normalization depends on Hugging Face `datasets` and network access. If those are unavailable, skip normalization and continue with the local synthetic corpora.

## Dry-run passes but training fails on model load

Use dry-run to validate configs and tokenization first. If full training still fails, confirm local model availability, adapter settings, and VRAM budget.

## Empty benchmark results

Check that the referenced benchmark JSONL exists and that the path in `evaluation/benchmarks/registry.yaml` matches the latest processed outputs.

## Frontend cannot reach the API

Ensure `NEXT_PUBLIC_API_BASE_URL` points to the running FastAPI server and verify `/v1/health` responds.
