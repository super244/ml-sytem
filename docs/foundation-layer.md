# Foundation Layer

This repository already had strong subsystem implementations for data preparation, training, evaluation, and inference. The foundation layer standardizes how those subsystems are executed and observed so future CLI, TUI, web, and desktop surfaces can share one backend contract.

## Updated Repo Structure

```text
ai_factory/
├── cli.py                         primary CLI entrypoint
└── core/
    ├── config/                    orchestration config schema + loader
    ├── decisions/                 rule-based post-eval next-step agent
    ├── execution/                 local and SSH execution backends
    ├── instances/                 instance manifest, store, queries, manager
    ├── monitoring/                progress + metrics collection hooks
    ├── orchestration/             SQLite control plane, runs, tasks, attempts, events
    └── platform/                  shared container + runtime settings

configs/
├── prepare.yaml                   data preparation instance template
├── train.yaml                     managed training template
├── finetune.yaml                  managed fine-tuning template
├── eval.yaml                      managed evaluation template
├── inference.yaml                 managed inference template
├── deploy.yaml                    managed deployment template
└── report.yaml                    managed failure-analysis template

training/                          existing model training stack
evaluation/                        existing benchmark + scoring stack
inference/                         existing FastAPI inference stack
frontend/                          existing monitoring/control dashboard surface
artifacts/
├── control_plane/                 SQLite orchestration state
└── instances/                     compatibility projection per managed instance
```

## Core Contract

Every operation is represented as an instance manifest stored under `artifacts/instances/<instance-id>/instance.json`.

```json
{
  "id": "finetune-...",
  "type": "finetune",
  "status": "pending",
  "environment": {"kind": "local"},
  "config_path": "configs/finetune.yaml",
  "metrics_summary": {},
  "artifact_refs": {},
  "orchestration_run_id": "run-..."
}
```

The instance manager projects that manifest onto the durable control plane:

- one orchestration run
- one primary orchestration task
- zero or more retries, child tasks, and child instances

## Example Config Flow

- `configs/prepare.yaml` wraps `data/configs/processing.yaml`
- `configs/train.yaml` and `configs/finetune.yaml` wrap `training/configs/profiles/*.yaml`
- `configs/eval.yaml` wraps `evaluation/configs/*.yaml`
- `configs/inference.yaml` wraps the existing FastAPI service
- `configs/deploy.yaml` wraps provider-specific deployment commands

The orchestration template stays small. Existing subsystem configs remain the source of truth for model, dataset, and hyperparameter detail.

## Example Instance Lifecycle

1. `ai-factory new --config configs/finetune.yaml`
2. `InstanceManager.create_instance()` writes the instance manifest and config snapshot.
3. `OrchestrationService.ensure_run_for_instance()` creates a control-plane run and task.
4. `LocalExecutor` or `SshExecutor` starts the existing subsystem command.
5. Logs stream into `artifacts/instances/<id>/logs/`.
6. Monitoring collectors project metrics and progress back onto the instance manifest.
7. Evaluation completion writes a decision plus recommendations such as `retrain`, `finetune`, or `deploy`.
8. Optional follow-up instances are queued using the same contract.

## Minimal CLI Surface

```bash
ai-factory new --config configs/finetune.yaml
ai-factory list
ai-factory status <instance-id>
ai-factory logs <instance-id>
ai-factory evaluate <instance-id>
ai-factory deploy <instance-id> --target ollama
```

Additional operator commands already exposed by the CLI:

```bash
ai-factory tasks <instance-id-or-run-id>
ai-factory events <instance-id-or-run-id>
ai-factory watch <instance-id-or-run-id> --timeout 30
ai-factory retry <instance-id-or-run-id>
ai-factory cancel <instance-id-or-run-id>
ai-factory recommendations <instance-id>
```

Read-only terminal UI over the same backend:

```bash
ai-factory tui
```

## Local And Cloud Execution

- Local execution uses a detached Python runner that streams stdout/stderr into instance logs.
- Cloud execution uses SSH with key-based auth, optional port forwards, and the same instance contract.
- Cloud connection profiles are stored outside the repo, referenced by `EnvironmentSpec`, and written with best-effort restrictive local permissions.

## Monitoring Hooks

The monitoring layer already supports:

- training metrics and progress from run artifacts
- evaluation metrics from benchmark artifacts
- inference request telemetry, latency, token throughput, and cache hit summaries
- GPU snapshots via `nvidia-smi` when available

Current V1 boundaries:

- `train` and `finetune` instances currently share the same supervised pretrained-model adaptation pipeline.
- Training from scratch, RLHF-class loops, and transformer-topology construction are not yet implemented in the underlying training stack.

The contract is intentionally backend-first so the TUI, the existing web app, and a future desktop wrapper can all read the same data.
