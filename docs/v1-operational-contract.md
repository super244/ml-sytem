# V1 Operational Contract

This document defines the production-facing contract for the V1 control plane, web UI, and desktop shells.

## Runtime Modes

- Production mode is the default. Routes must report explicit degraded state when inventory or dependencies are unavailable.
- Demo content is only allowed when the matching flag is enabled:
  - Backend: `AI_FACTORY_DEMO_MODE=1`
  - Frontend: `NEXT_PUBLIC_AI_FACTORY_DEMO_MODE=1`
  - Native desktop: `AI_FACTORY_DESKTOP_DEMO_MODE=1`
- Simulated write paths are gated behind backend demo mode. In production, read-only inventory stays available where durable artifacts exist, and write-only simulation endpoints return an explicit disabled error instead of synthetic success.

## Backend Endpoints

- `GET /v1/status`
  Returns API metadata, instance counts, model inventory, and cache state.
  Contract: includes `status` (`available` or `degraded`) and `errors`.

- `GET /v1/models`
  Returns the live model registry derived from `inference/configs/model_registry.yaml`.

- `GET /v1/prompts`
  Returns the live prompt preset library.
  Contract: may include `status` and `errors` when the prompt library is degraded.

- `GET /v1/workspace`
  Returns the workspace inventory used by the launcher and dashboard.
  Contract: includes `status` (`available` or `degraded`) and `errors`.

- `GET /v1/instances`
  Returns the current managed instance inventory from the control plane projection.

- `POST /v1/instances/{id}/actions`
  Runs lifecycle actions such as `evaluate`, `open_inference`, `deploy`, `finetune`, or `re_evaluate`.

- `GET /v1/instances/{id}/metrics`
  Returns instance-scoped summary metrics plus time-series points.

- `GET /v1/instances/{id}/logs`
  Returns instance stdout and stderr paths plus tailed content.

- `GET /v1/agents/logs`
  Returns persisted swarm log events. In demo mode the backend may append simulated log traffic.

- `GET /v1/automl/sweeps`
  Returns durable sweep inventory from `data/automl/sweeps.jsonl`.
  Contract: includes `write_enabled`; launching simulated sweeps remains demo-only.

## Frontend Contract

- The frontend must revalidate metadata and inventory on an interval plus focus regain. Hard-coded model ids are not valid production defaults.
- In production mode:
  - model and prompt selection must derive from live backend metadata
  - actions are disabled when status or workspace metadata is degraded
  - fallback model and prompt content must not render unless frontend demo mode is enabled

## Desktop Contract

- Electron opens the canonical dashboard route: `http://127.0.0.1:3000/dashboard` unless `AI_FACTORY_DESKTOP_URL` overrides it.
- SwiftUI uses the same API routes as the web shell:
  - `GET /v1/status`
  - `GET /v1/instances`
  - `POST /v1/instances/{id}/actions`
  - `GET /v1/instances/{id}/metrics`
  - `GET /v1/agents/logs`
- The native shell must not synthesize logs or metrics in production mode.
- Lab shells should treat agent deployment and synthetic AutoML launches as demo-only affordances even when read-only inventory is available.

## Artifact Layout

- Control plane state: `artifacts/control_plane/`
- Managed instances: `artifacts/instances/<instance-id>/`
- Training runs: `artifacts/runs/<run-id>/`
- Inference telemetry: `artifacts/inference/telemetry/requests.jsonl`
- Dataset processing outputs:
  - `data/processed/manifest.json`
  - `data/processed/pack_summary.json`

## Packaging Expectations

- Wheels must discover packages automatically via setuptools package discovery.
- Any shipped package must be importable from an installed environment, not only from a source checkout.
- Tests are excluded from wheel discovery and from the repo-wide mypy target.
