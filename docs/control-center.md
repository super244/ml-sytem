# Control Center Foundation

The Phase 1 control center turns the existing repo into a lifecycle-oriented management surface without replacing the current training, evaluation, inference, or deployment systems.

## Shared Lifecycle Contract

The foundation lives in the shared instance manifest and orchestration config:

- `InstanceManifest.lifecycle` carries training origin, learning mode, source model, architecture hints, evaluation suite intent, deployment targets, and the current lifecycle stage.
- `OrchestrationConfig.lifecycle` lets config templates define those defaults once so the CLI, API, TUI, and web dashboard see the same lifecycle metadata.
- `FeedbackRecommendation` and `DecisionResult` remain rules-first and deterministic, but now support `re_evaluate` and `open_inference` outcomes in addition to retrain, finetune, and deploy.

## Product Surfaces

Phase 1 adds or expands four connected surfaces:

- CLI: `ai-factory new` accepts lifecycle-oriented flags such as user level, origin, learning mode, architecture hints, evaluation suite, and deployment targets. `ai-factory inference <instance-id>` launches a managed inference child.
- TUI: the detail rail now shows lifecycle stage, origin/mode, and the next recommendation.
- API: richer instance creation requests plus a first-class inference child route keep the control plane reusable by the frontend and any future desktop wrapper.
- Web UI: the runs page becomes a control center with a launch form, richer lifecycle cards, and per-instance detail pages showing actions, metric trends, lineage, logs, and event history.

## Child Lifecycle Branches

Evaluation, deployment, and inference are all modeled as managed child instances. Child instances inherit lifecycle context from their parent branch and then override only the stage-specific fields they need.

- evaluation children inherit the branch context and attach an evaluation suite
- inference children generate a model registry overlay so a branch can be inspected without mutating the shared base registry
- deploy children inherit deployment targets and source artifacts from the evaluated branch

## Desktop Wrapper Direction

The repo is now better positioned for a macOS desktop app because the control center is split cleanly:

- the orchestration and lifecycle logic stays in Python
- the management UI stays in the Next.js frontend
- the API remains the seam between them

The thinnest future macOS wrapper is a small shell around the existing web UI and local API process. A Tauri shell is the best fit if the goal is low overhead and native-feeling distribution. Electron is still viable if richer browser-process tooling becomes more important than bundle size.

Phase 1 deliberately does not ship that wrapper yet. The goal here is to make the existing surfaces composable enough that adding one later is a packaging decision, not a rewrite.
