# Subsystem dependency matrix

Runtime subsystems live in top-level packages: `data/`, `training/`, `evaluation/`, `inference/`. Shared contracts and orchestration live in `ai_factory/` (especially `ai_factory.core`).

## Allowed direction

```text
         ┌─────────────────┐
         │  ai_factory.*   │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
 data/       training/    evaluation/
                  │             │
                  └──────┬──────┘
                         ▼
                    inference/
```

- **Subsystems may import** `ai_factory` and third-party libraries.
- **Subsystems must not import sibling roots** (e.g. `training` must not import `inference`, `evaluation` must not import `training`). Exceptions are tracked in `tests/test_architecture_boundaries.py` if temporarily unavoidable.
- **`ai_factory.core` must not import** any of `data`, `training`, `evaluation`, or `inference` (see `test_ai_factory_core_does_not_import_runtime_subsystems`).

## Rationale

- Keeps training, API, and evaluation independently deployable.
- Prevents accidental coupling of GPU training stacks to the inference server and vice versa.
- Centralizes shared types and orchestration in `ai_factory.core` so boundaries stay explicit.
