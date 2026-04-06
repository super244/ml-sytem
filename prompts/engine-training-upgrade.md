# Engine And Training Upgrade Prompt

Upgrade the AI-Factory engine-adjacent training stack. Your objective is to forge a highly performant, deterministic, and scalable fine-tuning and alignment pipeline that respects system boundaries while maximizing hardware utilization.

## Primary Goal

Overhaul the local ML workflows, fine-tuning scripts, weight packaging, and performance-sensitive paths. Shift the focus from prototype scripts to a robust, configuration-driven training backend.

## Read This First (Mandatory Ingestion)

- `prompts/shared-repo-context.md`
- `training/README.md`
- `training/src/config.py` (Hyperparameter definition)
- `training/src/trainer.py` (The training loop)
- `training/src/workflows.py` (E2E job coordination)
- `training/src/modeling.py` (Model loading and PEFT injection)
- `training/src/packaging.py` (Weight export and conversion)
- `training/scripts/` (CLI entry points)
- `ai_factory_titan/README.md`
- `ai_factory_titan/src/`

## Scope & Execution Directives

1. **Robust Configuration & Tracking**:
   - Centralize all hyperparameters in strongly-typed Pydantic schemas within `config.py`.
   - Ensure all training runs emit standardized artifacts (loss curves, gradient norms, checkpoint metadata) that can be read by the `inference/` metrics API.
2. **PEFT & Distributed Training Alignment**:
   - Standardize QLoRA targets and FSDP/DeepSpeed configurations. Provide sane defaults for 1x GPU vs 4x GPU setups.
   - Ensure memory optimizations (gradient checkpointing, flash attention) are explicitly configurable.
3. **Engine-Adjacent Export (Titan Integration)**:
   - The packaging step must produce weights that are natively consumable by `ai_factory_titan` (e.g., converting Safetensors to GGUF format).
   - Establish clear boundaries where the Python training code hands off to the Rust runtime for inference-based evaluations during the training loop.
4. **Reproducibility**:
   - All dataset seeds, shuffling logic, and initial weight hashes must be explicitly logged to the central orchestration registry.

## Definition Of Done

- Training workflows are fully declarative, driven by Pydantic configuration objects.
- High-performance memory and compute optimizations are correctly implemented and documented.
- The weight export process seamlessly interfaces with the Titan inference engine format.
- Tests (or high-fidelity smoke checks) validate the end-to-end flow from data ingestion to model packaging.