# Atlas Math Lab: Plain-English Codebase Explanation

This document is for judges, reviewers, and non-specialists who want to look inside the repository and understand what each important part does.

The short version is:

- this project builds a math-specialized AI system
- it prepares training data
- it fine-tunes a language model for competitive math and calculus
- it serves the model through an API
- it evaluates the model carefully
- it wraps everything in a polished web app

In other words, this is not just "a model" and it is not just "a website." It is a full workflow.

## 1. The Big Idea

You can think of the repository as a small factory with several rooms:

- `data/` creates and cleans the training material
- `training/` teaches the model
- `inference/` lets the trained model answer questions
- `evaluation/` checks how well it performs
- `frontend/` is the user-facing product
- `docs/` explains the design
- `notebooks/` are the research lab notebooks
- `artifacts/` stores the outputs from experiments and runs

The project is called **Atlas Math Lab** because it is meant to feel like a serious research-and-product platform for advanced math reasoning.

## 2. If A Judge Only Opens A Few Files

If someone only has a few minutes, these are the best files to open first:

- `README.md`
  The main overview of the entire project.
- `quickstart.md`
  The easiest guide for setting up the environment, training, and running the app.
- `explanation.md`
  This file, which gives the plain-English tour.
- `docs/architecture.md`
  The best technical explanation of how the whole system is organized.
- `training/train.py`
  The main training entrypoint.
- `inference/app/main.py`
  The main API entrypoint.
- `evaluation/evaluate.py`
  The main evaluation entrypoint.
- `frontend/app/page.tsx`
  The starting point of the web app.

These files show the "story" of the system from start to finish.

## 3. Top-Level Files

### `README.md`

The main landing page for the repository. It explains what Atlas Math Lab is, what problems it solves, and what major capabilities are included.

### `quickstart.md`

A practical setup guide. It explains how to install dependencies, prepare data, train the model, run the API, launch the frontend, and evaluate the system on either a local machine or a cloud machine.

### `explanation.md`

This guided walkthrough for judges and non-professional reviewers.

### `tune-guide.md`

A more tuning-focused guide. This is helpful for someone who wants to understand how to adjust training behavior or experiment with model specialization.

### `Makefile`

A shortcut file that groups common commands into easy labels like "refresh the workspace," "run evaluation," or "check the latest run."

### `pyproject.toml`

The main Python project configuration file. It defines the Python package, dependencies, and tooling setup.

### `requirements.txt`

A simple dependency list for Python environments.

### `requirements-dev.txt`

Extra packages used for development tasks such as testing and code-quality checks.

### `.env.example`

A template showing the environment variables the project uses, such as where the model registry lives and where artifacts should be stored.

### `pytest.ini`

The basic test configuration for Python tests.

### `Dockerfile.api`

Build instructions for packaging the backend API into a container.

### `Dockerfile.frontend`

Build instructions for packaging the frontend web app into a container.

### `docker-compose.yml`

A simple way to bring up the API and frontend together in a containerized setup.

## 4. The Shared Core: `ai_factory/`

This folder contains the reusable "glue" that keeps the rest of the codebase consistent.

### `ai_factory/core/schemas.py`

Defines the important data shapes used throughout the project. This is how the system keeps records consistent across data generation, training, evaluation, and serving.

### `ai_factory/core/answers.py`

Contains answer-parsing and answer-checking logic. This is important because math models often produce long reasoning, but the system still needs to extract and verify the final answer.

### `ai_factory/core/artifacts.py`

Creates and organizes the standard folder layout for experiment outputs. This keeps runs neat and reproducible.

### `ai_factory/core/hashing.py`

Creates stable identifiers and fingerprints for data examples. This helps with deduplication and tracking.

### `ai_factory/core/io.py`

Handles reading and writing structured files like JSON and JSONL.

### `ai_factory/core/reports.py`

Helps generate human-readable reports and machine-readable output files.

### `ai_factory/core/tokens.py`

Contains token-related helpers, which matter when working with language models.

### `ai_factory/core/discovery.py`

Finds saved runs and benchmark definitions so the app and scripts can discover what already exists.

### `ai_factory/artifacts.py`

A compatibility wrapper related to artifact handling.

### `ai_factory/schemas.py`

A compatibility wrapper related to schemas.

## 5. The Data System: `data/`

This is where the model gets its "study material."

The data system does several jobs:

- creates synthetic math problems
- normalizes outside datasets into a shared format
- scores and filters examples
- builds train, eval, and test splits
- produces special packs for training and benchmarking

### Main entry files

#### `data/prepare_dataset.py`

The main dataset pipeline. It collects raw and generated examples, cleans them, splits them, builds reports, and writes the processed datasets used later by training and evaluation.

#### `data/catalog.py`

Builds and reads the project-wide catalog of datasets.

#### `data/catalog.json`

A generated snapshot of what datasets exist in the project and basic metadata about them.

#### `data/mine_failure_cases.py`

Takes bad model outputs and turns them into "failure replay" data that can be used to improve the next training round.

### Data configuration files

#### `data/configs/generation.yaml`

Controls how synthetic datasets are generated.

#### `data/configs/processing.yaml`

Controls how the full corpus is assembled, split, filtered, and packaged.

### Synthetic data generation

#### `data/generator/generate_calculus_datasets.py`

The top-level script that generates the project’s custom synthetic math datasets.

#### `data/synthesis/base.py`

Defines the common rules for dataset generators.

#### `data/synthesis/registry.py`

Keeps track of which generator families exist and how to call them.

#### `data/synthesis/families/derivatives.py`

Generates derivative-focused problems.

#### `data/synthesis/families/integrals.py`

Generates integration-focused problems.

#### `data/synthesis/families/limits_series.py`

Generates limits and series problems.

#### `data/synthesis/families/multivariable.py`

Generates multivariable calculus problems.

#### `data/synthesis/families/odes_optimization.py`

Generates differential equation and optimization problems.

#### `data/synthesis/families/olympiad_reasoning.py`

Generates more proof-like olympiad-style reasoning problems.

### Public dataset support

#### `data/public/registry.yaml`

Lists outside datasets the project knows how to work with.

#### `data/public/download_public_datasets.py`

Helps fetch public datasets.

#### `data/public/normalize_public_datasets.py`

Converts public datasets into the shared internal format so they can mix cleanly with synthetic data.

#### `data/adapters/base.py`

Defines the shared adapter interface for public datasets.

### Data building and packaging

#### `data/builders/corpus_builder.py`

Combines many data sources into a single usable corpus.

#### `data/builders/pack_registry.py`

Defines the special packs used for training and evaluation, such as hard-calculus packs and benchmark packs.

### Data quality and safety

#### `data/quality/difficulty.py`

Estimates how hard a problem is.

#### `data/quality/scoring.py`

Assigns a quality score to examples.

#### `data/quality/contamination.py`

Checks whether training data overlaps too much with benchmark data.

#### `data/quality/mining.py`

Finds useful hard cases and failure-driven examples.

#### `data/quality/stats.py`

Builds summary statistics about the dataset.

### Data utilities

#### `data/tools/validate_dataset.py`

Checks that dataset files follow the expected schema.

#### `data/tools/preview_dataset.py`

Lets a developer quickly inspect a few examples.

#### `data/tools/audit_dataset.py`

Produces an audit-style view of the data.

#### `data/tools/export_subset.py`

Exports a smaller filtered subset of examples.

#### `data/tools/deduplicate_simhash.py`

Finds near-duplicate questions.

#### `data/tools/build_benchmark_pack.py`

Builds benchmark-specific packs from the processed dataset.

### Important generated data files

#### `data/custom/custom_derivative_mastery.jsonl`

Generated derivative dataset.

#### `data/custom/custom_integral_arena.jsonl`

Generated integral dataset.

#### `data/custom/custom_limits_series_lab.jsonl`

Generated limits and series dataset.

#### `data/custom/custom_multivariable_studio.jsonl`

Generated multivariable dataset.

#### `data/custom/custom_odes_optimization_lab.jsonl`

Generated ODE and optimization dataset.

#### `data/custom/custom_olympiad_reasoning_studio.jsonl`

Generated olympiad-style dataset.

For each of these, there is also:

- a `.manifest.json` file that records metadata
- a `.md` data card that explains the dataset in human-readable form

### Important processed outputs

#### `data/processed/train.jsonl`

The main training split.

#### `data/processed/eval.jsonl`

The validation split used during model development.

#### `data/processed/test.jsonl`

The held-out test split.

#### `data/processed/normalized_all.jsonl`

The combined processed dataset before final splitting.

#### `data/processed/manifest.json`

The main record of what was built.

#### `data/processed/stats.json`

Summary statistics about the processed data.

#### `data/processed/pack_summary.json`

A list of the special derived packs the system created.

#### `data/processed/card.md`

A plain-language description of the processed corpus.

#### `data/processed/size_report.md`

A size and volume report for the dataset.

## 6. The Training System: `training/`

This is the part that actually teaches the model.

### `training/train.py`

The main training entrypoint. This is the file that loads a config, prepares the run folder, validates the setup, loads the model and tokenizer, trains, evaluates, and packages the outputs.

### Training profile files

These files are "named experiment presets." They make it easy to run different styles of training.

#### `training/configs/profiles/baseline_qlora.yaml`

The default baseline fine-tuning profile.

#### `training/configs/profiles/calculus_specialist.yaml`

A profile focused more strongly on calculus-heavy examples.

#### `training/configs/profiles/curriculum_specialist.yaml`

A profile that changes the order of training to make learning more gradual.

#### `training/configs/profiles/failure_aware.yaml`

A profile that uses failure cases from earlier runs.

#### `training/configs/profiles/verifier_augmented.yaml`

A profile that emphasizes examples with stronger checking and verification structure.

#### `training/configs/profiles/long_context.yaml`

A profile intended for longer reasoning traces.

#### `training/configs/profiles/fast_iteration_small_model.yaml`

A lighter profile for faster iteration.

### Legacy or helper training configs

#### `training/configs/lora_math.yaml`

A lower-level LoRA-related config.

#### `training/configs/train_math_calculus_specialist.yaml`

An older-style or alternate training config focused on calculus specialization.

#### `training/configs/train_math_curriculum.yaml`

An alternate config focused on curriculum training.

#### `training/configs/train_math_qlora.yaml`

An alternate QLoRA config.

### Runtime scaling configs

#### `training/configs/runtime/accelerate_zero2.yaml`

A runtime config for larger or more distributed training runs.

#### `training/configs/runtime/deepspeed_zero2.json`

Another runtime config for scaling training more efficiently.

### Training engine internals

#### `training/src/config.py`

Loads and validates experiment configs.

#### `training/src/data.py`

Builds the tokenized training datasets.

#### `training/src/modeling.py`

Loads the model, tokenizer, and training adapters.

#### `training/src/trainer.py`

Contains the training loop logic.

#### `training/src/collators.py`

Builds the batches that are fed into the model during training.

#### `training/src/callbacks.py`

Records logs and metrics during training.

#### `training/src/analysis.py`

Summarizes dataset and run information.

#### `training/src/packaging.py`

Packages finished model artifacts for serving and inspection.

#### `training/src/validation.py`

Implements dry-run validation so the user can check setup without spending time or GPU resources.

#### `training/src/comparison.py`

Helps compare training runs.

### Training helper scripts

#### `training/scripts/package_adapter.py`

Packages a trained adapter.

#### `training/scripts/export_merged_model.py`

Exports a merged model artifact after adapter training.

#### `training/scripts/compare_runs.py`

Compares different runs side by side.

#### `training/scripts/run_train.sh`

Simple shell script to launch training.

#### `training/scripts/run_full_pipeline.sh`

Runs a broader end-to-end workflow.

#### `training/scripts/run_ablation_suite.sh`

Runs a set of experiment comparisons to test what matters most.

## 7. The Inference System: `inference/`

This is the serving layer. It turns a trained model into an app users can talk to.

### `inference/app/main.py`

The main FastAPI app. It registers the API routes and turns the backend into a web service.

### `inference/app/config.py`

Loads runtime settings like paths and environment variables.

### `inference/app/dependencies.py`

Wires together the reusable services used by the API.

### `inference/app/model_loader.py`

Loads available model variants using the model registry.

### `inference/app/prompts.py`

Loads and manages prompt presets that shape how the model reasons.

### `inference/app/generation.py`

Defines generation parameters and generation behavior.

### `inference/app/schemas.py`

Defines the request and response shapes for the API.

### `inference/app/cache.py`

Handles local caching of model responses.

### `inference/app/telemetry.py`

Handles request logging and usage tracking.

### `inference/app/tools.py`

Contains helper logic such as safe calculator-style support.

### `inference/app/metadata.py`

Builds metadata about models, datasets, runs, and prompts for the UI and API.

### `inference/app/dashboard.py`

Convenience accessors for metadata used by higher-level tooling.

### API route files

#### `inference/app/routers/health.py`

Simple health and status endpoints.

#### `inference/app/routers/metadata.py`

Endpoints for models, datasets, benchmarks, prompts, and runs.

#### `inference/app/routers/generation.py`

Endpoints for generation, verification, batch generation, and model comparison.

### Service files

#### `inference/app/services/generation_service.py`

The main service that produces answers, runs sampling, reranking, and verification steps.

#### `inference/app/services/metadata_service.py`

The service that gathers metadata for the frontend and API.

### Inference configuration files

#### `inference/configs/model_registry.yaml`

The list of available model variants and where they live.

#### `inference/configs/prompt_presets.yaml`

The list of prompt styles the app can use.

## 8. The Evaluation System: `evaluation/`

This is the grading room. It checks how good the model really is.

### `evaluation/evaluate.py`

The main evaluation runner. It loads a benchmark, asks two model variants to answer it, scores the outputs, and writes reports.

### `evaluation/benchmark_registry.py`

Finds the benchmark files used during evaluation.

### `evaluation/metrics.py`

Calculates the main scores, such as answer correctness and verification behavior.

### `evaluation/error_taxonomy.py`

Organizes failure types into categories, which makes mistakes easier to analyze.

### `evaluation/reporting.py`

Writes summaries and reports from evaluation runs.

### `evaluation/analysis/analyze_failures.py`

Looks closely at failure cases after evaluation.

### Benchmark definitions

#### `evaluation/benchmarks/registry.yaml`

The registry of benchmark datasets.

### Evaluation configs

#### `evaluation/configs/base_vs_finetuned.yaml`

Compares the baseline model against the fine-tuned model.

#### `evaluation/configs/verifier_on_off.yaml`

Tests the effect of answer verification features.

#### `evaluation/configs/curriculum_ablation.yaml`

Tests the effect of curriculum training choices.

#### `evaluation/configs/source_ablation_calculus_only.yaml`

Tests how specific source choices affect results.

#### `evaluation/configs/run_vs_run_template.yaml`

A template for comparing any two runs.

#### `evaluation/configs/eval_math.yaml`

A general evaluation config.

## 9. The Frontend Product: `frontend/`

This is the user-facing product. It is the part judges can see and interact with directly.

### `frontend/package.json`

Defines the frontend dependencies and commands.

### `frontend/app/layout.tsx`

The top-level page layout for the web app.

### `frontend/app/globals.css`

The main styling file. It controls the look and feel of the product.

### `frontend/app/page.tsx`

The main solve page entrypoint.

### `frontend/app/compare/page.tsx`

The model comparison page.

### `frontend/app/datasets/page.tsx`

The dataset explorer page.

### `frontend/app/benchmarks/page.tsx`

The benchmark explorer page.

### `frontend/app/runs/page.tsx`

The experiment and run viewer page.

### Main frontend components

#### `frontend/components/chat-shell.tsx`

The main interactive solve workspace.

#### `frontend/components/compare-lab.tsx`

The side-by-side model comparison UI.

#### `frontend/components/datasets-view.tsx`

The UI for browsing datasets and packs.

#### `frontend/components/benchmarks-view.tsx`

The UI for browsing benchmark definitions.

#### `frontend/components/runs-view.tsx`

The UI for browsing recorded training runs.

#### `frontend/components/math-block.tsx`

Renders math content cleanly.

### Layout and navigation pieces

#### `frontend/components/layout/app-shell.tsx`

The main application frame.

#### `frontend/components/layout/app-nav.tsx`

The navigation bar across the app.

### Reusable UI pieces

#### `frontend/components/ui/page-header.tsx`

Shared page header used across views.

#### `frontend/components/ui/state-panel.tsx`

Shared loading, error, and empty-state component.

#### `frontend/components/panels/candidate-inspector.tsx`

Lets users inspect multiple answer candidates from the model.

#### `frontend/components/panels/metric-badge.tsx`

Small metric display chips used throughout the UI.

#### `frontend/components/panels/model-chip.tsx`

Compact cards describing available models.

### Frontend logic and helpers

#### `frontend/hooks/use-lab-metadata.ts`

Loads shared metadata into the frontend, such as runs, datasets, and models.

#### `frontend/lib/api.ts`

The typed client used by the frontend to talk to the backend API.

#### `frontend/lib/demo-content.ts`

Fallback examples and demo content used when the API is not yet available.

#### `frontend/lib/formatting.ts`

Formatting helpers for counts, latencies, and sizes.

#### `frontend/lib/options.ts`

Shared option lists for things like solver modes and difficulty settings.

### Frontend configuration

#### `frontend/next.config.mjs`

The Next.js configuration file.

#### `frontend/tsconfig.json`

The TypeScript configuration file.

#### `frontend/next-env.d.ts`

Type support for Next.js.

## 10. The Notebook Lab: `notebooks/`

These notebooks are the "research journal" of the project. They make the system feel like a real machine learning lab rather than just a software demo.

### `notebooks/build_notebooks.py`

Rebuilds or regenerates the notebook set.

### `notebooks/README.md`

Explains the purpose of the notebook area.

### Notebook files

#### `notebooks/00_dataset_landscape.ipynb`

Explores the overall dataset collection.

#### `notebooks/01_calculus_generator_lab.ipynb`

Explores how custom calculus problems are generated.

#### `notebooks/02_transformer_exploration.ipynb`

Looks at model and tokenizer behavior.

#### `notebooks/03_base_vs_finetuned_inference.ipynb`

Shows the difference between the base model and the fine-tuned model.

#### `notebooks/04_evaluation_win_case_browser.ipynb`

Highlights strong model wins.

#### `notebooks/05_lora_experiment_board.ipynb`

Tracks LoRA-related experiments.

#### `notebooks/06_public_dataset_normalization.ipynb`

Explains how public datasets are normalized.

#### `notebooks/07_dataset_quality_audit.ipynb`

Checks the quality of data.

#### `notebooks/08_prompt_optimization_lab.ipynb`

Explores prompt wording and prompting strategies.

#### `notebooks/09_reranking_self_consistency.ipynb`

Explores multi-sample reasoning and reranking.

#### `notebooks/10_verifier_analysis.ipynb`

Examines verification behavior.

#### `notebooks/11_benchmark_slice_analysis.ipynb`

Analyzes different slices of benchmark performance.

#### `notebooks/12_error_driven_retraining.ipynb`

Shows how mistakes can be turned into better future training.

#### `notebooks/13_run_artifact_explorer.ipynb`

Explores experiment outputs and run artifacts.

## 11. The Documentation: `docs/`

These files make the project easier to understand, extend, and judge.

### `docs/architecture.md`

The core system design document.

### `docs/data-system.md`

Explains the data pipeline in detail.

### `docs/training-system.md`

Explains the training design.

### `docs/inference-system.md`

Explains the serving layer and runtime behavior.

### `docs/evaluation-system.md`

Explains the evaluation methodology.

### `docs/api-guide.md`

Shows how the API is used.

### `docs/benchmark-guide.md`

Explains benchmark packs and evaluation slices.

### `docs/experiment-playbook.md`

Explains how to run experiments thoughtfully.

### `docs/experiments.md`

A broader experiments-oriented document.

### `docs/runbook.md`

The practical operating guide for running the whole system.

### `docs/deployment-guide.md`

Explains how to deploy the app or demo stack.

### `docs/notebook-guide.md`

Explains the notebook lab.

### `docs/contributor-guide.md`

Explains how someone can work on the project safely.

### `docs/reproducibility.md`

Explains how the project keeps experiments reproducible.

### `docs/troubleshooting.md`

Common problems and fixes.

### `docs/model-card.md`

A model card explaining the model’s purpose and behavior.

### `docs/roadmap.md`

Future directions for the project.

## 12. The Operator Scripts: `scripts/`

These are helper tools that make the repo easier to operate.

### `scripts/common.py`

Shared helper utilities used by the other scripts.

### `scripts/doctor.py`

Checks whether the workspace is healthy and ready to use.

### `scripts/refresh_lab.py`

Runs a convenient refresh cycle for rebuilding datasets, validating the setup, and refreshing the project state.

### `scripts/latest_run.py`

Shows the latest discovered training run in a compact format.

### `scripts/api_smoke.py`

Performs a quick smoke-check on the API when the server is running.

## 13. The Tests: `tests/`

This folder contains focused checks for the most important logic.

### `tests/test_core_answers.py`

Tests the answer extraction and verification helpers.

### `tests/test_data_processing.py`

Tests the data pipeline logic.

### `tests/test_evaluation_metrics.py`

Tests the evaluation scoring behavior.

### `tests/test_inference_api.py`

Tests the API layer.

### `tests/test_training_config.py`

Tests the training configuration logic.

## 14. The Output Folder: `artifacts/`

This is where the project stores the results of experiments and model runs.

It is important because it shows that the system is not only source code. It also records evidence of runs, metrics, reports, and packaged models.

Important subfolders include:

- `artifacts/runs/`
  Per-run outputs such as manifests, metrics, and summaries.
- `artifacts/models/`
  Packaged model artifacts ready for later use.
- `artifacts/inference/`
  Inference cache and telemetry data.

## 15. What Is Generated Versus Handwritten

Some files are written by developers, and some are produced by the system itself.

Mostly handwritten:

- Python source files
- TypeScript and CSS files
- YAML config files
- Markdown docs

Mostly generated:

- `.jsonl` datasets
- `.manifest.json` files
- processed data outputs
- notebook artifacts
- run artifacts under `artifacts/`

This matters because judges may see many files that are evidence of the pipeline working, not necessarily hand-authored code.

## 16. Why The Structure Matters

The structure of the repo shows that the project was designed like a real ML product:

- the data work is separated from the training logic
- the training logic is separated from the serving layer
- the serving layer is separated from the UI
- the evaluation system is first-class, not an afterthought
- documentation and notebooks are included so the work is explainable

That separation is one of the strongest signals of engineering maturity in the repository.

## 17. Final One-Sentence Summary

Atlas Math Lab is a complete end-to-end math AI platform where data creation, model tuning, evaluation, backend serving, and frontend product experience are all built as one coherent system.
