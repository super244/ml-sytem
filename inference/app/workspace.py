from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml

from ai_factory.core.datasets import load_catalog, load_pack_summary
from ai_factory.core.discovery import list_training_runs, load_benchmark_registry
from ai_factory.core.foundation import build_foundation_catalog

REPO_ROOT = Path(__file__).resolve().parents[2]


def _has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _config_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def _command_recipe(
    recipe_id: str,
    title: str,
    description: str,
    command: str,
    category: str,
) -> dict[str, str]:
    return {
        "id": recipe_id,
        "title": title,
        "description": description,
        "command": command,
        "category": category,
    }


def _load_model_catalog(repo_root: Path) -> list[dict[str, Any]]:
    try:
        from inference.app.model_catalog import list_model_catalog

        return list_model_catalog(repo_root / "inference" / "configs" / "model_registry.yaml")
    except Exception:
        return []


def _load_orchestration_templates(repo_root: Path) -> list[dict[str, str]]:
    templates: list[dict[str, str]] = []
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        relative = path.relative_to(repo_root)
        instance = payload.get("instance") or {}
        experience = payload.get("experience") or {}
        templates.append(
            {
                "id": path.stem,
                "title": _config_title(path),
                "path": str(relative),
                "instance_type": str(instance.get("type", "unknown")),
                "user_level": str(experience.get("level", "hobbyist")),
                "orchestration_mode": str(payload.get("orchestration_mode", "single")),
                "command": f"ai-factory new --config {relative}",
            }
        )
    return templates


def build_workspace_overview(root: Path | None = None) -> dict[str, Any]:
    repo_root = (root or REPO_ROOT).resolve()
    catalog = load_catalog(repo_root / "data" / "catalog.json")
    packs = load_pack_summary(repo_root / "data" / "processed" / "pack_summary.json").get("packs", [])
    runs = list_training_runs(str(repo_root / "artifacts"))
    benchmarks = load_benchmark_registry(repo_root / "evaluation" / "benchmarks" / "registry.yaml")
    models = _load_model_catalog(repo_root)
    orchestration_templates = _load_orchestration_templates(repo_root)
    foundation = build_foundation_catalog(repo_root)

    training_profiles = []
    for path in sorted((repo_root / "training" / "configs" / "profiles").glob("*.yaml")):
        relative = path.relative_to(repo_root)
        training_profiles.append(
            {
                "id": path.stem,
                "title": _config_title(path),
                "path": str(relative),
                "dry_run_command": f"python -m training.train --config {relative} --dry-run",
                "train_command": f"python -m training.train --config {relative}",
            }
        )

    evaluation_configs = []
    for path in sorted((repo_root / "evaluation" / "configs").glob("*.yaml")):
        relative = path.relative_to(repo_root)
        evaluation_configs.append(
            {
                "id": path.stem,
                "title": _config_title(path),
                "path": str(relative),
                "run_command": f"python -m evaluation.evaluate --config {relative}",
            }
        )

    readiness_checks = [
        {
            "id": "python-deps",
            "label": "Python runtime deps",
            "ok": all(_has_package(name) for name in ("yaml", "fastapi", "pydantic")),
            "detail": "Checks for the config and API dependencies needed for local workflows.",
        },
        {
            "id": "ml-stack",
            "label": "Training and inference stack",
            "ok": all(_has_package(name) for name in ("torch", "transformers", "datasets", "sympy")),
            "detail": ("Covers the heavier packages required for training, evaluation, and symbolic verification."),
        },
        {
            "id": "frontend-install",
            "label": "Frontend dependencies",
            "ok": (repo_root / "frontend" / "node_modules").exists(),
            "detail": "The workspace UI is ready when frontend/node_modules is present.",
        },
        {
            "id": "processed-corpus",
            "label": "Processed corpus",
            "ok": (repo_root / "data" / "processed" / "manifest.json").exists(),
            "detail": ("Training and dataset browsing expect the processed manifest and pack summary outputs."),
        },
        {
            "id": "benchmark-registry",
            "label": "Benchmark registry",
            "ok": (repo_root / "evaluation" / "benchmarks" / "registry.yaml").exists(),
            "detail": ("Evaluation routes and comparisons depend on a discoverable benchmark registry."),
        },
    ]

    ready_count = sum(1 for item in readiness_checks if item["ok"])
    command_recipes = [
        _command_recipe(
            "doctor",
            "Workspace doctor",
            ("Inspect dependency readiness, dataset state, run discovery, and frontend installation in one pass."),
            "python scripts/doctor.py --json",
            "setup",
        ),
        _command_recipe(
            "refresh-lab",
            "Refresh lab",
            (
                "Regenerate data artifacts, validate the corpus, refresh notebooks, "
                "and optionally run dry validation/tests."
            ),
            "python scripts/refresh_lab.py",
            "orchestration",
        ),
        _command_recipe(
            "serve-api",
            "Serve API",
            ("Start the FastAPI layer that powers the solve workspace, compare lab, and metadata routes."),
            "uvicorn inference.app.main:app --reload",
            "serve",
        ),
        _command_recipe(
            "serve-frontend",
            "Serve frontend",
            "Launch the Next.js workspace against the local API.",
            "cd frontend && NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev",
            "serve",
        ),
        _command_recipe(
            "baseline-dry-run",
            "Baseline dry-run",
            ("Validate the default local specialist profile without entering a full training loop."),
            ("python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run"),
            "training",
        ),
        _command_recipe(
            "default-eval",
            "Default evaluation",
            "Run the main comparison config used for baseline-vs-finetuned benchmarking.",
            "python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml",
            "evaluation",
        ),
        _command_recipe(
            "orchestrated-finetune",
            "Orchestrated finetune",
            "Create a managed finetune instance that records progress, lineage, "
            "metrics, and follow-up recommendations.",
            "ai-factory new --config configs/finetune.yaml",
            "control-plane",
        ),
        _command_recipe(
            "cloud-finetune",
            "Cloud finetune",
            "Create a cloud-backed finetune instance with SSH profile resolution and remote execution.",
            "ai-factory new --config configs/finetune.yaml --environment cloud --cloud-profile default",
            "control-plane",
        ),
        _command_recipe(
            "managed-inference",
            "Managed inference sandbox",
            "Launch a child inference instance from an evaluated or finetuned branch.",
            "ai-factory inference <instance-id> --config configs/inference.yaml",
            "control-plane",
        ),
    ]

    orchestration_capabilities = [
        {
            "id": "shared-contracts",
            "title": "Shared control-plane contracts",
            "detail": "CLI, API, and future TUI/desktop layers consume the same instance manifests, "
            "progress state, recommendations, and child-instance lineage.",
        },
        {
            "id": "remote-ssh",
            "title": "Remote SSH orchestration",
            "detail": "Cloud profiles, SSH key selection, and SSH port-forward definitions are modeled "
            "directly in orchestration configs and instance metadata.",
        },
        {
            "id": "feedback-loop",
            "title": "Continuous feedback loop",
            "detail": "Training can recommend evaluation, evaluation can recommend deploy or more training, "
            "and bounded automatic child instances can carry those steps forward.",
        },
        {
            "id": "publish-hooks",
            "title": "Publish hooks",
            "detail": "Deployment hooks can target HuggingFace, Ollama, LM Studio, "
            "or custom APIs through the shared deployment pipeline.",
        },
        {
            "id": "control-center",
            "title": "Lifecycle control center",
            "detail": "The runs dashboard now acts as a control center: launch new branches, inspect lifecycle "
            "detail, open inference sandboxes, and prepare publish actions from the same surface.",
        },
        {
            "id": "plugin-registry",
            "title": "Extension-point registry",
            "detail": "Training methods, evaluation suites, and deployment targets are exposed as explicit "
            "extension points so every interface can discover the same backend capabilities.",
        },
        {
            "id": "live-state-manager",
            "title": "Live state manager",
            "detail": "The shared state layer projects orchestration metadata together with live progress and "
            "metric views, so CLI, TUI, API, and desktop surfaces stay aligned while jobs are running.",
        },
    ]

    return {
        "repo_root": str(repo_root),
        "summary": {
            "datasets": catalog.get("summary", {}).get("num_datasets", 0),
            "packs": len(packs),
            "models": len(models),
            "benchmarks": len(benchmarks),
            "runs": len(runs),
            "training_profiles": len(training_profiles),
            "evaluation_configs": len(evaluation_configs),
            "orchestration_templates": len(orchestration_templates),
            "interfaces": len(foundation.interfaces),
            "experience_tiers": len(foundation.experience_tiers),
            "extension_points": len(foundation.extension_points),
            "ready_checks": ready_count,
            "total_checks": len(readiness_checks),
        },
        "models": models,
        "readiness_checks": readiness_checks,
        "interfaces": [item.model_dump(mode="json") for item in foundation.interfaces],
        "experience_tiers": [item.model_dump(mode="json") for item in foundation.experience_tiers],
        "extension_points": [item.model_dump(mode="json") for item in foundation.extension_points],
        "command_recipes": command_recipes,
        "orchestration_capabilities": orchestration_capabilities,
        "orchestration_templates": orchestration_templates,
        "training_profiles": training_profiles,
        "evaluation_configs": evaluation_configs,
    }
