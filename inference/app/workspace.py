from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml

from ai_factory.core.datasets import inspect_json_asset, load_catalog, load_pack_summary
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
    try:
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
    except Exception:
        pass
    return templates


def _load_training_profiles(repo_root: Path) -> list[dict[str, str]]:
    training_profiles = []
    try:
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
    except Exception:
        pass
    return training_profiles


def _load_evaluation_configs(repo_root: Path) -> list[dict[str, str]]:
    evaluation_configs = []
    try:
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
    except Exception:
        pass
    return evaluation_configs


def _build_readiness_checks(repo_root: Path) -> list[dict[str, Any]]:
    catalog_status = inspect_json_asset(repo_root / "data" / "catalog.json")
    manifest_status = inspect_json_asset(repo_root / "data" / "processed" / "manifest.json")
    pack_summary_status = inspect_json_asset(repo_root / "data" / "processed" / "pack_summary.json")

    return [
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
            "id": "dataset-catalog",
            "label": "Dataset catalog",
            "ok": catalog_status["ok"],
            "detail": (
                "Dataset metadata is available for the workspace and API summaries."
                if catalog_status["ok"]
                else catalog_status["detail"]
            ),
        },
        {
            "id": "processed-corpus",
            "label": "Processed corpus",
            "ok": manifest_status["ok"] and pack_summary_status["ok"],
            "detail": (
                "Training and dataset browsing expect valid processed manifest and pack summary outputs."
                if manifest_status["ok"] and pack_summary_status["ok"]
                else (f"manifest: {manifest_status['detail']}; pack summary: {pack_summary_status['detail']}")
            ),
        },
        {
            "id": "benchmark-registry",
            "label": "Benchmark registry",
            "ok": (repo_root / "evaluation" / "benchmarks" / "registry.yaml").exists(),
            "detail": ("Evaluation routes and comparisons depend on a discoverable benchmark registry."),
        },
    ]


def _build_command_recipes() -> list[dict[str, str]]:
    return [
        _command_recipe(
            "doctor",
            "Workspace doctor",
            ("Inspect dependency readiness, dataset state, run discovery, and frontend installation in one pass."),
            "ai-factory doctor --json",
            "setup",
        ),
        _command_recipe(
            "refresh-lab",
            "Refresh lab",
            (
                "Regenerate data artifacts, validate the corpus, refresh notebooks, "
                "and optionally run dry validation/tests."
            ),
            "ai-factory refresh-lab",
            "orchestration",
        ),
        _command_recipe(
            "serve-api",
            "Serve API",
            ("Start the FastAPI layer that powers the solve workspace, compare lab, and metadata routes."),
            "ai-factory serve --host 127.0.0.1 --port 8000",
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
            "train-preflight",
            "Training preflight",
            "Validate training inputs, artifacts, hardware, and tokenizer readiness before a real launch.",
            "ai-factory train-preflight --config training/configs/profiles/failure_aware.yaml",
            "training",
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
        _command_recipe(
            "titan-status",
            "Titan hardware status",
            "Inspect the active Titan backend, unified-memory profile, and fallback mode.",
            "ai-factory titan status --write-hardware-doc",
            "hardware",
        ),
    ]


def _build_orchestration_capabilities() -> list[dict[str, str]]:
    return [
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


def build_workspace_overview(root: Path | None = None) -> dict[str, Any]:
    """Build workspace overview with optimized loading and caching."""
    repo_root = (root or REPO_ROOT).resolve()
    errors: list[str] = []
    catalog_status = inspect_json_asset(repo_root / "data" / "catalog.json")
    pack_summary_status = inspect_json_asset(repo_root / "data" / "processed" / "pack_summary.json")

    if catalog_status["ok"]:
        try:
            catalog = load_catalog(repo_root / "data" / "catalog.json")
        except Exception as exc:
            errors.append(f"catalog: {exc}")
            catalog = {"summary": {"num_datasets": 0}}
    else:
        errors.append(f"catalog: {catalog_status['detail']}")
        catalog = {"summary": {"num_datasets": 0}}

    if pack_summary_status["ok"]:
        try:
            packs = load_pack_summary(repo_root / "data" / "processed" / "pack_summary.json").get("packs", [])
        except Exception as exc:
            errors.append(f"pack summary: {exc}")
            packs = []
    else:
        errors.append(f"pack summary: {pack_summary_status['detail']}")
        packs = []

    try:
        runs = list_training_runs(str(repo_root / "artifacts"))
    except Exception as exc:
        errors.append(f"training runs: {exc}")
        runs = []

    try:
        benchmarks = load_benchmark_registry(repo_root / "evaluation" / "benchmarks" / "registry.yaml")
    except Exception as exc:
        errors.append(f"benchmark registry: {exc}")
        benchmarks = []

    try:
        foundation = build_foundation_catalog(repo_root)
        interfaces = [item.model_dump(mode="json") for item in foundation.interfaces]
        experience_tiers = [item.model_dump(mode="json") for item in foundation.experience_tiers]
        extension_points = [item.model_dump(mode="json") for item in foundation.extension_points]
    except Exception as exc:
        errors.append(f"foundation catalog: {exc}")
        interfaces = []
        experience_tiers = []
        extension_points = []

    try:
        models = _load_model_catalog(repo_root)
    except Exception as exc:
        errors.append(f"model catalog: {exc}")
        models = []
    try:
        orchestration_templates = _load_orchestration_templates(repo_root)
    except Exception as exc:
        errors.append(f"orchestration templates: {exc}")
        orchestration_templates = []
    try:
        training_profiles = _load_training_profiles(repo_root)
    except Exception as exc:
        errors.append(f"training profiles: {exc}")
        training_profiles = []
    try:
        evaluation_configs = _load_evaluation_configs(repo_root)
    except Exception as exc:
        errors.append(f"evaluation configs: {exc}")
        evaluation_configs = []
    readiness_checks = _build_readiness_checks(repo_root)
    command_recipes = _build_command_recipes()
    orchestration_capabilities = _build_orchestration_capabilities()

    ready_count = sum(1 for item in readiness_checks if item["ok"])

    return {
        "status": "available" if not errors else "degraded",
        "errors": errors,
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
            "interfaces": len(interfaces),
            "experience_tiers": len(experience_tiers),
            "extension_points": len(extension_points),
            "ready_checks": ready_count,
            "total_checks": len(readiness_checks),
        },
        "models": models,
        "readiness_checks": readiness_checks,
        "interfaces": interfaces,
        "experience_tiers": experience_tiers,
        "extension_points": extension_points,
        "command_recipes": command_recipes,
        "orchestration_capabilities": orchestration_capabilities,
        "orchestration_templates": orchestration_templates,
        "training_profiles": training_profiles,
        "evaluation_configs": evaluation_configs,
    }


def build_workspace_overview_fast(root: Path | None = None) -> dict[str, Any]:
    """Fast version that returns minimal data for quick responses."""
    repo_root = (root or REPO_ROOT).resolve()
    errors: list[str] = []

    readiness_checks = _build_readiness_checks(repo_root)
    ready_count = sum(1 for item in readiness_checks if item["ok"])
    try:
        model_count = len(_load_model_catalog(repo_root))
    except Exception as exc:
        errors.append(f"model catalog: {exc}")
        model_count = 0
    try:
        template_count = len(_load_orchestration_templates(repo_root))
    except Exception as exc:
        errors.append(f"orchestration templates: {exc}")
        template_count = 0
    try:
        training_profile_count = len(_load_training_profiles(repo_root))
    except Exception as exc:
        errors.append(f"training profiles: {exc}")
        training_profile_count = 0
    try:
        evaluation_config_count = len(_load_evaluation_configs(repo_root))
    except Exception as exc:
        errors.append(f"evaluation configs: {exc}")
        evaluation_config_count = 0

    return {
        "status": "available" if not errors else "degraded",
        "errors": errors,
        "repo_root": str(repo_root),
        "summary": {
            "datasets": 0,
            "packs": 0,
            "models": model_count,
            "benchmarks": 0,
            "runs": 0,
            "training_profiles": training_profile_count,
            "evaluation_configs": evaluation_config_count,
            "orchestration_templates": template_count,
            "interfaces": 0,
            "experience_tiers": 0,
            "extension_points": 0,
            "ready_checks": ready_count,
            "total_checks": len(readiness_checks),
        },
        "readiness_checks": readiness_checks,
        "models": [],
        "interfaces": [],
        "experience_tiers": [],
        "extension_points": [],
        "command_recipes": [],
        "orchestration_capabilities": [],
        "orchestration_templates": [],
        "training_profiles": [],
        "evaluation_configs": [],
    }
