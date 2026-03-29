from __future__ import annotations

import argparse
import importlib.util

from common import emit_payload, repo_root

from ai_factory.core.discovery import (
    latest_training_run,
    list_training_runs,
    load_benchmark_registry,
)
from data.catalog import load_catalog, load_pack_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the local AI-Factory environment and artifact state."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    args = parse_args()
    root = repo_root()
    catalog = load_catalog(root / "data" / "catalog.json")
    packs = load_pack_summary(root / "data" / "processed" / "pack_summary.json").get("packs", [])
    runs = list_training_runs(str(root / "artifacts"))
    benchmarks = load_benchmark_registry(root / "evaluation" / "benchmarks" / "registry.yaml")
    frontend_ready = (root / "frontend" / "node_modules").exists()

    recommended_next_steps = [
        "python scripts/refresh_lab.py",
        "uvicorn inference.app.main:app --reload",
        "python scripts/api_smoke.py",
    ]
    if runs:
        recommended_next_steps.append("python scripts/latest_run.py")
    else:
        recommended_next_steps.append(
            "python -m training.train --config "
            "training/configs/profiles/baseline_qlora.yaml --dry-run"
        )
    if not frontend_ready:
        recommended_next_steps.append("cd frontend && npm install")

    payload = {
        "repo_root": str(root),
        "python_packages": {
            "yaml": has_package("yaml"),
            "pytest": has_package("pytest"),
            "torch": has_package("torch"),
            "transformers": has_package("transformers"),
            "datasets": has_package("datasets"),
            "sympy": has_package("sympy"),
        },
        "data": {
            "catalog_present": (root / "data" / "catalog.json").exists(),
            "processed_manifest_present": (root / "data" / "processed" / "manifest.json").exists(),
            "num_catalog_datasets": catalog.get("summary", {}).get("num_datasets", 0),
            "num_derived_packs": len(packs),
        },
        "artifacts": {
            "num_runs": len(runs),
            "latest_run": (latest_training_run(runs) or {}).get("run_name"),
        },
        "evaluation": {
            "num_benchmarks": len(benchmarks),
        },
        "frontend": {
            "package_json_present": (root / "frontend" / "package.json").exists(),
            "node_modules_present": frontend_ready,
        },
        "recommended_next_steps": recommended_next_steps,
    }
    emit_payload(payload, as_json=args.json)


if __name__ == "__main__":
    main()
