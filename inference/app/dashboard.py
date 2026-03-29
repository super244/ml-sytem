from __future__ import annotations

from ai_factory.core.discovery import list_training_runs as _list_training_runs
from inference.app.dependencies import get_metadata_service


def dataset_dashboard() -> dict:
    return get_metadata_service().dataset_dashboard()


def prompt_library(limit: int = 12) -> dict:
    return get_metadata_service().prompt_library(limit=limit)


def list_training_runs(artifacts_dir: str = "artifacts") -> list[dict]:
    return _list_training_runs(artifacts_dir)
