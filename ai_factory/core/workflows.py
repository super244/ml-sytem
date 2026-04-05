from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ai_factory.core.model_scales import DEFAULT_FOUNDATION_MODEL

DEFAULT_BASE_MODEL = DEFAULT_FOUNDATION_MODEL


@dataclass(frozen=True)
class WorkflowLayout:
    root: Path
    datasets_dir: Path
    configs_dir: Path
    reports_dir: Path


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[2]


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "run"


def parse_csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_workflow_layout(
    workflow_name: str,
    run_name: str,
    *,
    repo_root: str | Path | None = None,
) -> WorkflowLayout:
    root = resolve_repo_root(repo_root) / "artifacts" / "workflows" / workflow_name / slugify(run_name)
    datasets_dir = root / "datasets"
    configs_dir = root / "configs"
    reports_dir = root / "reports"
    for path in (datasets_dir, configs_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return WorkflowLayout(root=root, datasets_dir=datasets_dir, configs_dir=configs_dir, reports_dir=reports_dir)
