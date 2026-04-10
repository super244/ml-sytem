from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return REPO_ROOT


@dataclass(frozen=True)
class TrainingRunData:
    run_dir: Path
    dataset_data: dict
    model_data: dict
    final_metrics: dict
    metrics_df: pd.DataFrame


def load_training_run(run_dir: Path) -> TrainingRunData:
    """Load dataset report, model report, final metrics, and training_metrics.jsonl into a DataFrame."""
    run_dir = run_dir.resolve()
    with open(run_dir / "metrics/dataset_report.json") as f:
        dataset_data = json.load(f)
    with open(run_dir / "metrics/model_report.json") as f:
        model_data = json.load(f)
    with open(run_dir / "metrics/metrics.json") as f:
        final_metrics = json.load(f)
    training_metrics: list[dict] = []
    with open(run_dir / "logs/training_metrics.jsonl") as f:
        for line in f:
            training_metrics.append(json.loads(line.strip()))
    df = pd.DataFrame(training_metrics)
    return TrainingRunData(
        run_dir=run_dir,
        dataset_data=dataset_data,
        model_data=model_data,
        final_metrics=final_metrics,
        metrics_df=df,
    )
