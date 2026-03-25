from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from ai_factory.core.io import write_json, write_markdown
from data.quality.stats import compute_record_stats
from training.src.data import load_jsonl


def count_parameters(model: Any) -> dict[str, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def dataset_summary(path: str, max_preview: int = 3) -> dict[str, Any]:
    rows = load_jsonl(path)
    lengths = [len(row.get("question", "")) + len(row.get("solution", "")) for row in rows]
    preview = rows[:max_preview]
    return {
        "path": path,
        "num_rows": len(rows),
        "avg_question_solution_chars": mean(lengths) if lengths else 0.0,
        "preview_ids": [row.get("id") for row in preview],
        "stats": compute_record_stats(rows),
    }


def write_run_summary(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# {summary.get('run_name', 'Training Run')}",
        "",
        f"- Profile: `{summary.get('profile_name', 'unknown')}`",
        f"- Base model: `{summary.get('base_model', 'unknown')}`",
        f"- Train rows: `{summary.get('train_rows', 0)}`",
        f"- Eval rows: `{summary.get('eval_rows', 0)}`",
        "",
        "## Parameter Report",
        "",
        json.dumps(summary.get("parameter_report", {}), indent=2),
        "",
        "## Metrics",
        "",
        json.dumps(summary.get("metrics", {}), indent=2),
        "",
    ]
    write_markdown(path, "\n".join(lines))


__all__ = ["count_parameters", "dataset_summary", "write_json", "write_markdown", "write_run_summary"]
