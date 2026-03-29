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


def _section(title: str, body: str) -> list[str]:
    return [f"## {title}", "", body.strip(), ""]


def _json_block(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"


def write_run_summary(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [f"# {summary.get('run_name', 'Training Run')}", ""]
    overview = [
        f"- Profile: `{summary.get('profile_name', 'unknown')}`",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Base model: `{summary.get('base_model', 'unknown')}`",
        f"- Train rows: `{summary.get('train_rows', 0)}`",
        f"- Eval rows: `{summary.get('eval_rows', 0)}`",
    ]
    if summary.get("run_dir"):
        overview.append(f"- Run dir: `{summary['run_dir']}`")
    if summary.get("resume_from_checkpoint"):
        overview.append(f"- Resumed from: `{summary['resume_from_checkpoint']}`")
    if summary.get("published"):
        overview.append(f"- Published artifacts: `{json.dumps(summary['published'], sort_keys=True)}`")
    lines.extend(overview)
    lines.append("")
    if summary.get("validation"):
        lines.extend(_section("Validation", _json_block(summary["validation"])))
    if summary.get("parameter_report"):
        lines.extend(_section("Parameter Report", _json_block(summary["parameter_report"])))
    if summary.get("dataset_report"):
        lines.extend(_section("Dataset Report", _json_block(summary["dataset_report"])))
    if summary.get("metrics"):
        lines.extend(_section("Metrics", _json_block(summary["metrics"])))
    if summary.get("artifacts"):
        lines.extend(_section("Artifacts", _json_block(summary["artifacts"])))
    if summary.get("environment"):
        lines.extend(_section("Environment", _json_block(summary["environment"])))
    if summary.get("tracking"):
        lines.extend(_section("Tracking", _json_block(summary["tracking"])))
    if summary.get("notes"):
        lines.extend(_section("Notes", _json_block(summary["notes"])))
    write_markdown(path, "\n".join(lines))


__all__ = ["count_parameters", "dataset_summary", "write_json", "write_markdown", "write_run_summary"]
