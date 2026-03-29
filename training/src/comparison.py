from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_factory.core.io import load_json


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path, default={})


def _metric_lower_is_better(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("loss", "error", "perplexity", "latency", "time", "duration"))


def _best_metric_key(metrics: dict[str, Any]) -> str | None:
    preferred = ("eval_loss", "loss", "eval_accuracy", "accuracy", "eval_f1", "f1")
    for key in preferred:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key
    numeric_keys = sorted(key for key, value in metrics.items() if isinstance(value, (int, float)))
    return numeric_keys[0] if numeric_keys else None


def _json_block(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"


def _pct_change(old: float, new: float) -> float | None:
    if old == 0:
        return None
    return round(((new - old) / abs(old)) * 100, 2)


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    base = Path(run_dir)
    manifest = _load_json_if_exists(base / "manifests" / "run_manifest.json")
    metrics = _load_json_if_exists(base / "metrics" / "metrics.json")
    tracking_summary = _load_json_if_exists(base / "metrics" / "tracking_summary.json")
    dataset_report = _load_json_if_exists(base / "metrics" / "dataset_report.json")
    model_report = _load_json_if_exists(base / "metrics" / "model_report.json")
    environment_snapshot = _load_json_if_exists(base / "manifests" / "environment_snapshot.json")
    config_snapshot = _load_json_if_exists(base / "manifests" / "config_snapshot.json")
    checkpoint_report = _load_json_if_exists(base / "manifests" / "validation_report.json")
    return {
        "run_dir": str(base),
        "run_id": manifest.get("run_id"),
        "run_name": manifest.get("run_name"),
        "profile_name": manifest.get("profile_name"),
        "base_model": manifest.get("base_model"),
        "model_name": manifest.get("model_name"),
        "created_at": manifest.get("created_at"),
        "status": tracking_summary.get("status"),
        "metrics": metrics,
        "dataset_report": dataset_report,
        "model_report": model_report,
        "tracking_summary": tracking_summary,
        "environment_snapshot": environment_snapshot,
        "config_snapshot": config_snapshot,
        "checkpoint_report": checkpoint_report,
        "metadata": manifest.get("metadata", {}),
        "manifest": manifest,
    }


def compare_runs(
    left_run_dir: str | Path,
    right_run_dir: str | Path,
    *,
    primary_metric: str | None = None,
) -> dict[str, Any]:
    left = load_run_summary(left_run_dir)
    right = load_run_summary(right_run_dir)
    keys = sorted(set(left["metrics"]) | set(right["metrics"]))
    delta: dict[str, Any] = {}
    for key in keys:
        left_value = left["metrics"].get(key, 0.0)
        right_value = right["metrics"].get(key, 0.0)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            abs_delta = round((right_value or 0.0) - (left_value or 0.0), 12)
            delta[key] = {
                "left": left_value,
                "right": right_value,
                "delta": abs_delta,
                "pct_change": _pct_change(left_value, right_value),
                "direction": "improved" if (
                    (_metric_lower_is_better(key) and abs_delta < 0)
                    or (not _metric_lower_is_better(key) and abs_delta > 0)
                ) else ("regressed" if abs_delta != 0 else "unchanged"),
            }
    metric_name = primary_metric or _best_metric_key({**left["metrics"], **right["metrics"]})
    winner = None
    if metric_name and metric_name in delta:
        lower_is_better = _metric_lower_is_better(metric_name)
        change = delta[metric_name]["delta"]
        if change == 0:
            winner = "tie"
        elif lower_is_better:
            winner = "right" if change < 0 else "left"
        else:
            winner = "right" if change > 0 else "left"

    config_diff = _compare_configs(
        left.get("config_snapshot", {}),
        right.get("config_snapshot", {}),
    )

    recommendations = _build_comparison_recommendations(delta, metric_name, winner, left, right)

    return {
        "left": left,
        "right": right,
        "delta": delta,
        "primary_metric": metric_name,
        "winner": winner,
        "metric_orientation": (
            "lower_is_better" if metric_name and _metric_lower_is_better(metric_name) else "higher_is_better"
        ),
        "shared_metrics": [key for key in keys if key in left["metrics"] and key in right["metrics"]],
        "config_diff": config_diff,
        "recommendations": recommendations,
    }


def _compare_configs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    all_keys = sorted(set(list(left.keys()) + list(right.keys())))
    for key in all_keys:
        lv = left.get(key)
        rv = right.get(key)
        if isinstance(lv, dict) and isinstance(rv, dict):
            sub = _compare_configs(lv, rv)
            if sub:
                diff[key] = sub
        elif lv != rv:
            diff[key] = {"left": lv, "right": rv}
    return diff


def _build_comparison_recommendations(
    delta: dict[str, Any],
    primary_metric: str | None,
    winner: str | None,
    left: dict[str, Any],
    right: dict[str, Any],
) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    if not primary_metric or not winner:
        recs.append({
            "action": "add_metrics",
            "reason": "No comparable metrics found. Ensure both runs produce evaluation metrics.",
        })
        return recs
    if winner == "tie":
        recs.append({
            "action": "differentiate",
            "reason": f"Both runs tied on {primary_metric}. Try adjusting learning rate, data mix, or adapter rank.",
        })
    elif winner == "right":
        recs.append({
            "action": "iterate_on_right",
            "reason": f"Right run is better on {primary_metric}. Consider continuing from its checkpoint.",
        })
    else:
        recs.append({
            "action": "iterate_on_left",
            "reason": f"Left run is better on {primary_metric}. Consider continuing from its checkpoint.",
        })

    loss_info = delta.get("eval_loss") or delta.get("loss")
    if loss_info and isinstance(loss_info, dict):
        if loss_info.get("direction") == "regressed":
            recs.append({
                "action": "check_overfitting",
                "reason": "Loss regressed between runs. Check for overfitting or data distribution shift.",
            })

    return recs


def format_comparison_report(report: dict[str, Any]) -> str:
    lines = [
        "# Run Comparison",
        "",
        f"- Left: `{report['left'].get('run_name')}` (`{report['left'].get('run_dir')}`)",
        f"- Right: `{report['right'].get('run_name')}` (`{report['right'].get('run_dir')}`)",
        f"- Primary metric: `{report.get('primary_metric') or 'auto'}`",
        f"- Winner: `{report.get('winner') or 'undetermined'}`",
        "",
        "## Metric Deltas",
        "",
    ]
    delta = report.get("delta", {})
    if delta:
        lines.append("| Metric | Left | Right | Delta | Change | Direction |")
        lines.append("|--------|------|-------|-------|--------|-----------|")
        for key, info in sorted(delta.items()):
            if isinstance(info, dict):
                pct = f"{info.get('pct_change', 'n/a')}%" if info.get("pct_change") is not None else "n/a"
                lines.append(
                    f"| {key} | {info.get('left', 'n/a')} | {info.get('right', 'n/a')} "
                    f"| {info.get('delta', 'n/a')} | {pct} | {info.get('direction', '')} |"
                )
        lines.append("")

    if report.get("config_diff"):
        lines.extend([
            "## Config Differences",
            "",
            _json_block(report["config_diff"]),
            "",
        ])

    if report.get("recommendations"):
        lines.extend(["## Recommendations", ""])
        for rec in report["recommendations"]:
            lines.append(f"- **{rec['action']}**: {rec['reason']}")
        lines.append("")

    if report["left"].get("dataset_report") or report["right"].get("dataset_report"):
        lines.extend(
            [
                "## Dataset Snapshots",
                "",
                _json_block(
                    {
                        "left": report["left"].get("dataset_report", {}),
                        "right": report["right"].get("dataset_report", {}),
                    }
                ),
                "",
            ]
        )
    return "\n".join(lines)
