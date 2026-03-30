from __future__ import annotations

import statistics
from typing import Any, Literal

from ai_factory.core.instances.models import MetricPoint


def metric_points_from_summary(summary: dict[str, Any], *, stage: str) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for key, value in summary.items():
        if isinstance(value, (int, float, bool)):
            points.append(MetricPoint(name=key, value=value, tags={"stage": stage}))
    return points


TrendDirection = Literal["improving", "degrading", "stable", "insufficient_data"]

HIGHER_IS_BETTER = frozenset(
    {
        "accuracy",
        "parse_rate",
        "verifier_agreement_rate",
        "avg_tokens_per_second",
        "trainable_ratio",
    }
)

LOWER_IS_BETTER = frozenset(
    {
        "eval_loss",
        "train_loss",
        "no_answer_rate",
        "avg_latency_s",
        "avg_time_to_first_token_s",
    }
)


def metric_direction(name: str) -> str:
    if name in HIGHER_IS_BETTER:
        return "higher_is_better"
    if name in LOWER_IS_BETTER:
        return "lower_is_better"
    return "unknown"


def detect_trend(
    points: list[MetricPoint],
    metric_name: str,
    *,
    min_points: int = 3,
    stability_threshold: float = 0.01,
) -> TrendDirection:
    values = [float(p.value) for p in points if p.name == metric_name and isinstance(p.value, (int, float))]
    if len(values) < min_points:
        return "insufficient_data"
    recent_half = values[len(values) // 2 :]
    early_half = values[: len(values) // 2]
    if not early_half or not recent_half:
        return "insufficient_data"
    early_mean = statistics.mean(early_half)
    recent_mean = statistics.mean(recent_half)
    if early_mean == 0:
        delta_pct = 0.0
    else:
        delta_pct = (recent_mean - early_mean) / abs(early_mean)
    if abs(delta_pct) < stability_threshold:
        return "stable"
    direction = metric_direction(metric_name)
    if direction == "higher_is_better":
        return "improving" if delta_pct > 0 else "degrading"
    if direction == "lower_is_better":
        return "improving" if delta_pct < 0 else "degrading"
    return "improving" if delta_pct > 0 else "degrading"


def aggregate_metric_points(
    points: list[MetricPoint],
    metric_name: str,
) -> dict[str, Any]:
    values = [float(p.value) for p in points if p.name == metric_name and isinstance(p.value, (int, float))]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "latest": None, "stdev": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "latest": values[-1],
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def check_metric_health(
    summary: dict[str, Any],
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    threshold_map = {
        "accuracy": ("min", thresholds.get("min_accuracy")),
        "parse_rate": ("min", thresholds.get("min_parse_rate")),
        "verifier_agreement_rate": ("min", thresholds.get("min_verifier_agreement")),
        "no_answer_rate": ("max", thresholds.get("max_no_answer_rate")),
        "avg_latency_s": ("max", thresholds.get("max_latency_s")),
    }
    for metric_name, (bound_type, threshold) in threshold_map.items():
        value = summary.get(metric_name)
        if not isinstance(value, (int, float)) or threshold is None:
            continue
        if bound_type == "min":
            ok = float(value) >= float(threshold)
            gap = float(value) - float(threshold)
        else:
            ok = float(value) <= float(threshold)
            gap = float(threshold) - float(value)
        checks.append(
            {
                "metric": metric_name,
                "value": float(value),
                "threshold": float(threshold),
                "bound": bound_type,
                "ok": ok,
                "gap": round(gap, 6),
            }
        )
    return checks


def compare_metric_summaries(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> list[dict[str, Any]]:
    all_keys = sorted(set(left.keys()) | set(right.keys()))
    results: list[dict[str, Any]] = []
    for key in all_keys:
        lv = left.get(key)
        rv = right.get(key)
        if not isinstance(lv, (int, float)) or not isinstance(rv, (int, float)):
            continue
        delta = float(rv) - float(lv)
        pct = (delta / abs(float(lv))) if float(lv) != 0 else 0.0
        direction = metric_direction(key)
        if direction == "higher_is_better":
            assessment = "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged")
        elif direction == "lower_is_better":
            assessment = "improved" if delta < 0 else ("regressed" if delta > 0 else "unchanged")
        else:
            assessment = "changed" if delta != 0 else "unchanged"
        results.append(
            {
                "metric": key,
                left_label: float(lv),
                right_label: float(rv),
                "delta": round(delta, 6),
                "pct_change": round(pct, 6),
                "direction": direction,
                "assessment": assessment,
            }
        )
    return results
