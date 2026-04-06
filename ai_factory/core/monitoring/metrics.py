from __future__ import annotations

import math
import statistics
from collections import defaultdict
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


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize_metric_series(
    points: list[MetricPoint],
    metric_name: str,
    *,
    min_points: int = 3,
    anomaly_zscore_threshold: float = 2.5,
    anomaly_relative_threshold: float = 0.2,
) -> dict[str, Any]:
    values = [numeric for p in points if p.name == metric_name and (numeric := _numeric_value(p.value)) is not None]
    if not values:
        return {
            "metric": metric_name,
            "count": 0,
            "status": "insufficient_data",
            "latest": None,
            "mean": None,
            "stdev": None,
            "baseline_mean": None,
            "latest_delta": None,
            "latest_pct_change": None,
            "latest_z_score": None,
            "anomaly_count": 0,
            "anomalies": [],
            "trend": "insufficient_data",
        }

    aggregate = aggregate_metric_points(points, metric_name)
    latest = values[-1]
    baseline_values = values[:-1] if len(values) > 1 else values
    baseline_mean = statistics.mean(baseline_values)
    baseline_stdev = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0.0
    latest_delta = latest - baseline_mean
    latest_pct_change = latest_delta / abs(baseline_mean) if baseline_mean else 0.0
    latest_z_score = (latest_delta / baseline_stdev) if baseline_stdev else None
    anomalies: list[dict[str, Any]] = []

    for index, value in enumerate(values):
        if len(values) < min_points:
            break
        if index == len(values) - 1:
            current_delta = value - baseline_mean
            current_pct_change = current_delta / abs(baseline_mean) if baseline_mean else 0.0
            current_z_score = (current_delta / baseline_stdev) if baseline_stdev else None
            anomaly = False
            reasons: list[str] = []
            if current_z_score is not None and abs(current_z_score) >= anomaly_zscore_threshold:
                anomaly = True
                reasons.append("z_score")
            if abs(current_pct_change) >= anomaly_relative_threshold:
                anomaly = True
                reasons.append("relative_delta")
            if anomaly:
                anomalies.append(
                    {
                        "index": index,
                        "value": value,
                        "delta": round(current_delta, 6),
                        "pct_change": round(current_pct_change, 6),
                        "z_score": round(current_z_score, 6) if current_z_score is not None else None,
                        "reasons": reasons,
                    }
                )
        else:
            previous_mean = statistics.mean(values[:index]) if index > 0 else value
            previous_stdev = statistics.stdev(values[:index]) if index > 1 else 0.0
            if not previous_stdev:
                continue
            delta = value - previous_mean
            z_score = delta / previous_stdev
            if abs(z_score) >= anomaly_zscore_threshold:
                anomalies.append(
                    {
                        "index": index,
                        "value": value,
                        "delta": round(delta, 6),
                        "pct_change": round(delta / abs(previous_mean), 6) if previous_mean else 0.0,
                        "z_score": round(z_score, 6),
                        "reasons": ["z_score"],
                    }
                )

    trend = detect_trend(points, metric_name, min_points=min_points)
    status = "anomalous" if anomalies else ("stable" if trend == "stable" else "observed")
    return {
        **aggregate,
        "metric": metric_name,
        "status": status,
        "baseline_mean": baseline_mean,
        "baseline_stdev": baseline_stdev,
        "latest_delta": latest_delta,
        "latest_pct_change": latest_pct_change,
        "latest_z_score": latest_z_score,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "trend": trend,
    }


def _is_utilization_metric(metric_name: str) -> bool:
    lowered = metric_name.strip().lower()
    unit_suffixes = ("_mb", "_gb", "_bytes", "_b", "_ms", "_s", "_sec", "_secs", "_second", "_seconds")
    if lowered.endswith(unit_suffixes) and not any(token in lowered for token in ("pct", "percent", "%")):
        return False
    return any(
        token in lowered
        for token in ("usage", "utilization", "utilisation", "load", "occupancy", "gpu", "cpu", "memory", "disk")
    )


def _walk_numeric_metrics(prefix: str, value: Any) -> list[tuple[str, float]]:
    walk: list[tuple[str, float]] = []
    numeric = _numeric_value(value)
    if numeric is not None:
        walk.append((prefix, numeric))
        return walk
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            walk.extend(_walk_numeric_metrics(nested_prefix, nested))
        return walk
    if isinstance(value, list):
        for index, nested in enumerate(value):
            nested_prefix = f"{prefix}[{index}]"
            walk.extend(_walk_numeric_metrics(nested_prefix, nested))
    return walk


def build_utilization_rollup(summary: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for key, value in summary.items():
        if key == "utilization_rollup":
            continue
        for metric_name, numeric in _walk_numeric_metrics(key, value):
            if not _is_utilization_metric(metric_name):
                continue
            normalized = numeric * 100.0 if 0.0 <= numeric <= 1.0 else numeric
            candidates.append(
                {
                    "metric": metric_name,
                    "value": numeric,
                    "normalized_pct": round(normalized, 6),
                }
            )

    if not candidates:
        return {
            "metric_count": 0,
            "average_pct": None,
            "peak_pct": None,
            "floor_pct": None,
            "hotspots": [],
        }

    normalized_values = [candidate["normalized_pct"] for candidate in candidates]
    hotspots = sorted(candidates, key=lambda item: item["normalized_pct"], reverse=True)[:5]
    return {
        "metric_count": len(candidates),
        "average_pct": round(statistics.mean(normalized_values), 6),
        "peak_pct": round(max(normalized_values), 6),
        "floor_pct": round(min(normalized_values), 6),
        "hotspots": hotspots,
    }


def build_observability_summary(
    points: list[MetricPoint],
    summary: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    by_metric: dict[str, list[MetricPoint]] = defaultdict(list)
    for point in points:
        by_metric[point.name].append(point)

    metric_summaries = {
        metric_name: summarize_metric_series(metric_points, metric_name, min_points=1 if len(metric_points) == 1 else 3)
        for metric_name, metric_points in by_metric.items()
    }
    anomaly_summaries = [item for item in metric_summaries.values() if item.get("anomaly_count")]
    return {
        "stage": stage,
        "utilization_rollup": build_utilization_rollup(summary),
        "metric_summaries": metric_summaries,
        "anomalies": anomaly_summaries,
        "anomaly_count": sum(item.get("anomaly_count", 0) or 0 for item in metric_summaries.values()),
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
