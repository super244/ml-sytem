from __future__ import annotations

from typing import Any

from ai_factory.core.instances.models import MetricPoint


def metric_points_from_summary(summary: dict[str, Any], *, stage: str) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for key, value in summary.items():
        if isinstance(value, (int, float, bool)):
            points.append(MetricPoint(name=key, value=value, tags={"stage": stage}))
    return points
