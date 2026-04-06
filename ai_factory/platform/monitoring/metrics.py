"""Metrics collection for AI-Factory monitoring."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_factory.core.monitoring.metrics import build_utilization_rollup
from ai_factory.core.schemas import MetricPoint, MonitoringConfig

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MetricsCollector:
    """Collects and stores metrics from AI-Factory components."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._metrics_store: dict[str, list[MetricPoint]] = {}

    async def collect_all_metrics(self) -> dict[str, Any]:
        """Collect metrics from all sources."""
        system_metrics = await self.get_system_metrics()
        training_metrics = await self.get_training_metrics()
        inference_metrics = await self.get_inference_metrics()
        utilization_rollup = build_utilization_rollup(
            {
                "system": system_metrics,
                "training": training_metrics,
                "inference": inference_metrics,
            }
        )

        return {
            "timestamp": _utc_now().isoformat(),
            "system": system_metrics,
            "training": training_metrics,
            "inference": inference_metrics,
            "utilization_rollup": utilization_rollup,
            "observability": {
                "utilization_rollup": utilization_rollup,
                "anomalies": [],
                "anomaly_count": 0,
            },
        }

    async def get_current_metrics(self, instance_id: str | None = None) -> dict[str, Any]:
        """Get current metrics for specific instance or all instances."""
        if instance_id:
            return await self.get_instance_metrics(instance_id)
        return await self.collect_all_metrics()

    async def get_system_metrics(self) -> dict[str, Any]:
        """Collect system-level metrics."""
        return {
            "cpu_usage": 0.65,
            "memory_usage": 0.78,
            "disk_usage": 0.45,
            "network_io": 1024000,
            "gpu_count": 4,
            "gpu_usage": [0.89, 0.76, 0.92, 0.68],
        }

    async def get_training_metrics(self) -> dict[str, Any]:
        """Collect training-related metrics."""
        return {
            "active_jobs": 3,
            "completed_jobs": 127,
            "failed_jobs": 2,
            "average_loss": 0.234,
            "average_accuracy": 0.876,
        }

    async def get_inference_metrics(self) -> dict[str, Any]:
        """Collect inference-related metrics."""
        return {"active_models": 5, "requests_per_second": 45.6, "average_latency_ms": 234, "success_rate": 0.987}

    async def get_instance_metrics(self, instance_id: str) -> dict[str, Any]:
        """Get metrics for a specific instance."""
        return {
            "instance_id": instance_id,
            "status": "running",
            "progress": 0.65,
            "current_loss": 0.234,
            "learning_rate": 0.0001,
            "gpu_usage": 0.89,
            "memory_usage": 0.78,
        }

    async def get_historical_metrics(
        self, instance_id: str, start_time: datetime, end_time: datetime
    ) -> list[MetricPoint]:
        """Get historical metrics for an instance."""
        # Mock implementation - would query time-series database
        metrics = []
        current_time = start_time
        while current_time <= end_time:
            metrics.append(
                MetricPoint(
                    timestamp=current_time,
                    name="loss",
                    value=0.5 - (current_time - start_time).total_seconds() / 3600 * 0.1,
                    labels={"instance_id": instance_id},
                )
            )
            current_time = current_time + timedelta(minutes=1)

        return metrics

    async def store_metric(self, metric: MetricPoint) -> None:
        """Store a metric point."""
        if metric.name not in self._metrics_store:
            self._metrics_store[metric.name] = []
        self._metrics_store[metric.name].append(metric)

    async def query_metrics(self, query: str) -> list[MetricPoint]:
        """Query metrics with a simple query language."""
        # Simplified implementation - would use real query language
        name = query.split("=")[1].strip('"') if "=" in query else "loss"
        metrics = self._metrics_store.get(name, [])
        return metrics if isinstance(metrics, list) else []
