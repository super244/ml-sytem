from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_factory.core.schemas import Alert, MonitoringConfig
from ai_factory.platform.monitoring.hardware import get_cluster_nodes, get_system_ram_gb
from ai_factory.platform.monitoring.manager import MonitoringManager
from ai_factory.platform.monitoring.metrics import MetricsCollector


def test_get_system_ram_gb_has_reasonable_fallback() -> None:
    value = get_system_ram_gb()
    assert value >= 1


def test_get_cluster_nodes_returns_local_and_remote() -> None:
    nodes = get_cluster_nodes()
    assert len(nodes) >= 2
    assert any(node["id"] == "local-primary" for node in nodes)
    assert all("status" in node for node in nodes)


@pytest.mark.asyncio
async def test_monitoring_manager_system_health_responds_to_alerts(tmp_path: Path) -> None:
    config = MonitoringConfig(
        collection_interval_seconds=5.0,
        thresholds={"cpu_usage": 80, "memory_usage": 80},
    )
    manager = MonitoringManager(config, tmp_path)

    async def _fake_metrics():
        return {
            "system": {"cpu_usage": 95, "memory_usage": 92},
            "utilization_rollup": {"peak_pct": 95.0, "average_pct": 93.5},
            "observability": {
                "anomalies": [{"metric": "cpu_usage", "value": 95.0}],
                "utilization_rollup": {"peak_pct": 95.0},
            },
        }

    async def _fake_alerts():
        return [
            Alert(
                id="critical-1",
                severity="critical",
                message="high pressure",
                source="test",
                timestamp=datetime.now(timezone.utc),
            )
        ]

    manager.metrics_collector.collect_all_metrics = _fake_metrics  # type: ignore[assignment]
    manager.alert_manager.get_active_alerts = _fake_alerts  # type: ignore[assignment]

    health = await manager.get_system_health()
    assert health["status"] == "unhealthy"
    assert health["critical_alerts"] == 1
    assert health["utilization_rollup"]["peak_pct"] == 95.0
    assert health["anomaly_summary"]


@pytest.mark.asyncio
async def test_metrics_collector_historical_points_use_time_delta() -> None:
    collector = MetricsCollector(MonitoringConfig())
    start = datetime(2026, 1, 1, 0, 0, 30)
    end = start + timedelta(minutes=2)

    points = await collector.get_historical_metrics("instance-1", start, end)
    assert len(points) == 3
    assert points[1].timestamp - points[0].timestamp == timedelta(minutes=1)
