from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_factory.core.schemas import Alert, MonitoringConfig
from ai_factory.platform.monitoring.manager import MonitoringManager
from ai_factory.platform.monitoring.metrics import MetricsCollector
from ai_factory.platform.monitoring.realtime import IntelligentAlertManager, RealTimeMonitoringSystem


@pytest.fixture
def monitoring_config() -> MonitoringConfig:
    return MonitoringConfig(
        collection_interval_seconds=0.01,
        storage_backend="file",
        alert_channels=["console"],
        thresholds={"cpu_usage": 90, "memory_usage": 85},
    )


@pytest.mark.asyncio
async def test_metrics_collector_historical_series(monitoring_config: MonitoringConfig) -> None:
    collector = MetricsCollector(monitoring_config)
    start = datetime.now(timezone.utc)
    end = start + timedelta(minutes=2)
    points = await collector.get_historical_metrics("instance-1", start, end)

    assert len(points) == 3
    assert points[0].timestamp == start
    assert points[-1].timestamp == start + timedelta(minutes=2)


@pytest.mark.asyncio
async def test_monitoring_manager_health_score(monitoring_config: MonitoringConfig, tmp_path: Path) -> None:
    manager = MonitoringManager(monitoring_config, tmp_path)
    alerts = [
        Alert(
            id="a1",
            severity="critical",
            message="critical",
            source="test",
            timestamp=datetime.now(timezone.utc),
        )
    ]
    metrics = {"system": {"cpu_usage": 95, "memory_usage": 40}}
    score = manager._calculate_health_score(metrics, alerts)
    assert score < 0.6


@pytest.mark.asyncio
async def test_monitoring_manager_stream_metrics_once(monitoring_config: MonitoringConfig, tmp_path: Path) -> None:
    manager = MonitoringManager(monitoring_config, tmp_path)
    manager._running = True
    stream = manager.stream_metrics()
    payload = await anext(stream)
    manager._running = False
    await stream.aclose()

    assert "timestamp" in payload
    assert "metrics" in payload


def test_realtime_monitoring_pipeline_and_alerts() -> None:
    alert_manager = IntelligentAlertManager()
    assert alert_manager.evaluate_metrics({"cpu_usage_pct": 91.0}) == ["High CPU Usage Detected"]

    system = RealTimeMonitoringSystem()
    captured: dict[str, object] = {}
    system.streamer.broadcast_update = lambda data: captured.update({"payload": data})

    system.start()
    system.process_tick()
    system.stop()

    assert "payload" in captured
    status = system.get_status()
    assert status["is_running"] is False
