"""Test platform monitoring alerts."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_factory.core.schemas import Alert, MonitoringConfig
from ai_factory.platform.monitoring.alerts import AlertManager


@pytest.fixture
def monitoring_config():
    """Create a test monitoring configuration."""
    return MonitoringConfig(
        collection_interval_seconds=5.0,
        storage_backend="file",
        alert_channels=["console"],
        thresholds={
            "cpu_usage": 90,
            "memory_usage": 85,
        },
    )


@pytest.fixture
def alert_manager(monitoring_config):
    """Create an AlertManager instance for testing."""
    return AlertManager(monitoring_config)


@pytest.mark.asyncio
async def test_alert_manager_initialization(alert_manager):
    """Test AlertManager initialization."""
    assert alert_manager.config is not None
    assert alert_manager._active_alerts == []
    assert alert_manager._alert_history == []


@pytest.mark.asyncio
async def test_check_alerts_high_cpu(alert_manager):
    """Test alert generation for high CPU usage."""
    metrics = {
        "system": {"cpu_usage": 95.0, "memory_usage": 50.0},
        "training": {},
    }

    alerts = await alert_manager.check_alerts(metrics)

    assert len(alerts) == 1
    assert alerts[0].severity == "warning"
    assert "CPU" in alerts[0].message
    assert alerts[0].metadata["metric"] == "cpu_usage"
    assert alerts[0].metadata["observed_value"] == 95.0
    assert alerts[0].metadata["threshold"] == 90.0
    assert "utilization_rollup" in alerts[0].metadata


@pytest.mark.asyncio
async def test_check_alerts_high_memory(alert_manager):
    """Test alert generation for high memory usage."""
    metrics = {
        "system": {"cpu_usage": 50.0, "memory_usage": 90.0},
        "training": {},
    }

    alerts = await alert_manager.check_alerts(metrics)

    assert len(alerts) == 1
    assert alerts[0].severity == "warning"
    assert "memory" in alerts[0].message.lower()
    assert alerts[0].metadata["metric"] == "memory_usage"


@pytest.mark.asyncio
async def test_check_alerts_training_failures(alert_manager):
    """Test alert generation for training failures."""
    metrics = {
        "system": {"cpu_usage": 50.0, "memory_usage": 50.0},
        "training": {"failed_jobs": 3},
    }

    alerts = await alert_manager.check_alerts(metrics)

    assert len(alerts) == 1
    assert alerts[0].severity == "critical"
    assert "failures" in alerts[0].message.lower()
    assert alerts[0].metadata["metric"] == "failed_jobs"
    assert alerts[0].metadata["observed_value"] == 3.0


@pytest.mark.asyncio
async def test_check_alerts_no_alerts(alert_manager):
    """Test no alerts when metrics are within thresholds."""
    metrics = {
        "system": {"cpu_usage": 50.0, "memory_usage": 50.0},
        "training": {},
    }

    alerts = await alert_manager.check_alerts(metrics)

    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_store_alert(alert_manager):
    """Test storing alerts."""
    alert = Alert(
        id="test_alert_1",
        severity="warning",
        message="Test alert",
        source="test",
        timestamp=datetime.now(UTC),
    )

    await alert_manager.store_alert(alert)

    assert alert in alert_manager._alert_history
    assert alert in alert_manager._active_alerts


@pytest.mark.asyncio
async def test_get_active_alerts(alert_manager):
    """Test retrieving active alerts."""
    alert = Alert(
        id="test_alert_1",
        severity="warning",
        message="Test alert",
        source="test",
        timestamp=datetime.now(UTC),
    )

    await alert_manager.store_alert(alert)
    active = await alert_manager.get_active_alerts()

    assert len(active) == 1
    assert active[0].id == "test_alert_1"


@pytest.mark.asyncio
async def test_get_alert_history(alert_manager):
    """Test retrieving alert history."""
    alert = Alert(
        id="test_alert_1",
        severity="info",
        message="Test alert",
        source="test",
        timestamp=datetime.now(UTC),
    )

    await alert_manager.store_alert(alert)
    history = await alert_manager.get_alert_history()

    assert len(history) == 1
    assert history[0].id == "test_alert_1"


@pytest.mark.asyncio
async def test_send_alert_writes_structured_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = MonitoringConfig(alert_channels=["file"])
    manager = AlertManager(config)
    monkeypatch.chdir(tmp_path)
    alert = Alert(
        id="structured-alert",
        severity="warning",
        message="High CPU usage: 95.0%",
        source="system_monitor",
        timestamp=datetime.now(UTC),
        metadata={"metric": "cpu_usage", "observed_value": 95.0, "threshold": 90.0},
    )

    await manager.send_alert(alert)

    payload = json.loads((tmp_path / "alerts.log").read_text().strip())
    assert payload["id"] == "structured-alert"
    assert payload["metadata"]["metric"] == "cpu_usage"
