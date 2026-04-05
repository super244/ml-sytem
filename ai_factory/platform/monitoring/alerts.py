"""Alert management for AI-Factory monitoring."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_factory.core.schemas import Alert, MonitoringConfig

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AlertManager:
    """Manages alerts and notifications for AI-Factory."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._active_alerts: list[Alert] = []
        self._alert_history: list[Alert] = []

    def _build_context(self, metrics: dict[str, Any], *, source: str, metric: str, bound: str) -> dict[str, Any]:
        observability = metrics.get("observability") or {}
        system_metrics = metrics.get("system") or {}
        training_metrics = metrics.get("training") or {}
        inference_metrics = metrics.get("inference") or {}
        return {
            "source": source,
            "metric": metric,
            "bound": bound,
            "utilization_rollup": metrics.get("utilization_rollup") or observability.get("utilization_rollup"),
            "anomalies": observability.get("anomalies", []),
            "system": {k: v for k, v in system_metrics.items() if isinstance(v, (int, float, bool))},
            "training": {k: v for k, v in training_metrics.items() if isinstance(v, (int, float, bool))},
            "inference": {k: v for k, v in inference_metrics.items() if isinstance(v, (int, float, bool))},
        }

    def _build_alert(
        self,
        *,
        alert_id: str,
        severity: str,
        message: str,
        source: str,
        metric: str,
        observed_value: float,
        threshold: float,
        bound: str,
        metrics: dict[str, Any],
    ) -> Alert:
        context = self._build_context(metrics, source=source, metric=metric, bound=bound)
        context.update(
            {
                "observed_value": observed_value,
                "threshold": threshold,
                "delta": round(observed_value - threshold, 6)
                if bound == "min"
                else round(threshold - observed_value, 6),
            }
        )
        return Alert(
            id=alert_id,
            severity=severity,
            message=message,
            source=source,
            timestamp=_utc_now(),
            metadata=context,
        )

    async def check_alerts(self, metrics: dict[str, Any]) -> list[Alert]:
        """Check metrics against thresholds and generate alerts."""
        alerts: list[Alert] = []

        # Check system metrics
        system_metrics = metrics.get("system", {})
        cpu_usage = float(system_metrics.get("cpu_usage", 0) or 0)
        cpu_threshold = float(self.config.thresholds.get("cpu_usage", 90))
        if cpu_usage > cpu_threshold:
            alerts.append(
                self._build_alert(
                    alert_id=f"cpu_high_{int(_utc_now().timestamp())}",
                    severity="warning",
                    message=f"High CPU usage: {cpu_usage:.1f}%",
                    source="system_monitor",
                    metric="cpu_usage",
                    observed_value=cpu_usage,
                    threshold=cpu_threshold,
                    bound="max",
                    metrics=metrics,
                )
            )

        memory_usage = float(system_metrics.get("memory_usage", 0) or 0)
        memory_threshold = float(self.config.thresholds.get("memory_usage", 85))
        if memory_usage > memory_threshold:
            alerts.append(
                self._build_alert(
                    alert_id=f"memory_high_{int(_utc_now().timestamp())}",
                    severity="warning",
                    message=f"High memory usage: {memory_usage:.1f}%",
                    source="system_monitor",
                    metric="memory_usage",
                    observed_value=memory_usage,
                    threshold=memory_threshold,
                    bound="max",
                    metrics=metrics,
                )
            )

        # Check training metrics
        training_metrics = metrics.get("training", {})
        failed_jobs = int(training_metrics.get("failed_jobs", 0) or 0)
        if failed_jobs > 0:
            alerts.append(
                self._build_alert(
                    alert_id=f"training_failures_{int(_utc_now().timestamp())}",
                    severity="critical",
                    message=f"Training job failures: {failed_jobs}",
                    source="training_monitor",
                    metric="failed_jobs",
                    observed_value=float(failed_jobs),
                    threshold=0.0,
                    bound="min",
                    metrics=metrics,
                )
            )

        # Store alerts
        for alert in alerts:
            await self.store_alert(alert)

        return alerts

    async def store_alert(self, alert: Alert) -> None:
        """Store an alert in the alert history."""
        self._alert_history.append(alert)
        if alert.severity in ["critical", "warning"]:
            self._active_alerts.append(alert)

    async def send_alert(self, alert: Alert) -> None:
        """Send alert notification through configured channels."""
        payload = {
            "id": alert.id,
            "severity": alert.severity,
            "message": alert.message,
            "source": alert.source,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata,
        }
        logger.warning("ALERT: %s", payload)

        # Send to configured channels
        for channel in self.config.alert_channels:
            if channel == "console":
                print(json.dumps(payload, sort_keys=True))
            elif channel == "file":
                await self._write_alert_to_file(payload)
            # Add more channels as needed

    async def _write_alert_to_file(self, payload: dict[str, Any]) -> None:
        """Write alert to log file."""
        log_file = Path("alerts.log")
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    async def get_active_alerts(self) -> list[Alert]:
        """Get currently active alerts."""
        return self._active_alerts.copy()

    async def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Get alert history."""
        return self._alert_history[-limit:]

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert to remove it from active alerts."""
        for i, alert in enumerate(self._active_alerts):
            if alert.id == alert_id:
                self._active_alerts.pop(i)
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False

    async def clear_alerts(self) -> int:
        """Clear all active alerts."""
        count = len(self._active_alerts)
        self._active_alerts.clear()
        logger.info(f"Cleared {count} active alerts")
        return count
