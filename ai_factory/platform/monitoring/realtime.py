"""Real-time monitoring system implementation."""

import logging
from typing import Any

from ai_factory.core.monitoring.metrics import build_utilization_rollup

logger = logging.getLogger(__name__)


class HighFrequencyMetricsCollector:
    """Collects high-frequency metrics from various system components."""

    def __init__(self, sample_rate_hz: int = 100) -> None:
        self.sample_rate_hz = sample_rate_hz
        self._is_collecting = False

    def start_collection(self) -> None:
        """Starts the metrics collection process."""
        self._is_collecting = True
        logger.info(f"Started high-frequency metrics collection at {self.sample_rate_hz}Hz.")

    def stop_collection(self) -> None:
        """Stops the metrics collection process."""
        self._is_collecting = False
        logger.info("Stopped high-frequency metrics collection.")

    def get_latest_metrics(self) -> dict[str, Any]:
        """Retrieves the most recently collected metrics."""
        return {"cpu_usage_pct": 0.0, "memory_usage_mb": 0.0, "gpu_utilization_pct": 0.0}


class IntelligentAlertManager:
    """Analyzes metrics and triggers alerts based on dynamic thresholds."""

    def __init__(self, sensitivity: str = "medium") -> None:
        self.sensitivity = sensitivity

    def evaluate_metrics(self, metrics: dict[str, Any]) -> list[str]:
        """Evaluates metrics against thresholds and returns triggered alerts."""
        alerts: list[str] = []
        if metrics.get("cpu_usage_pct", 0) > 90.0:
            alerts.append("High CPU Usage Detected")
        if metrics.get("gpu_utilization_pct", 0) > 90.0:
            alerts.append("High GPU Utilization Detected")
        if metrics.get("anomalies"):
            alerts.append(f"{len(metrics['anomalies'])} anomaly signal(s) detected")
        return alerts

    def build_alert_payloads(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Build structured alert payloads for downstream dashboards."""
        rollup = metrics.get("utilization_rollup") or build_utilization_rollup(metrics)
        payloads: list[dict[str, Any]] = []
        for message in self.evaluate_metrics(metrics):
            severity = "critical" if "anomaly" in message.lower() else "warning"
            payloads.append(
                {
                    "message": message,
                    "severity": severity,
                    "utilization_rollup": rollup,
                    "observed": {
                        "cpu_usage_pct": metrics.get("cpu_usage_pct"),
                        "gpu_utilization_pct": metrics.get("gpu_utilization_pct"),
                        "memory_usage_mb": metrics.get("memory_usage_mb"),
                    },
                }
            )
        return payloads

    def dispatch_alert(self, alert_msg: str) -> None:
        """Dispatches an alert to configured notification channels."""
        logger.warning("ALERT DISPATCHED: %s", alert_msg)


class WebSocketDashboardStreamer:
    """Streams real-time metrics and alerts to connected dashboard clients."""

    def __init__(self, endpoint: str = "/ws/monitoring") -> None:
        self.endpoint = endpoint
        self._connected_clients = 0

    def broadcast_update(self, data: dict[str, Any]) -> None:
        """Broadcasts a data payload to all connected WebSocket clients."""
        logger.info(f"Broadcasting update to {self._connected_clients} clients")
        # Log data details only at debug level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Broadcast data: {data}")

    def get_client_count(self) -> int:
        """Returns the number of currently connected clients."""
        return self._connected_clients


class PerformanceAnalyzer:
    """Analyzes historical and real-time trends to predict performance degradation."""

    def __init__(self, history_window_minutes: int = 60) -> None:
        self.history_window_minutes = history_window_minutes

    def analyze_trends(self, current_metrics: dict[str, Any]) -> dict[str, Any]:
        """Analyzes trends and returns a report on system performance health."""
        utilization_rollup = build_utilization_rollup(current_metrics)
        anomalies: list[dict[str, Any]] = []
        if current_metrics.get("cpu_usage_pct", 0) > 90.0:
            anomalies.append(
                {"metric": "cpu_usage_pct", "value": current_metrics.get("cpu_usage_pct"), "severity": "high"}
            )
        if current_metrics.get("gpu_utilization_pct", 0) > 90.0:
            anomalies.append(
                {
                    "metric": "gpu_utilization_pct",
                    "value": current_metrics.get("gpu_utilization_pct"),
                    "severity": "high",
                }
            )
        status = "degraded" if anomalies else "healthy"
        trend = "volatile" if anomalies else "stable"
        return {
            "status": status,
            "trend": trend,
            "utilization_rollup": utilization_rollup,
            "anomalies": anomalies,
        }


class RealTimeMonitoringSystem:
    """
    Central orchestration class for real-time monitoring.

    Composes collection, analysis, alerting, and streaming components into
    a unified real-time observation pipeline.
    """

    def __init__(self) -> None:
        self.collector = HighFrequencyMetricsCollector()
        self.alert_manager = IntelligentAlertManager()
        self.streamer = WebSocketDashboardStreamer()
        self.analyzer = PerformanceAnalyzer()
        self._is_running = False

    def start(self) -> None:
        """Starts the real-time monitoring system and all its subcomponents."""
        logger.info("Starting RealTimeMonitoringSystem...")
        self.collector.start_collection()
        self._is_running = True
        logger.info("RealTimeMonitoringSystem started successfully.")

    def stop(self) -> None:
        """Stops the real-time monitoring system."""
        logger.info("Stopping RealTimeMonitoringSystem...")
        self.collector.stop_collection()
        self._is_running = False
        logger.info("RealTimeMonitoringSystem stopped.")

    def process_tick(self) -> None:
        """Executes a single processing tick of the monitoring pipeline."""
        if not self._is_running:
            return

        # 1. Collect
        metrics = self.collector.get_latest_metrics()

        # 2. Analyze
        analysis = self.analyzer.analyze_trends(metrics)

        # 3. Alert
        alerts = self.alert_manager.evaluate_metrics(metrics)
        alert_payloads = self.alert_manager.build_alert_payloads(metrics)
        for alert in alerts:
            self.alert_manager.dispatch_alert(alert)

        # 4. Stream
        payload = {
            "metrics": metrics,
            "analysis": analysis,
            "alerts": alerts,
            "alert_payloads": alert_payloads,
        }
        self.streamer.broadcast_update(payload)

    def get_status(self) -> dict[str, Any]:
        """Returns the current operational status of the monitoring system."""
        return {
            "is_running": self._is_running,
            "components": {
                "collector_active": self.collector._is_collecting,
                "connected_clients": self.streamer.get_client_count(),
            },
        }
