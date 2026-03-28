"""Alert management for AI-Factory monitoring."""

from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime

from ai_factory.core.schemas import Alert, MonitoringConfig


logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alerts and notifications for AI-Factory."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._active_alerts = []
        self._alert_history = []
    
    async def check_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        # Check system metrics
        system_metrics = metrics.get("system", {})
        if system_metrics.get("cpu_usage", 0) > self.config.thresholds.get("cpu_usage", 90):
            alerts.append(Alert(
                id=f"cpu_high_{int(datetime.utcnow().timestamp())}",
                severity="warning",
                message=f"High CPU usage: {system_metrics.get('cpu_usage', 0):.1f}%",
                source="system_monitor",
                timestamp=datetime.utcnow()
            ))
        
        if system_metrics.get("memory_usage", 0) > self.config.thresholds.get("memory_usage", 85):
            alerts.append(Alert(
                id=f"memory_high_{int(datetime.utcnow().timestamp())}",
                severity="warning", 
                message=f"High memory usage: {system_metrics.get('memory_usage', 0):.1f}%",
                source="system_monitor",
                timestamp=datetime.utcnow()
            ))
        
        # Check training metrics
        training_metrics = metrics.get("training", {})
        if training_metrics.get("failed_jobs", 0) > 0:
            alerts.append(Alert(
                id=f"training_failures_{int(datetime.utcnow().timestamp())}",
                severity="critical",
                message=f"Training job failures: {training_metrics.get('failed_jobs', 0)}",
                source="training_monitor",
                timestamp=datetime.utcnow()
            ))
        
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
        logger.warning(f"ALERT: {alert.message}")
        
        # Send to configured channels
        for channel in self.config.alert_channels:
            if channel == "console":
                print(f"[{alert.severity.upper()}] {alert.message}")
            elif channel == "file":
                await self._write_alert_to_file(alert)
            # Add more channels as needed
    
    async def _write_alert_to_file(self, alert: Alert) -> None:
        """Write alert to log file."""
        log_file = Path("alerts.log")
        with open(log_file, "a") as f:
            f.write(f"{alert.timestamp.isoformat()} [{alert.severity}] {alert.message}\n")
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return self._active_alerts.copy()
    
    async def get_alert_history(self, limit: int = 100) -> List[Alert]:
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
