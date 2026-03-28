"""Monitoring manager for real-time metrics and alerting."""

from typing import Dict, List, Any, Optional, AsyncIterator
from pathlib import Path
import asyncio
import json
import logging
from datetime import datetime

from ai_factory.core.schemas import MonitoringConfig, Alert, MetricPoint
from .metrics import MetricsCollector
from .alerts import AlertManager


logger = logging.getLogger(__name__)


class MonitoringManager:
    """Manages real-time monitoring of AI-Factory operations."""
    
    def __init__(self, config: MonitoringConfig, repo_root: Path):
        self.config = config
        self.repo_root = repo_root
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> None:
        """Start the monitoring service."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring service started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring service."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics from all sources
                metrics = await self.metrics_collector.collect_all_metrics()
                
                # Check for alerts
                alerts = await self.alert_manager.check_alerts(metrics)
                
                # Store metrics
                await self._store_metrics(metrics)
                
                # Send alerts if any
                for alert in alerts:
                    await self.alert_manager.send_alert(alert)
                
                # Sleep until next collection
                await asyncio.sleep(self.config.collection_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def get_realtime_metrics(self, instance_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current metrics for all instances or specific instance."""
        return await self.metrics_collector.get_current_metrics(instance_id)
    
    async def stream_metrics(self, instance_id: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """Stream real-time metrics."""
        while self._running:
            metrics = await self.get_realtime_metrics(instance_id)
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics
            }
            await asyncio.sleep(1)  # 1-second updates
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        metrics = await self.metrics_collector.get_system_metrics()
        alerts = await self.alert_manager.get_active_alerts()
        
        health_score = self._calculate_health_score(metrics, alerts)
        
        return {
            "health_score": health_score,
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "unhealthy",
            "metrics": metrics,
            "active_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.severity == "critical"])
        }
    
    async def get_historical_metrics(
        self, 
        instance_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[MetricPoint]:
        """Get historical metrics for an instance."""
        return await self.metrics_collector.get_historical_metrics(instance_id, start_time, end_time)
    
    def _calculate_health_score(self, metrics: Dict[str, Any], alerts: List[Alert]) -> float:
        """Calculate overall system health score."""
        base_score = 1.0
        
        # Deduct for active alerts
        for alert in alerts:
            if alert.severity == "critical":
                base_score -= 0.3
            elif alert.severity == "warning":
                base_score -= 0.1
            elif alert.severity == "info":
                base_score -= 0.05
        
        # Consider resource utilization
        cpu_usage = metrics.get("system", {}).get("cpu_usage", 0)
        memory_usage = metrics.get("system", {}).get("memory_usage", 0)
        
        if cpu_usage > 90 or memory_usage > 90:
            base_score -= 0.2
        elif cpu_usage > 80 or memory_usage > 80:
            base_score -= 0.1
        
        return max(0.0, base_score)
    
    async def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store metrics in time-series database."""
        # Implementation would depend on chosen storage backend
        # For now, just log metrics
        logger.debug(f"Storing metrics: {len(metrics)} data points")
