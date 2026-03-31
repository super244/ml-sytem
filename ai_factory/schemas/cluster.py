from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict


class GPUMetric(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    index: int
    name: str
    utilization: float
    vram_used_gb: float
    vram_total_gb: float
    temperature_celsius: float
    power_draw_watts: float
    memory_bandwidth_gbps: float | None = None


class NodeStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    type: Literal["local_gpu", "apple_silicon", "ssh_remote"]
    status: Literal["online", "offline", "degraded", "maintenance"]
    gpus: list[GPUMetric]
    cpu_utilization: float
    ram_used_gb: float
    ram_total_gb: float
    network_rx_mbps: float
    network_tx_mbps: float
    active_jobs: list[str]
    cost_per_hour: float | None = None
    last_seen: datetime
