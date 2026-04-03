import importlib
import logging
import platform
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def get_system_ram_gb() -> int:
    try:
        if platform.system() == "Darwin":
            res = subprocess.run(  # nosec B603
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            return int(res.stdout.strip()) // (1024**3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // (1024**2)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.debug("unable to read system memory: %s", exc)
    return 16  # Fallback


def _get_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None
    except RuntimeError as exc:
        logger.debug("torch import failed: %s", exc)
        return None


def get_cluster_nodes() -> list[dict[str, Any]]:
    torch = _get_torch()
    # Local Node Logic
    node_name = "Local Engine"
    memory_str = f"{get_system_ram_gb()}GB"
    usage_pct = 0
    hw_type = "CPU"
    status = "online"

    if torch is not None and torch.cuda.is_available():
        hw_type = torch.cuda.get_device_name(0)
        try:
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            if reserved > 0:
                usage_pct = int((allocated / reserved) * 100)
            else:
                usage_pct = 10
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.debug("cuda memory probe failed: %s", exc)
    elif torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        hw_type = "MPS (Apple Silicon)"
        try:
            alloc = torch.mps.current_allocated_memory()
            # recommended_max_memory is often around total RAM * 0.7 for Unified memory
            rec = get_system_ram_gb() * (1024**3) * 0.7
            if rec > 0:
                usage_pct = min(100, int((alloc / rec) * 100))
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.debug("mps memory probe failed: %s", exc)

    local_node = {
        "id": "local-primary",
        "name": node_name,
        "type": hw_type,
        "memory": memory_str,
        "usage": usage_pct,
        "status": status,
        "activeJobs": 1,
    }

    # Mocking disconnected/ssh nodes
    remote_node = {
        "id": "remote-cluster-1",
        "name": "Cluster Worker 1",
        "type": "NVIDIA T4",
        "memory": "16GB",
        "usage": 0,
        "status": "idle",
        "activeJobs": 0,
    }

    return [local_node, remote_node]
