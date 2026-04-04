from __future__ import annotations

import time

from fastapi import APIRouter

from inference.app.config import get_settings
from inference.app.dependencies import get_instance_service
from inference.app.services.mission_control_service import MissionControlService, MissionControlSnapshot

router = APIRouter(tags=["lab"])
_MISSION_CONTROL_CACHE_TTL_S = 1.0
_mission_control_cache: dict[str, tuple[float, MissionControlSnapshot]] = {}


def _mission_control_snapshot() -> MissionControlSnapshot:
    settings = get_settings()
    repo_root = settings.repo_root
    now = time.monotonic()
    cached = _mission_control_cache.get(repo_root)
    if cached and now - cached[0] < _MISSION_CONTROL_CACHE_TTL_S:
        return cached[1]
    service = MissionControlService(settings, instance_service=get_instance_service())
    snapshot = service.snapshot()
    _mission_control_cache[repo_root] = (now, snapshot)
    return snapshot


@router.get("/lab/mission-control", response_model=MissionControlSnapshot)
def mission_control() -> MissionControlSnapshot:
    return _mission_control_snapshot()
