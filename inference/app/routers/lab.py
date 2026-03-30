from __future__ import annotations

from fastapi import APIRouter

from inference.app.config import get_settings
from inference.app.dependencies import get_instance_service
from inference.app.services.mission_control_service import MissionControlService, MissionControlSnapshot

router = APIRouter(tags=["lab"])


def _mission_control_snapshot() -> MissionControlSnapshot:
    settings = get_settings()
    service = MissionControlService(settings, instance_service=get_instance_service())
    return service.snapshot()


@router.get("/lab/mission-control", response_model=MissionControlSnapshot)
@router.get("/mission-control", response_model=MissionControlSnapshot)
def mission_control() -> MissionControlSnapshot:
    return _mission_control_snapshot()
