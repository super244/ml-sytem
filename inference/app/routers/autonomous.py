from __future__ import annotations
from typing import Any

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from inference.app.config import get_settings
from inference.app.dependencies import get_instance_service
from inference.app.services.autonomous_lab import (
    AutonomousExperimentRequest,
    AutonomousExperimentResponse,
    AutonomousLabService,
)
from inference.app.services.autonomous_lab import (
    AutonomousLoopSnapshot as AutonomousCampaignSnapshot,
)
from inference.app.services.autonomous_loop_service import (
    AutonomousLoopRun,
    AutonomousLoopService,
    AutonomousLoopSnapshot,
)
from inference.app.services.mission_control_service import AutonomyOverview, MissionControlService

router = APIRouter(prefix="/experiments/autonomous", tags=["autonomous"])


class AutonomousLoopRunRequest(BaseModel):
    max_actions: int = Field(default=2, ge=1, le=8)
    dry_run: bool = False
    start_instances: bool = False


class AutonomousLoopPlanRequest(BaseModel):
    max_actions: int = Field(default=6, ge=1, le=25)


def _loop_service(settings: Any = Depends(get_settings), instance_service: Any = Depends(get_instance_service)) -> AutonomousLoopService:
    return AutonomousLoopService(settings, instance_service=instance_service)


def _lab_service(settings: Any = Depends(get_settings), instance_service: Any = Depends(get_instance_service)) -> AutonomousLabService:
    return AutonomousLabService(settings, instance_service=instance_service)


def _mission_control_service(settings: Any = Depends(get_settings), instance_service: Any = Depends(get_instance_service)) -> MissionControlService:
    return MissionControlService(settings, instance_service=instance_service)


@router.get("", response_model=AutonomousLoopSnapshot)
def autonomous_snapshot(service: AutonomousLoopService = Depends(_loop_service)) -> AutonomousLoopSnapshot:
    return service.snapshot()


@router.get("/overview", response_model=AutonomyOverview)
def autonomous_overview(service: MissionControlService = Depends(_mission_control_service)) -> AutonomyOverview:
    return service.autonomy_overview()


@router.post("/plan", response_model=AutonomousLoopRun)
def plan_autonomous_loop(request: AutonomousLoopPlanRequest, service: AutonomousLoopService = Depends(_loop_service)) -> AutonomousLoopRun:
    return service.plan(max_actions=request.max_actions)


@router.post("/run", response_model=AutonomousLoopRun)
def run_autonomous_loop(request: AutonomousLoopRunRequest, service: AutonomousLoopService = Depends(_loop_service)) -> AutonomousLoopRun:
    return service.execute(
        max_actions=request.max_actions,
        dry_run=request.dry_run,
        start_instances=request.start_instances,
    )


@router.post("/execute", response_model=AutonomousLoopRun)
def execute_autonomous_loop(request: AutonomousLoopRunRequest, service: AutonomousLoopService = Depends(_loop_service)) -> AutonomousLoopRun:
    return service.execute(
        max_actions=request.max_actions,
        dry_run=request.dry_run,
        start_instances=request.start_instances,
    )


@router.get("/campaigns", response_model=AutonomousCampaignSnapshot)
def autonomous_campaigns(service: AutonomousLabService = Depends(_lab_service)) -> AutonomousCampaignSnapshot:
    return service.snapshot()


@router.post("/campaigns/run", response_model=AutonomousExperimentResponse, status_code=status.HTTP_202_ACCEPTED)
def run_autonomous_campaign(request: AutonomousExperimentRequest, service: AutonomousLabService = Depends(_lab_service)) -> AutonomousExperimentResponse:
    campaign = service.create_campaign(request)
    return AutonomousExperimentResponse(
        experiment_id=campaign.campaign_id,
        status=campaign.status,
        message=f"Autonomous campaign '{request.experiment_name}' created.",
        campaign=campaign,
    )


@router.get("/campaigns/{experiment_id}", response_model=AutonomousExperimentResponse)
def autonomous_campaign_detail(experiment_id: str, service: AutonomousLabService = Depends(_lab_service)) -> AutonomousExperimentResponse:
    campaign = service.get_campaign(experiment_id)
    return AutonomousExperimentResponse(
        experiment_id=campaign.campaign_id,
        status=campaign.status,
        message=f"Autonomous campaign '{campaign.experiment_name}'.",
        campaign=campaign,
    )
