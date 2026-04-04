from __future__ import annotations

from fastapi import APIRouter, status
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


def _loop_service() -> AutonomousLoopService:
    return AutonomousLoopService(get_settings(), instance_service=get_instance_service())


def _lab_service() -> AutonomousLabService:
    return AutonomousLabService(get_settings(), instance_service=get_instance_service())


def _mission_control_service() -> MissionControlService:
    return MissionControlService(get_settings(), instance_service=get_instance_service())


@router.get("", response_model=AutonomousLoopSnapshot)
def autonomous_snapshot() -> AutonomousLoopSnapshot:
    return _loop_service().snapshot()


@router.get("/overview", response_model=AutonomyOverview)
def autonomous_overview() -> AutonomyOverview:
    return _mission_control_service().autonomy_overview()


@router.post("/plan", response_model=AutonomousLoopRun)
def plan_autonomous_loop(request: AutonomousLoopPlanRequest) -> AutonomousLoopRun:
    return _loop_service().plan(max_actions=request.max_actions)


@router.post("/run", response_model=AutonomousLoopRun)
def run_autonomous_loop(request: AutonomousLoopRunRequest) -> AutonomousLoopRun:
    return _loop_service().execute(
        max_actions=request.max_actions,
        dry_run=request.dry_run,
        start_instances=request.start_instances,
    )


@router.post("/execute", response_model=AutonomousLoopRun)
def execute_autonomous_loop(request: AutonomousLoopRunRequest) -> AutonomousLoopRun:
    return run_autonomous_loop(request)


@router.get("/campaigns", response_model=AutonomousCampaignSnapshot)
def autonomous_campaigns() -> AutonomousCampaignSnapshot:
    return _lab_service().snapshot()


@router.post("/campaigns/run", response_model=AutonomousExperimentResponse, status_code=status.HTTP_202_ACCEPTED)
def run_autonomous_campaign(request: AutonomousExperimentRequest) -> AutonomousExperimentResponse:
    campaign = _lab_service().create_campaign(request)
    return AutonomousExperimentResponse(
        experiment_id=campaign.campaign_id,
        status=campaign.status,
        message=f"Autonomous campaign '{request.experiment_name}' created.",
        campaign=campaign,
    )


@router.get("/campaigns/{experiment_id}", response_model=AutonomousExperimentResponse)
def autonomous_campaign_detail(experiment_id: str) -> AutonomousExperimentResponse:
    campaign = _lab_service().get_campaign(experiment_id)
    return AutonomousExperimentResponse(
        experiment_id=campaign.campaign_id,
        status=campaign.status,
        message=f"Autonomous campaign '{campaign.experiment_name}'.",
        campaign=campaign,
    )
