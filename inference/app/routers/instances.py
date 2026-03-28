from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from inference.app.dependencies import get_instance_service
from inference.app.schemas import (
    FoundationOverviewResponse,
    InstanceActionRequest,
    InstanceCreateRequest,
    InstanceDeployRequest,
    InstanceDetail,
    InstanceEvaluateRequest,
    InstanceInferenceRequest,
    InstanceListResponse,
    InstanceLogsResponse,
    InstanceMetricsResponse,
    InstanceStreamResponse,
)

router = APIRouter(tags=["instances"])


@router.get("/instances", response_model=InstanceListResponse)
def list_instances(
    instance_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    parent_instance_id: str | None = Query(default=None),
) -> InstanceListResponse:
    service = get_instance_service()
    return InstanceListResponse(
        instances=service.list_instances(
            instance_type=instance_type,
            status=status,
            parent_instance_id=parent_instance_id,
        )
    )


@router.post("/instances", response_model=InstanceDetail)
def create_instance(request: InstanceCreateRequest) -> InstanceDetail:
    return get_instance_service().create_instance(request)


@router.get("/instances/{instance_id}", response_model=InstanceDetail)
def get_instance(instance_id: str) -> InstanceDetail:
    return get_instance_service().get_instance(instance_id)


@router.get("/instances/{instance_id}/logs", response_model=InstanceLogsResponse)
def get_instance_logs(instance_id: str) -> InstanceLogsResponse:
    return get_instance_service().get_logs(instance_id)


@router.get("/instances/{instance_id}/metrics", response_model=InstanceMetricsResponse)
def get_instance_metrics(instance_id: str) -> InstanceMetricsResponse:
    return get_instance_service().get_metrics(instance_id)


@router.get("/instances/{instance_id}/live", response_model=InstanceStreamResponse)
def get_instance_live(instance_id: str) -> InstanceStreamResponse:
    return get_instance_service().get_live_snapshot(instance_id)


@router.get("/instances/{instance_id}/stream")
async def stream_instance(
    instance_id: str,
    poll_interval_s: float = Query(default=1.0, ge=0.25, le=30.0),
    log_tail_chars: int = Query(default=4000, ge=256, le=20000),
    metric_tail_points: int = Query(default=200, ge=10, le=2000),
    event_limit: int = Query(default=50, ge=1, le=500),
    task_limit: int = Query(default=20, ge=1, le=200),
) -> StreamingResponse:
    service = get_instance_service()
    service.get_live_snapshot(instance_id)

    async def event_stream():
        async for frame in service.control.stream_instance(
            instance_id,
            poll_interval_s=poll_interval_s,
            log_tail_chars=log_tail_chars,
            metric_tail_points=metric_tail_points,
            event_limit=event_limit,
            task_limit=task_limit,
        ):
            yield f"event: snapshot\ndata: {frame.model_dump_json()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/instances/{instance_id}/evaluate", response_model=InstanceDetail)
def evaluate_instance(instance_id: str, request: InstanceEvaluateRequest | None = None) -> InstanceDetail:
    return get_instance_service().evaluate_instance(instance_id, request)


@router.post("/instances/{instance_id}/inference", response_model=InstanceDetail)
def start_inference_instance(
    instance_id: str,
    request: InstanceInferenceRequest | None = None,
) -> InstanceDetail:
    return get_instance_service().start_inference_instance(instance_id, request)


@router.post("/instances/{instance_id}/deploy", response_model=InstanceDetail)
def deploy_instance(instance_id: str, request: InstanceDeployRequest) -> InstanceDetail:
    return get_instance_service().deploy_instance(instance_id, request)


@router.post("/instances/{instance_id}/actions", response_model=InstanceDetail)
def run_instance_action(instance_id: str, request: InstanceActionRequest) -> InstanceDetail:
    return get_instance_service().run_instance_action(instance_id, request)


@router.get("/foundation", response_model=FoundationOverviewResponse)
def get_foundation() -> FoundationOverviewResponse:
    return get_instance_service().get_foundation_overview()
