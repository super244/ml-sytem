from __future__ import annotations

from fastapi import APIRouter, Query

from inference.app.dependencies import get_instance_service
from inference.app.schemas import (
    InstanceCreateRequest,
    InstanceDeployRequest,
    InstanceDetail,
    InstanceEvaluateRequest,
    InstanceInferenceRequest,
    InstanceListResponse,
    InstanceLogsResponse,
    InstanceMetricsResponse,
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
