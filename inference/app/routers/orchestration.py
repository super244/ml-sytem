from __future__ import annotations

from fastapi import APIRouter, Query

from inference.app.dependencies import get_instance_service
from inference.app.schemas import (
    InstanceCreateRequest,
    OrchestrationEventListResponse,
    OrchestrationRunDetail,
    OrchestrationRunListResponse,
    OrchestrationSummaryResponse,
    OrchestrationTaskListResponse,
)

router = APIRouter(tags=["orchestration"])


@router.get("/orchestration/runs", response_model=OrchestrationRunListResponse)
def list_runs() -> OrchestrationRunListResponse:
    try:
        return get_instance_service().list_orchestration_runs()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc


@router.post("/orchestration/runs", response_model=OrchestrationRunDetail)
def create_run(request: InstanceCreateRequest) -> OrchestrationRunDetail:
    try:
        detail = get_instance_service().create_instance(request)
        run_id = detail.orchestration_run_id or detail.id
        return get_instance_service().get_orchestration_run(run_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc


@router.get("/orchestration/runs/{run_id}", response_model=OrchestrationRunDetail)
def get_run(run_id: str) -> OrchestrationRunDetail:
    try:
        return get_instance_service().get_orchestration_run(run_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc


@router.get("/orchestration/runs/{run_id}/tasks", response_model=OrchestrationTaskListResponse)
def list_tasks(run_id: str) -> OrchestrationTaskListResponse:
    try:
        return get_instance_service().list_orchestration_tasks(run_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc


@router.get("/orchestration/runs/{run_id}/events", response_model=OrchestrationEventListResponse)
def list_events(
    run_id: str,
    limit: int | None = Query(default=None, ge=1, le=1000),
) -> OrchestrationEventListResponse:
    try:
        return get_instance_service().list_orchestration_events(run_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc


@router.post("/orchestration/runs/{run_id}/cancel", response_model=OrchestrationRunDetail)
def cancel_run(run_id: str) -> OrchestrationRunDetail:
    try:
        return get_instance_service().cancel_orchestration_run(run_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc


@router.post("/orchestration/tasks/{task_id}/retry", response_model=OrchestrationRunDetail)
def retry_task(task_id: str) -> OrchestrationRunDetail:
    try:
        return get_instance_service().retry_orchestration_task(task_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc


@router.get("/orchestration/summary", response_model=OrchestrationSummaryResponse)
def get_summary() -> OrchestrationSummaryResponse:
    try:
        return get_instance_service().get_orchestration_summary()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc
