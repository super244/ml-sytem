from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from inference.app.dependencies import get_instance_service
from inference.app.schemas import (
    AgentTypeFilter,
    EventLevelFilter,
    InstanceCreateRequest,
    OrchestrationEventListResponse,
    OrchestrationRecoveryResponse,
    OrchestrationRunDetail,
    OrchestrationRunListResponse,
    OrchestrationSummaryResponse,
    OrchestrationTaskListResponse,
    RunStatusFilter,
    TaskStatusFilter,
    TaskTypeFilter,
)

router = APIRouter(tags=["orchestration"])


def _raise_orchestration_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    raise HTTPException(status_code=503, detail=f"Orchestration service unavailable: {str(exc)}") from exc


@router.get("/orchestration/runs", response_model=OrchestrationRunListResponse)
def list_runs(
    status: RunStatusFilter | None = Query(default=None),
    limit: int | None = Query(default=100, ge=1, le=1000),
    service: Any = Depends(get_instance_service),
) -> OrchestrationRunListResponse:
    try:
        return service.list_orchestration_runs(status=status, limit=limit)
    except Exception as exc:
        _raise_orchestration_error(exc)


@router.post("/orchestration/runs", response_model=OrchestrationRunDetail)
def create_run(request: InstanceCreateRequest, service: Any = Depends(get_instance_service)) -> OrchestrationRunDetail:
    try:
        detail = service.create_instance(request)
        run_id = detail.orchestration_run_id or detail.id
        return service.get_orchestration_run(run_id)
    except Exception as exc:
        _raise_orchestration_error(exc)


@router.get("/orchestration/runs/{run_id}", response_model=OrchestrationRunDetail)
def get_run(run_id: str, service: Any = Depends(get_instance_service)) -> OrchestrationRunDetail:
    try:
        return service.get_orchestration_run(run_id)
    except Exception as exc:
        _raise_orchestration_error(exc)


@router.get("/orchestration/runs/{run_id}/tasks", response_model=OrchestrationTaskListResponse)
def list_tasks(
    run_id: str,
    status: TaskStatusFilter | None = Query(default=None),
    task_type: TaskTypeFilter | None = Query(default=None),
    agent_type: AgentTypeFilter | None = Query(default=None),
    limit: int | None = Query(default=200, ge=1, le=1000),
    service: Any = Depends(get_instance_service),
) -> OrchestrationTaskListResponse:
    try:
        return service.list_orchestration_tasks(
            run_id,
            status=status,
            task_type=task_type,
            agent_type=agent_type,
            limit=limit,
        )
    except Exception as exc:
        _raise_orchestration_error(exc)


@router.get("/orchestration/runs/{run_id}/events", response_model=OrchestrationEventListResponse)
def list_events(
    run_id: str,
    limit: int | None = Query(default=200, ge=1, le=1000),
    event_type: str | None = Query(default=None, min_length=1),
    level: EventLevelFilter | None = Query(default=None),
    service: Any = Depends(get_instance_service),
) -> OrchestrationEventListResponse:
    try:
        return service.list_orchestration_events(run_id, limit=limit, event_type=event_type, level=level)
    except Exception as exc:
        _raise_orchestration_error(exc)


@router.post("/orchestration/runs/{run_id}/cancel", response_model=OrchestrationRunDetail)
def cancel_run(run_id: str, service: Any = Depends(get_instance_service)) -> OrchestrationRunDetail:
    try:
        return service.cancel_orchestration_run(run_id)
    except Exception as exc:
        _raise_orchestration_error(exc)


@router.post("/orchestration/recover-stalled", response_model=OrchestrationRecoveryResponse)
def recover_stalled_tasks(service: Any = Depends(get_instance_service)) -> OrchestrationRecoveryResponse:
    try:
        return service.recover_stalled_orchestration_tasks()
    except Exception as exc:
        _raise_orchestration_error(exc)


@router.post("/orchestration/tasks/{task_id}/retry", response_model=OrchestrationRunDetail)
def retry_task(task_id: str, service: Any = Depends(get_instance_service)) -> OrchestrationRunDetail:
    try:
        return service.retry_orchestration_task(task_id)
    except Exception as exc:
        _raise_orchestration_error(exc)


@router.get("/orchestration/summary", response_model=OrchestrationSummaryResponse)
def get_summary(service: Any = Depends(get_instance_service)) -> OrchestrationSummaryResponse:
    try:
        return service.get_orchestration_summary()
    except Exception as exc:
        _raise_orchestration_error(exc)
