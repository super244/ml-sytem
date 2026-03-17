from __future__ import annotations

from fastapi import APIRouter

from inference.app.dependencies import get_metadata_service


router = APIRouter(tags=["metadata"])


@router.get("/models")
def list_models() -> dict:
    return {"models": get_metadata_service().models()}


@router.get("/datasets")
def list_datasets() -> dict:
    return get_metadata_service().dataset_dashboard()


@router.get("/prompts")
def list_prompts(limit: int = 12) -> dict:
    return get_metadata_service().prompt_library(limit=limit)


@router.get("/benchmarks")
def list_benchmarks() -> dict:
    return {"benchmarks": get_metadata_service().benchmark_library()}


@router.get("/runs")
def list_runs() -> dict:
    return {"runs": get_metadata_service().runs()}
