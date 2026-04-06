from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from inference.app.dependencies import get_metadata_service

router = APIRouter(tags=["metadata"])


def _openai_model_record(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model["name"],
        "object": "model",
        "created": 0,
        "owned_by": "ai-factory",
        "permission": [],
        "root": model["name"],
        "parent": None,
        "metadata": model,
    }


@router.get("/models")
def list_models(service: Any = Depends(get_metadata_service)) -> dict:
    try:
        models = service.models()
        return {
            "object": "list",
            "data": [_openai_model_record(model) for model in models],
            "models": models,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Metadata service unavailable: {str(exc)}") from exc


@router.get("/models/{model_id}")
def get_model(model_id: str, service: Any = Depends(get_metadata_service)) -> dict:
    try:
        models = service.models()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Metadata service unavailable: {str(exc)}") from exc

    for model in models:
        if model["name"] == model_id:
            return _openai_model_record(model)
    raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")


@router.get("/prompts")
def list_prompts(limit: int = 12, service: Any = Depends(get_metadata_service)) -> dict:
    try:
        return service.prompt_library(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prompt library unavailable: {str(exc)}") from exc


@router.get("/benchmarks")
def list_benchmarks() -> dict:
    try:
        return {"benchmarks": get_metadata_service().benchmark_library()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Benchmark library unavailable: {str(exc)}") from exc


@router.get("/runs")
def list_runs() -> dict:
    try:
        return {"runs": get_metadata_service().runs()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Runs library unavailable: {str(exc)}") from exc
