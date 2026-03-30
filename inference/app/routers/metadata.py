from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

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
def list_models() -> dict:
    models = get_metadata_service().models()
    return {
        "object": "list",
        "data": [_openai_model_record(model) for model in models],
        "models": models,
    }


@router.get("/models/{model_id}")
def get_model(model_id: str) -> dict:
    models = get_metadata_service().models()
    for model in models:
        if model["name"] == model_id:
            return _openai_model_record(model)
    raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")


@router.get("/prompts")
def list_prompts(limit: int = 12) -> dict:
    return get_metadata_service().prompt_library(limit=limit)


@router.get("/benchmarks")
def list_benchmarks() -> dict:
    return {"benchmarks": get_metadata_service().benchmark_library()}


@router.get("/runs")
def list_runs() -> dict:
    return {"runs": get_metadata_service().runs()}
