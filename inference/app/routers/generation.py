from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from inference.app.dependencies import get_generation_service
from inference.app.parameters import GenerationParameters
from inference.app.schemas import (
    CompareRequest,
    CompareResponse,
    GenerateBatchRequest,
    GenerateBatchResponse,
    GenerateRequest,
    GenerateResponse,
    VerifyRequest,
    VerifyResponse,
)

router = APIRouter(tags=["generation"])


@router.post("/verify", response_model=VerifyResponse)
def verify_answer(request: VerifyRequest) -> VerifyResponse:
    from ai_factory.core.answers import verify_prediction

    verification = verify_prediction(
        request.prediction_text or f"Final Answer: {request.candidate_answer}",
        request.reference_answer,
        request.step_checks,
    )
    return VerifyResponse(
        equivalent=verification.equivalent,
        step_correctness=verification.step_correctness,
        formatting_failure=verification.formatting_failure,
        arithmetic_slip=verification.arithmetic_slip,
        error_type=verification.error_type,
        details={"final_answer": verification.final_answer},
    )


@router.post("/generate", response_model=GenerateResponse)
def generate_answer(request: GenerateRequest, service: Any = Depends(get_generation_service)) -> GenerateResponse:

    try:
        primary = service.generate(
            GenerationParameters(
                question=request.question,
                model_variant=request.model_variant,
                prompt_preset=request.prompt_preset,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_new_tokens,
                show_reasoning=request.show_reasoning,
                difficulty_target=request.difficulty_target,
                num_samples=request.num_samples,
                use_calculator=request.use_calculator,
                solver_mode=request.solver_mode,
                output_format=request.output_format,
                use_cache=request.use_cache,
                reference_answer=request.reference_answer,
                step_checks=request.step_checks,
            )
        )
        comparison = None
        compare_to_model = request.compare_to_model or (
            "base" if request.compare_to_base and request.model_variant != "base" else None
        )
        if compare_to_model:
            comparison = service.generate(
                GenerationParameters(
                    question=request.question,
                    model_variant=compare_to_model,
                    prompt_preset=request.prompt_preset,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=request.max_new_tokens,
                    show_reasoning=request.show_reasoning,
                    difficulty_target=request.difficulty_target,
                    num_samples=request.num_samples,
                    use_calculator=request.use_calculator,
                    solver_mode=request.solver_mode,
                    output_format=request.output_format,
                    use_cache=request.use_cache,
                    reference_answer=request.reference_answer,
                    step_checks=request.step_checks,
                )
            )
        primary["comparison"] = comparison
        return GenerateResponse(**primary)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/compare", response_model=CompareResponse)
def compare_models(request: CompareRequest, service: Any = Depends(get_generation_service)) -> CompareResponse:

    try:
        comparison = service.compare(
            GenerationParameters(
                question=request.question,
                model_variant=request.primary_model,
                prompt_preset=request.prompt_preset,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_new_tokens,
                show_reasoning=request.show_reasoning,
                difficulty_target=request.difficulty_target,
                num_samples=request.num_samples,
                use_calculator=request.use_calculator,
                solver_mode=request.solver_mode,
                output_format=request.output_format,
                use_cache=request.use_cache,
                reference_answer=request.reference_answer,
                step_checks=request.step_checks,
            ),
            GenerationParameters(
                question=request.question,
                model_variant=request.secondary_model,
                prompt_preset=request.prompt_preset,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_new_tokens,
                show_reasoning=request.show_reasoning,
                difficulty_target=request.difficulty_target,
                num_samples=request.num_samples,
                use_calculator=request.use_calculator,
                solver_mode=request.solver_mode,
                output_format=request.output_format,
                use_cache=request.use_cache,
                reference_answer=request.reference_answer,
                step_checks=request.step_checks,
            ),
        )
        return CompareResponse(**comparison)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate/batch", response_model=GenerateBatchResponse)
def generate_batch(
    request: GenerateBatchRequest, service: Any = Depends(get_generation_service)
) -> GenerateBatchResponse:
    results = [generate_answer(item, service=service) for item in request.requests]
    return GenerateBatchResponse(results=results)
