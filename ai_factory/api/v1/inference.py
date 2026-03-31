from fastapi import APIRouter
from ai_factory.services.inference_service import InferenceService
from ai_factory.schemas.inference import CompletionRequest, CompletionResponse

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    service = InferenceService()
    return await service.generate_completion(request)
