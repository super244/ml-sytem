from fastapi import APIRouter
from ai_factory.schemas.inference import FeedbackFlagRequest, FeedbackFlagResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/flag", response_model=FeedbackFlagResponse)
async def flag_feedback(request: FeedbackFlagRequest):
    return FeedbackFlagResponse(
        status="flagged",
        completion_id=request.completion_id,
    )
