from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.database import get_session
from ai_factory.services.model_service import ModelService
from ai_factory.schemas.model_registry import ModelSummary, LineageGraph

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelSummary])
async def list_models(db: AsyncSession = Depends(get_session)):
    service = ModelService(db)
    return await service.list_models()


@router.get("/{model_id}/lineage", response_model=LineageGraph)
async def get_lineage(model_id: str, db: AsyncSession = Depends(get_session)):
    service = ModelService(db)
    result = await service.get_lineage(model_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return result
