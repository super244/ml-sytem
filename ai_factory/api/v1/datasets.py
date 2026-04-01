from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.database import get_session
from ai_factory.services.dataset_service import DatasetService
from ai_factory.schemas.dataset import DatasetSummary, DatasetSamplesResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(db: AsyncSession = Depends(get_session)):
    service = DatasetService(db)
    return await service.list_datasets()


@router.get("/{dataset_id}/samples", response_model=DatasetSamplesResponse)
async def get_samples(
    dataset_id: str,
    page: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
):
    service = DatasetService(db)
    result = await service.get_samples(dataset_id, page, limit)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return result
