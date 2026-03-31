from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.database import get_session
from ai_factory.services.automl_service import AutoMLService
from ai_factory.schemas.automl import AutoMLSearchSummary, AutoMLSearchDetail

router = APIRouter(prefix="/automl", tags=["automl"])


@router.get("/searches", response_model=list[AutoMLSearchSummary])
async def list_searches(db: AsyncSession = Depends(get_session)):
    service = AutoMLService(db)
    return await service.list_searches()


@router.get("/searches/{search_id}", response_model=AutoMLSearchDetail)
async def get_search(search_id: str, db: AsyncSession = Depends(get_session)):
    service = AutoMLService(db)
    result = await service.get_search(search_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Search not found")
    return result
