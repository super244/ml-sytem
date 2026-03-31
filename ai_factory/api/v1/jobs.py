from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.database import get_session
from ai_factory.services.job_service import JobService
from ai_factory.schemas.job import JobSummary, JobDetail, JobCreateRequest, JobStopResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobSummary])
async def list_jobs(db: AsyncSession = Depends(get_session)):
    service = JobService(db)
    return await service.list_jobs()


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(job_id: str, db: AsyncSession = Depends(get_session)):
    service = JobService(db)
    job = await service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("", response_model=JobDetail)
async def create_job(request: JobCreateRequest, db: AsyncSession = Depends(get_session)):
    service = JobService(db)
    return await service.create_job(request)


@router.post("/{job_id}/stop", response_model=JobStopResponse)
async def stop_job(job_id: str, db: AsyncSession = Depends(get_session)):
    service = JobService(db)
    success = await service.stop_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStopResponse(status="stopping", job_id=job_id)
