from datetime import datetime, timezone, timedelta
from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.models.job import TrainingJob
from ai_factory.schemas.job import JobSummary, JobDetail, JobCreateRequest, LogLine


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_jobs(self) -> list[JobSummary]:
        result = await self.db.execute(select(TrainingJob).order_by(TrainingJob.created_at.desc()))
        jobs = result.scalars().all()
        return [self._to_summary(j) for j in jobs]

    async def get_job(self, job_id: str) -> JobDetail | None:
        result = await self.db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return None
        return self._to_detail(job)

    async def create_job(self, request: JobCreateRequest) -> JobDetail:
        now = datetime.now(timezone.utc)
        job = TrainingJob(
            id=str(uuid4()),
            name=f"run-{str(uuid4())[:4]}-{request.type}",
            type=request.type,
            status="queued",
            base_model=request.base_model,
            config_json=request.config.model_dump(),
            dataset_id=request.dataset_id,
            total_steps=request.config.max_steps,
            started_at=now,
            created_at=now,
            gpu_utilization=[],
            vram_used_gb=0.0,
            vram_total_gb=80.0,
            loss_history=[],
            step_history=[],
            logs_tail=[],
        )
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        return self._to_detail(job)

    async def stop_job(self, job_id: str) -> bool:
        result = await self.db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return False
        job.status = "stopped"
        job.completed_at = datetime.now(timezone.utc)
        await self.db.commit()
        return True

    async def get_running_jobs(self) -> list[TrainingJob]:
        result = await self.db.execute(
            select(TrainingJob).where(TrainingJob.status == "running")
        )
        return list(result.scalars().all())

    def _to_summary(self, job: TrainingJob) -> JobSummary:
        progress = job.current_step / max(job.total_steps, 1)
        loss_delta = None
        if job.loss_history and len(job.loss_history) >= 2:
            loss_delta = job.loss_history[-1] - job.loss_history[-2]

        eta_seconds = None
        if job.status == "running" and job.current_step > 0:
            remaining = job.total_steps - job.current_step
            eta_seconds = int(remaining * 0.5)

        estimated_completion = None
        if eta_seconds is not None:
            estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)

        return JobSummary(
            id=job.id,
            name=job.name,
            type=job.type,
            status=job.status,
            base_model=job.base_model,
            progress=progress,
            current_step=job.current_step,
            total_steps=job.total_steps,
            current_loss=job.current_loss,
            loss_delta=loss_delta,
            eta_seconds=eta_seconds,
            gpu_utilization=job.gpu_utilization or [],
            vram_used_gb=job.vram_used_gb,
            vram_total_gb=job.vram_total_gb,
            started_at=job.started_at or job.created_at,
            estimated_completion=estimated_completion,
            node_id=job.node_id,
        )

    def _to_detail(self, job: TrainingJob) -> JobDetail:
        summary = self._to_summary(job)
        logs = []
        for log in (job.logs_tail or []):
            if isinstance(log, dict):
                logs.append(LogLine(**log))

        return JobDetail(
            **summary.model_dump(),
            config=job.config_json or {},
            dataset_hash=job.dataset_hash,
            parent_model_id=job.parent_model_id,
            loss_history=job.loss_history or [],
            step_history=job.step_history or [],
            eval_scores=job.eval_scores,
            logs_tail=logs,
            lineage_id=job.lineage_node_id,
        )
