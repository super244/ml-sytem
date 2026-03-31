from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.models.dataset import Dataset
from ai_factory.schemas.dataset import DatasetSummary, PackSummary, Sample, DatasetSamplesResponse


class DatasetService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_datasets(self) -> list[DatasetSummary]:
        result = await self.db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
        datasets = result.scalars().all()
        return [self._to_summary(d) for d in datasets]

    async def get_samples(self, dataset_id: str, page: int = 0, limit: int = 20) -> DatasetSamplesResponse | None:
        result = await self.db.execute(select(Dataset).where(Dataset.id == dataset_id))
        ds = result.scalar_one_or_none()
        if not ds:
            return None
        samples_data = ds.samples_json or []
        total = len(samples_data)
        start = page * limit
        end = start + limit
        page_samples = samples_data[start:end]
        samples = [Sample(**s) for s in page_samples]
        return DatasetSamplesResponse(samples=samples, total=total)

    def _to_summary(self, ds: Dataset) -> DatasetSummary:
        pack_summary = None
        if ds.pack_summary_json:
            pack_summary = PackSummary(**ds.pack_summary_json)
        return DatasetSummary(
            id=ds.id,
            name=ds.name,
            domain=ds.domain,
            status=ds.status,
            sample_count=ds.sample_count,
            quality_score_mean=ds.quality_score_mean,
            quality_score_p10=ds.quality_score_p10,
            quality_score_p90=ds.quality_score_p90,
            size_mb=ds.size_mb,
            content_hash=ds.content_hash,
            pipeline_config_hash=ds.pipeline_config_hash,
            git_sha=ds.git_sha,
            created_at=ds.created_at,
            pack_summary=pack_summary,
        )
