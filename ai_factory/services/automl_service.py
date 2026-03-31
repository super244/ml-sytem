from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.models.automl import AutoMLSearch, AutoMLRun
from ai_factory.schemas.automl import AutoMLSearchSummary, AutoMLSearchDetail, AutoMLRunSchema


class AutoMLService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_searches(self) -> list[AutoMLSearchSummary]:
        result = await self.db.execute(select(AutoMLSearch).order_by(AutoMLSearch.created_at.desc()))
        searches = result.scalars().all()
        summaries = []
        for s in searches:
            runs = await self._get_runs(s.id)
            summaries.append(self._to_summary(s, runs))
        return summaries

    async def get_search(self, search_id: str) -> AutoMLSearchDetail | None:
        result = await self.db.execute(select(AutoMLSearch).where(AutoMLSearch.id == search_id))
        search = result.scalar_one_or_none()
        if not search:
            return None
        runs = await self._get_runs(search_id)
        summary = self._to_summary(search, runs)
        run_schemas = [self._run_to_schema(r) for r in runs]
        return AutoMLSearchDetail(
            **summary.model_dump(),
            search_space=search.search_space or {},
            runs=run_schemas,
        )

    async def _get_runs(self, search_id: str) -> list[AutoMLRun]:
        result = await self.db.execute(select(AutoMLRun).where(AutoMLRun.search_id == search_id))
        return list(result.scalars().all())

    def _to_summary(self, search: AutoMLSearch, runs: list[AutoMLRun]) -> AutoMLSearchSummary:
        total = len(runs)
        completed = sum(1 for r in runs if r.status == "completed")
        running = sum(1 for r in runs if r.status == "running")
        pruned = sum(1 for r in runs if r.status == "pruned")
        promoted = sum(1 for r in runs if r.status == "promoted")
        return AutoMLSearchSummary(
            id=search.id,
            strategy=search.strategy,
            status=search.status,
            total_runs=total,
            completed_runs=completed,
            running_runs=running,
            pruned_runs=pruned,
            promoted_runs=promoted,
            best_loss=search.best_loss,
            best_config=search.best_config,
            created_at=search.created_at,
        )

    def _run_to_schema(self, run: AutoMLRun) -> AutoMLRunSchema:
        return AutoMLRunSchema(
            id=run.id,
            search_id=run.search_id,
            status=run.status,
            hyperparams=run.hyperparams or {},
            eval_loss=run.eval_loss,
            composite_score=run.composite_score,
            training_minutes=run.training_minutes,
            step_pruned=run.step_pruned,
            prune_reason=run.prune_reason,
            job_id=run.job_id,
        )
