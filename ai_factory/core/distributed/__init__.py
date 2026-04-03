"""Lightweight distributed job coordination primitives."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any


class DistributedProcessor:
    """In-process distributed task coordinator used by orchestration services."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def distribute_training_job(self, job_config: dict[str, Any]) -> str:
        """Register a training job and split it into deterministic subtasks."""
        job_id = str(job_config.get("id") or f"train-{uuid.uuid4().hex[:10]}")
        dataset = job_config.get("dataset")
        num_chunks = int(job_config.get("num_chunks") or 4)
        chunks = self._split_dataset(dataset, num_chunks=max(num_chunks, 1))

        subtask_ids = [f"{job_id}-chunk-{idx}" for idx, _ in enumerate(chunks)]
        metadata = {
            "job_id": job_id,
            "status": "distributed",
            "dataset": dataset,
            "chunks": chunks,
            "subtasks": {subtask_id: {"status": "queued", "result": None} for subtask_id in subtask_ids},
        }
        async with self._lock:
            self._tasks[job_id] = metadata
        return job_id

    async def mark_subtask_complete(self, job_id: str, subtask_id: str, result: dict[str, Any]) -> None:
        """Mark a subtask complete with a partial result payload."""
        async with self._lock:
            task = self._tasks.get(job_id)
            if task is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            subtask = task["subtasks"].get(subtask_id)
            if subtask is None:
                raise KeyError(f"Unknown subtask_id: {subtask_id}")
            subtask["status"] = "completed"
            subtask["result"] = result

    async def aggregate_results(self, task_id: str) -> dict[str, Any]:
        """Aggregate completed subtask results and compute progress."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"Unknown task_id: {task_id}")
            subtasks = task["subtasks"]
            completed = [value["result"] for value in subtasks.values() if value["status"] == "completed"]
            progress = len(completed) / len(subtasks) if subtasks else 1.0
            status = "completed" if progress >= 1.0 else "running"
            summary = self._aggregate_models(completed)
            return {
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "subtask_results": completed,
                "model_summary": summary,
            }

    def _split_dataset(self, dataset: Any, num_chunks: int = 4) -> list[str]:
        """Build logical chunk ids for list- or path-like dataset references."""
        if isinstance(dataset, list):
            if not dataset:
                return ["chunk-0"]
            chunk_size = max(1, len(dataset) // num_chunks)
            chunks = []
            for start in range(0, len(dataset), chunk_size):
                stop = min(start + chunk_size, len(dataset))
                chunks.append(f"rows[{start}:{stop}]")
            return chunks
        if isinstance(dataset, str) and dataset:
            return [f"{dataset}::part-{i}" for i in range(num_chunks)]
        return [f"synthetic::part-{i}" for i in range(num_chunks)]

    def _aggregate_models(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate simple numeric model metrics from subtask results."""
        if not results:
            return {"num_results": 0}
        losses = [float(item["loss"]) for item in results if isinstance(item, dict) and "loss" in item]
        accuracies = [float(item["accuracy"]) for item in results if isinstance(item, dict) and "accuracy" in item]
        return {
            "num_results": len(results),
            "avg_loss": sum(losses) / len(losses) if losses else None,
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        }


__all__ = ["DistributedProcessor"]
