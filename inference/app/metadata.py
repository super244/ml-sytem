from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ai_factory.core.datasets import list_sample_prompts, load_catalog, load_dataset_provenance
from ai_factory.core.discovery import list_training_runs, load_benchmark_registry
from inference.app.config import AppSettings

_START_TIME = time.monotonic()


class MetadataService:
    def __init__(
        self,
        settings: AppSettings,
        models_catalog: list[dict[str, Any]],
        prompt_presets: dict[str, Any],
        cache: Any,
        instance_service: Any = None,
        start_time: float | None = None,
    ):
        self.settings = settings
        self.models_catalog = models_catalog
        self.prompt_presets = prompt_presets
        self.cache = cache
        self.instance_service = instance_service
        self.start_time = start_time or _START_TIME

    def dataset_dashboard(self) -> dict[str, Any]:
        repo_root = self.settings.repo_root
        catalog = load_catalog(Path(repo_root) / "data" / "catalog.json", repo_root=repo_root)
        provenance = load_dataset_provenance(
            repo_root=repo_root,
            processed_manifest_path=Path(repo_root) / "data" / "processed" / "manifest.json",
            pack_summary_path=Path(repo_root) / "data" / "processed" / "pack_summary.json",
        )
        catalog["packs"] = provenance["pack_summary"].get("packs", [])
        catalog["provenance"] = provenance
        return catalog

    def prompt_library(self, limit: int = 12) -> dict[str, Any]:
        repo_root = self.settings.repo_root
        presets = [
            {
                "id": preset.id,
                "title": preset.title,
                "description": preset.description,
                "style_instructions": preset.style_instructions,
            }
            for preset in self.prompt_presets.values()
        ]
        return {
            "presets": presets,
            "examples": list_sample_prompts(
                limit=limit,
                path=Path(repo_root) / "data" / "catalog.json",
                repo_root=repo_root,
            ),
        }

    def benchmark_library(self) -> list[dict[str, Any]]:
        return load_benchmark_registry(self.settings.benchmark_registry_path)

    def runs(self) -> list[dict[str, Any]]:
        return list_training_runs(self.settings.artifacts_dir)

    def models(self) -> list[dict[str, Any]]:
        return self.models_catalog

    def _instance_counts(self) -> tuple[int, int]:
        try:
            from inference.app.dependencies import get_instance_service

            svc = get_instance_service()
            all_instances = svc.store.list_instances()
            total = len(all_instances)
            running = sum(1 for i in all_instances if getattr(i, "status", "") == "running")
            return total, running
        except Exception:
            return 0, 0

    def status(self) -> dict[str, Any]:
        instance_count, running_count = self._instance_counts()
        return {
            "title": self.settings.title,
            "version": self.settings.version,
            "models": self.models(),
            "cache": self.cache.stats() if self.cache is not None else {"enabled": False},
            "benchmarks": len(self.benchmark_library()),
            "runs": len(self.runs()),
            "instance_count": instance_count,
            "running_count": running_count,
            "uptime_seconds": round(time.monotonic() - self.start_time, 1),
        }
