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
        self._cached_status = None
        self._status_cache_time = 0
        self._cache_ttl = 5.0  # Cache status for 5 seconds
        self._dataset_dashboard_cache: dict[str, Any] | None = None
        self._prompt_library_cache: dict[int, dict[str, Any]] = {}
        self._benchmark_library_cache: list[dict[str, Any]] | None = None
        self._runs_cache: list[dict[str, Any]] | None = None

    def dataset_dashboard(self) -> dict[str, Any]:
        """Get dataset dashboard with error handling and caching."""
        if self._dataset_dashboard_cache is not None:
            return self._dataset_dashboard_cache
        try:
            repo_root = self.settings.repo_root
            catalog = load_catalog(Path(repo_root) / "data" / "catalog.json", repo_root=repo_root)
            provenance = load_dataset_provenance(
                repo_root=repo_root,
                processed_manifest_path=Path(repo_root) / "data" / "processed" / "manifest.json",
                pack_summary_path=Path(repo_root) / "data" / "processed" / "pack_summary.json",
            )
            catalog["packs"] = provenance["pack_summary"].get("packs", [])
            catalog["provenance"] = provenance
            self._dataset_dashboard_cache = catalog
        except Exception:
            self._dataset_dashboard_cache = {
                "generated_at": None,
                "summary": {"num_datasets": 0, "custom_datasets": 0, "public_datasets": 0},
                "datasets": [],
                "packs": [],
                "provenance": None,
            }
        return self._dataset_dashboard_cache

    def prompt_library(self, limit: int = 12) -> dict[str, Any]:
        """Get prompt library with error handling."""
        cached = self._prompt_library_cache.get(limit)
        if cached is not None:
            return cached
        try:
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
        except Exception:
            result = {"presets": [], "examples": []}
        else:
            result = {
                "presets": presets,
                "examples": list_sample_prompts(
                    limit=limit,
                    path=Path(repo_root) / "data" / "catalog.json",
                    repo_root=repo_root,
                ),
            }
        self._prompt_library_cache[limit] = result
        return result

    def benchmark_library(self) -> list[dict[str, Any]]:
        """Get benchmark library with error handling."""
        if self._benchmark_library_cache is not None:
            return self._benchmark_library_cache
        try:
            self._benchmark_library_cache = load_benchmark_registry(self.settings.benchmark_registry_path)
        except Exception:
            self._benchmark_library_cache = []
        return self._benchmark_library_cache

    def runs(self) -> list[dict[str, Any]]:
        """Get training runs with error handling."""
        if self._runs_cache is not None:
            return self._runs_cache
        try:
            self._runs_cache = list_training_runs(self.settings.artifacts_dir)
        except Exception:
            self._runs_cache = []
        return self._runs_cache

    def models(self) -> list[dict[str, Any]]:
        """Get models catalog."""
        return self.models_catalog

    def _instance_counts(self) -> tuple[int, int]:
        """Get instance counts with error handling."""
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
        """Get status with caching to avoid expensive operations."""
        current_time = time.monotonic()

        # Return cached status if still valid
        if self._cached_status and current_time - self._status_cache_time < self._cache_ttl:
            return self._cached_status

        try:
            instance_count, running_count = self._instance_counts()
            status_data = {
                "title": self.settings.title,
                "version": self.settings.version,
                "models": self.models(),
                "cache": self.cache.stats() if self.cache is not None else {"enabled": False},
                "benchmarks": len(self.benchmark_library()),
                "runs": len(self.runs()),
                "instance_count": instance_count,
                "running_count": running_count,
                "uptime_seconds": round(current_time - self.start_time, 1),
            }

            # Cache the result
            self._cached_status = status_data
            self._status_cache_time = current_time

            return status_data
        except Exception as exc:
            # Return minimal status on error
            return {
                "title": self.settings.title,
                "version": self.settings.version,
                "models": [],
                "cache": {"enabled": False, "error": str(exc)},
                "benchmarks": 0,
                "runs": 0,
                "instance_count": 0,
                "running_count": 0,
                "uptime_seconds": round(current_time - self.start_time, 1),
                "error": "Some components failed to load",
            }
