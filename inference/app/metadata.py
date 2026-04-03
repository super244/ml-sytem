from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ai_factory.core.datasets import list_sample_prompts, load_catalog, load_dataset_provenance
from ai_factory.core.discovery import list_training_runs, load_benchmark_registry
from inference.app.config import AppSettings
from inference.app.model_catalog import list_model_catalog
from inference.app.prompts import PromptPreset, load_prompt_presets

_START_TIME = time.monotonic()


class MetadataService:
    def __init__(
        self,
        settings: AppSettings,
        models_catalog: list[dict[str, Any]],
        prompt_presets: dict[str, PromptPreset],
        cache: Any,
        instance_service: Any = None,
        start_time: float | None = None,
    ) -> None:
        self.settings = settings
        self.models_catalog = models_catalog
        self.prompt_presets = prompt_presets
        self.cache = cache
        self.instance_service = instance_service
        self.start_time = start_time or _START_TIME
        self._models_catalog = models_catalog
        self._prompt_presets = prompt_presets
        self._cached_status: dict[str, Any] | None = None
        self._status_cache_time = 0.0
        self._cache_ttl = 5.0
        self._dataset_dashboard_cache: dict[str, Any] | None = None
        self._dataset_dashboard_cache_time = 0.0
        self._prompt_library_cache: dict[int, dict[str, Any]] = {}
        self._prompt_library_cache_time: dict[int, float] = {}
        self._benchmark_library_cache: list[dict[str, Any]] | None = None
        self._benchmark_library_cache_time = 0.0
        self._runs_cache: list[dict[str, Any]] | None = None
        self._runs_cache_time = 0.0
        self._models_cache_time = 0.0
        self._prompt_presets_cache_time = 0.0
        self._models_error: str | None = None
        self._prompt_presets_error: str | None = None
        self._benchmark_error: str | None = None
        self._runs_error: str | None = None

    def _cache_valid(self, cache_time: float) -> bool:
        return time.monotonic() - cache_time < self._cache_ttl

    def _refresh_models_catalog(self) -> tuple[list[dict[str, Any]], list[str]]:
        if self._cache_valid(self._models_cache_time):
            return self._models_catalog, [self._models_error] if self._models_error else []
        try:
            self._models_catalog = list_model_catalog(self.settings.model_registry_path)
            self._models_error = None
        except Exception as exc:
            self._models_catalog = []
            self._models_error = f"model registry: {exc}"
        self._models_cache_time = time.monotonic()
        return self._models_catalog, [self._models_error] if self._models_error else []

    def _refresh_prompt_presets(self) -> tuple[dict[str, PromptPreset], list[str]]:
        if self._cache_valid(self._prompt_presets_cache_time):
            return self._prompt_presets, [self._prompt_presets_error] if self._prompt_presets_error else []
        try:
            self._prompt_presets = load_prompt_presets(self.settings.prompt_library_path)
            self._prompt_presets_error = None
        except Exception as exc:
            self._prompt_presets = {}
            self._prompt_presets_error = f"prompt library: {exc}"
        self._prompt_presets_cache_time = time.monotonic()
        return self._prompt_presets, [self._prompt_presets_error] if self._prompt_presets_error else []

    def dataset_dashboard(self) -> dict[str, Any]:
        """Get dataset dashboard with explicit degraded-state reporting."""
        if self._dataset_dashboard_cache is not None and self._cache_valid(self._dataset_dashboard_cache_time):
            return self._dataset_dashboard_cache
        errors: list[str] = []
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
            catalog["status"] = "available"
            catalog["errors"] = []
            self._dataset_dashboard_cache = catalog
        except Exception as exc:
            errors.append(f"dataset dashboard: {exc}")
            self._dataset_dashboard_cache = {
                "status": "degraded",
                "errors": errors,
                "generated_at": None,
                "summary": {"num_datasets": 0, "custom_datasets": 0, "public_datasets": 0},
                "datasets": [],
                "packs": [],
                "provenance": None,
            }
        self._dataset_dashboard_cache_time = time.monotonic()
        return self._dataset_dashboard_cache

    def prompt_library(self, limit: int = 12) -> dict[str, Any]:
        """Get prompt library with explicit degraded-state reporting."""
        cached = self._prompt_library_cache.get(limit)
        if cached is not None and self._cache_valid(self._prompt_library_cache_time.get(limit, 0.0)):
            return cached
        repo_root = self.settings.repo_root
        prompts, errors = self._refresh_prompt_presets()
        try:
            presets = [
                {
                    "id": preset.id,
                    "title": preset.title,
                    "description": preset.description,
                    "style_instructions": preset.style_instructions,
                }
                for preset in prompts.values()
            ]
        except Exception as exc:
            errors.append(f"prompt serialization: {exc}")
            result = {"status": "degraded", "errors": errors, "presets": [], "examples": []}
        else:
            result = {
                "status": "available" if not errors else "degraded",
                "errors": errors,
                "presets": presets,
                "examples": list_sample_prompts(
                    limit=limit,
                    path=Path(repo_root) / "data" / "catalog.json",
                    repo_root=repo_root,
                ),
            }
        self._prompt_library_cache[limit] = result
        self._prompt_library_cache_time[limit] = time.monotonic()
        return result

    def benchmark_library(self) -> list[dict[str, Any]]:
        """Get benchmark library with refreshable caching."""
        if self._benchmark_library_cache is not None and self._cache_valid(self._benchmark_library_cache_time):
            return self._benchmark_library_cache
        try:
            self._benchmark_library_cache = load_benchmark_registry(self.settings.benchmark_registry_path)
            self._benchmark_error = None
        except Exception as exc:
            self._benchmark_library_cache = []
            self._benchmark_error = f"benchmark registry: {exc}"
        self._benchmark_library_cache_time = time.monotonic()
        return self._benchmark_library_cache

    def runs(self) -> list[dict[str, Any]]:
        """Get training runs with refreshable caching."""
        if self._runs_cache is not None and self._cache_valid(self._runs_cache_time):
            return self._runs_cache
        try:
            self._runs_cache = list_training_runs(self.settings.artifacts_dir)
            self._runs_error = None
        except Exception as exc:
            self._runs_cache = []
            self._runs_error = f"training runs: {exc}"
        self._runs_cache_time = time.monotonic()
        return self._runs_cache

    def models(self) -> list[dict[str, Any]]:
        """Get a live view of the model catalog."""
        models, _ = self._refresh_models_catalog()
        return models

    def _instance_counts(self) -> tuple[int, int, list[str]]:
        """Get instance counts and explicit load errors."""
        try:
            from inference.app.dependencies import get_instance_service

            svc = get_instance_service()
            all_instances = svc.store.list_instances()
            total = len(all_instances)
            running = sum(1 for i in all_instances if getattr(i, "status", "") == "running")
            return total, running, []
        except Exception as exc:
            return 0, 0, [f"instances: {exc}"]

    def status(self) -> dict[str, Any]:
        """Get status with caching to avoid expensive operations."""
        current_time = time.monotonic()

        # Return cached status if still valid
        if self._cached_status and current_time - self._status_cache_time < self._cache_ttl:
            return self._cached_status

        models, model_errors = self._refresh_models_catalog()
        _, prompt_errors = self._refresh_prompt_presets()
        benchmarks = self.benchmark_library()
        runs = self.runs()
        instance_count, running_count, instance_errors = self._instance_counts()
        cache_stats = self.cache.stats() if self.cache is not None else {"enabled": False}
        errors = [
            *model_errors,
            *prompt_errors,
            *( [self._benchmark_error] if self._benchmark_error else [] ),
            *( [self._runs_error] if self._runs_error else [] ),
            *instance_errors,
        ]
        status_data = {
            "title": self.settings.title,
            "version": self.settings.version,
            "status": "available" if not errors else "degraded",
            "errors": errors,
            "models": models,
            "cache": cache_stats,
            "benchmarks": len(benchmarks),
            "runs": len(runs),
            "instance_count": instance_count,
            "running_count": running_count,
            "uptime_seconds": round(current_time - self.start_time, 1),
        }
        self._cached_status = status_data
        self._status_cache_time = current_time
        return status_data
