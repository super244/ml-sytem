from __future__ import annotations

from typing import Any

from ai_factory.core.discovery import list_training_runs, load_benchmark_registry
from data.catalog import list_sample_prompts, load_catalog, load_pack_summary
from inference.app.config import AppSettings


class MetadataService:
    def __init__(
        self,
        settings: AppSettings,
        models_catalog: list[dict[str, Any]],
        prompt_presets: dict[str, Any],
        cache: Any,
    ):
        self.settings = settings
        self.models_catalog = models_catalog
        self.prompt_presets = prompt_presets
        self.cache = cache

    def dataset_dashboard(self) -> dict[str, Any]:
        catalog = load_catalog()
        catalog["packs"] = load_pack_summary().get("packs", [])
        return catalog

    def prompt_library(self, limit: int = 12) -> dict[str, Any]:
        presets = [
            {
                "id": preset.id,
                "title": preset.title,
                "description": preset.description,
                "style_instructions": preset.style_instructions,
            }
            for preset in self.prompt_presets.values()
        ]
        return {"presets": presets, "examples": list_sample_prompts(limit=limit)}

    def benchmark_library(self) -> list[dict[str, Any]]:
        return load_benchmark_registry(self.settings.benchmark_registry_path)

    def runs(self) -> list[dict[str, Any]]:
        return list_training_runs(self.settings.artifacts_dir)

    def models(self) -> list[dict[str, Any]]:
        return self.models_catalog

    def status(self) -> dict[str, Any]:
        return {
            "title": self.settings.title,
            "version": self.settings.version,
            "models": self.models(),
            "cache": self.cache.stats() if self.cache is not None else {"enabled": False},
            "benchmarks": len(self.benchmark_library()),
            "runs": len(self.runs()),
        }
