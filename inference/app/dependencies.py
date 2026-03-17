from __future__ import annotations

from functools import lru_cache

from inference.app.cache import FileResponseCache
from inference.app.config import AppSettings, get_settings
from inference.app.generation import MathGenerator
from inference.app.metadata import MetadataService
from inference.app.model_loader import MathModelRegistry, load_registry_from_yaml
from inference.app.prompts import load_prompt_presets
from inference.app.services.generation_service import GenerationService
from inference.app.telemetry import JsonlTelemetryLogger


@lru_cache(maxsize=1)
def get_app_services() -> dict[str, object]:
    settings = get_settings()
    registry = MathModelRegistry(load_registry_from_yaml(settings.model_registry_path))
    presets = load_prompt_presets(settings.prompt_library_path)
    cache = FileResponseCache(settings.cache_dir) if settings.cache_enabled else None
    telemetry = JsonlTelemetryLogger(settings.telemetry_path) if settings.telemetry_enabled else None
    generator = MathGenerator(registry, prompt_presets=presets, cache=cache, telemetry=telemetry)
    return {
        "settings": settings,
        "generation_service": GenerationService(generator, settings),
        "metadata_service": MetadataService(settings, registry, presets, cache),
    }


def get_app_settings() -> AppSettings:
    return get_app_services()["settings"]  # type: ignore[return-value]


def get_generation_service() -> GenerationService:
    return get_app_services()["generation_service"]  # type: ignore[return-value]


def get_metadata_service() -> MetadataService:
    return get_app_services()["metadata_service"]  # type: ignore[return-value]
