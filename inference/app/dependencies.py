from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from inference.app.cache import FileResponseCache
from inference.app.config import AppSettings, get_settings
from inference.app.metadata import MetadataService
from inference.app.telemetry import JsonlTelemetryLogger

if TYPE_CHECKING:
    from inference.app.services.generation_service import GenerationService
    from inference.app.services.instance_service import InstanceService


def get_app_settings() -> AppSettings:
    return get_settings()


@lru_cache(maxsize=1)
def get_metadata_service() -> MetadataService:
    settings = get_settings()
    from inference.app.model_catalog import list_model_catalog
    from inference.app.prompts import load_prompt_presets

    model_catalog = list_model_catalog(settings.model_registry_path)
    presets = load_prompt_presets(settings.prompt_library_path)
    cache = FileResponseCache(settings.cache_dir) if settings.cache_enabled else None
    return MetadataService(settings, model_catalog, presets, cache)


@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    settings = get_settings()
    from inference.app.generation import MathGenerator
    from inference.app.model_loader import MathModelRegistry, load_registry_from_yaml
    from inference.app.prompts import load_prompt_presets
    from inference.app.services.generation_service import GenerationService

    registry = MathModelRegistry(load_registry_from_yaml(settings.model_registry_path))
    presets = load_prompt_presets(settings.prompt_library_path)
    cache = FileResponseCache(settings.cache_dir) if settings.cache_enabled else None
    telemetry = (
        JsonlTelemetryLogger(settings.telemetry_path)
        if settings.telemetry_enabled
        else None
    )
    generator = MathGenerator(registry, prompt_presets=presets, cache=cache, telemetry=telemetry)
    return GenerationService(generator, settings)


@lru_cache(maxsize=1)
def get_instance_service() -> InstanceService:
    settings = get_settings()
    from inference.app.services.instance_service import InstanceService

    return InstanceService(settings)
