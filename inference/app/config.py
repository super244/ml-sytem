from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AppSettings:
    title: str
    version: str
    cors_origins: list[str]
    model_registry_path: str
    prompt_library_path: str
    benchmark_registry_path: str
    artifacts_dir: str
    cache_dir: str
    telemetry_path: str
    cache_enabled: bool
    telemetry_enabled: bool


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no"}


def get_settings() -> AppSettings:
    cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]
    return AppSettings(
        title="Atlas Math Lab API",
        version="0.2.0",
        cors_origins=cors_origins,
        model_registry_path=os.getenv("MODEL_REGISTRY_PATH", "inference/configs/model_registry.yaml"),
        prompt_library_path=os.getenv("PROMPT_LIBRARY_PATH", "inference/configs/prompt_presets.yaml"),
        benchmark_registry_path=os.getenv("BENCHMARK_REGISTRY_PATH", "evaluation/benchmarks/registry.yaml"),
        artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
        cache_dir=os.getenv("INFERENCE_CACHE_DIR", "artifacts/inference/cache"),
        telemetry_path=os.getenv("INFERENCE_TELEMETRY_PATH", "artifacts/inference/telemetry/requests.jsonl"),
        cache_enabled=_env_flag("INFERENCE_CACHE_ENABLED", "1"),
        telemetry_enabled=_env_flag("INFERENCE_TELEMETRY_ENABLED", "1"),
    )
