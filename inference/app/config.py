from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ai_factory.version import VERSION


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
    demo_mode: bool = False
    repo_root: str = field(default_factory=lambda: str(Path(__file__).resolve().parents[2]))
    openai_api_keys: list[str] = field(default_factory=list)
    openai_rate_limit_requests_per_minute: int = 0
    openai_rate_limit_window_seconds: int = 60


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no"}


def _resolve_repo_path(value: str, *, repo_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def get_settings() -> AppSettings:
    cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]
    repo_root = Path(os.getenv("AI_FACTORY_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
    artifacts_dir = _resolve_repo_path(os.getenv("ARTIFACTS_DIR", "artifacts"), repo_root=repo_root)
    cache_dir = _resolve_repo_path(
        os.getenv("INFERENCE_CACHE_DIR", str(Path(artifacts_dir) / "inference" / "cache")),
        repo_root=repo_root,
    )
    telemetry_path = _resolve_repo_path(
        os.getenv(
            "INFERENCE_TELEMETRY_PATH",
            str(Path(artifacts_dir) / "inference" / "telemetry" / "requests.jsonl"),
        ),
        repo_root=repo_root,
    )
    return AppSettings(
        title="AI-Factory API",
        version=os.getenv("AI_FACTORY_API_VERSION", VERSION),
        repo_root=str(repo_root),
        cors_origins=cors_origins,
        model_registry_path=_resolve_repo_path(
            os.getenv("MODEL_REGISTRY_PATH", "inference/configs/model_registry.yaml"),
            repo_root=repo_root,
        ),
        prompt_library_path=_resolve_repo_path(
            os.getenv("PROMPT_LIBRARY_PATH", "inference/configs/prompt_presets.yaml"),
            repo_root=repo_root,
        ),
        benchmark_registry_path=_resolve_repo_path(
            os.getenv("BENCHMARK_REGISTRY_PATH", "evaluation/benchmarks/registry.yaml"),
            repo_root=repo_root,
        ),
        artifacts_dir=artifacts_dir,
        cache_dir=cache_dir,
        telemetry_path=telemetry_path,
        cache_enabled=_env_flag("INFERENCE_CACHE_ENABLED", "1"),
        telemetry_enabled=_env_flag("INFERENCE_TELEMETRY_ENABLED", "1"),
        demo_mode=_env_flag("AI_FACTORY_DEMO_MODE", "0"),
        openai_api_keys=[key.strip() for key in os.getenv("OPENAI_API_KEYS", "").split(",") if key.strip()],
        openai_rate_limit_requests_per_minute=int(os.getenv("OPENAI_RATE_LIMIT_REQUESTS_PER_MINUTE", "0")),
        openai_rate_limit_window_seconds=int(os.getenv("OPENAI_RATE_LIMIT_WINDOW_SECONDS", "60")),
    )
