from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ai_factory.core.config.schema import OrchestrationConfig
from ai_factory.core.instances.models import EnvironmentSpec


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _resolve_ref(config_path: Path, ref: str | None) -> str | None:
    if not ref:
        return None
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return str(ref_path)
    return str((config_path.parent / ref_path).resolve())


def _coerce_environment(raw: dict[str, Any]) -> dict[str, Any]:
    environment = raw.get("environment")
    if isinstance(environment, str):
        raw["environment"] = {"kind": environment}
    return raw


def build_orchestration_config(payload: dict[str, Any], *, config_path: str | None = None) -> OrchestrationConfig:
    instance_payload = _coerce_environment(dict(payload.get("instance", {})))
    raw = dict(payload)
    raw["instance"] = instance_payload
    config = OrchestrationConfig.model_validate(raw)
    config.config_path = config_path
    return config


def load_orchestration_config(path: str) -> OrchestrationConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    config = build_orchestration_config(raw, config_path=str(config_path))
    resolved_ref = _resolve_ref(config_path, config.subsystem.config_ref)
    config.resolved_subsystem_config_path = resolved_ref
    if resolved_ref and Path(resolved_ref).exists():
        config.resolved_subsystem_config = _load_yaml(Path(resolved_ref))
    return config


def apply_environment_override(
    config: OrchestrationConfig,
    override: EnvironmentSpec | None,
) -> OrchestrationConfig:
    if override is None:
        return config
    merged = config.model_dump(mode="json")
    current = merged["instance"].get("environment", {})
    for key, value in override.model_dump(mode="json").items():
        if value is not None and value != {}:
            current[key] = value
    merged["instance"]["environment"] = current
    return build_orchestration_config(merged, config_path=config.config_path)


def _cloud_profile_store_path() -> Path:
    override = os.getenv("AI_FACTORY_CLOUD_PROFILES_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ai-factory" / "cloud_profiles.json"


def load_cloud_profile(name: str) -> EnvironmentSpec | None:
    path = _cloud_profile_store_path()
    if not path.exists():
        return None
    payload = _load_yaml(path)
    profile = (payload.get("profiles") or {}).get(name)
    if not profile:
        return None
    return EnvironmentSpec.model_validate({"kind": "cloud", "profile_name": name, **profile})


def save_cloud_profile(name: str, environment: EnvironmentSpec) -> Path:
    path = _cloud_profile_store_path()
    payload = _load_yaml(path) if path.exists() else {}
    payload.setdefault("profiles", {})
    payload["profiles"][name] = {
        key: value
        for key, value in environment.model_dump(mode="json").items()
        if key != "kind" and value not in (None, {})
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path
