from __future__ import annotations

import os
import tempfile
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


def resolve_path_from_config(config_path: str | None, ref: str | None) -> str | None:
    if not ref:
        return None
    if not config_path:
        return str(Path(ref).resolve()) if Path(ref).is_absolute() else ref
    return _resolve_ref(Path(config_path).resolve(), ref)


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


def apply_experience_guardrails(config: OrchestrationConfig) -> OrchestrationConfig:
    merged = config.model_dump(mode="json")
    experience = config.experience
    notes: list[str] = list(merged.setdefault("metadata", {}).get("guardrail_notes", []))

    if not experience.allow_command_override and merged["subsystem"].get("command_override"):
        merged["subsystem"]["command_override"] = None
        notes.append("command_override removed by user-level guardrails")

    if not experience.allow_extra_args and merged["subsystem"].get("extra_args"):
        merged["subsystem"]["extra_args"] = []
        notes.append("extra_args removed by user-level guardrails")

    if not experience.allow_remote_shell and merged["instance"].get("environment", {}).get("kind") == "cloud":
        merged["instance"]["environment"] = {"kind": "local"}
        merged["remote_access"] = {"enable_ssh": False, "sync_before_start": False, "sync_mode": "none"}
        merged["orchestration_mode"] = "single"
        notes.append("cloud execution downgraded to local by user-level guardrails")

    if experience.require_safe_defaults and config.instance.type in {"train", "finetune"}:
        current = int((merged.get("sub_agents") or {}).get("max_parallelism") or 1)
        safe_limit = int(experience.max_parallel_sub_agents or 1)
        if current > safe_limit:
            merged["sub_agents"]["max_parallelism"] = safe_limit
            notes.append(f"sub-agent parallelism capped at {safe_limit}")

    if experience.require_safe_defaults and config.instance.type == "deploy":
        merged.setdefault("subsystem", {}).setdefault("provider_options", {})
        if not merged["subsystem"]["provider_options"].get("dry_run", False):
            merged["subsystem"]["provider_options"]["dry_run"] = True
            notes.append("deployment switched to dry_run by user-level guardrails")

    merged["metadata"]["guardrail_notes"] = notes
    return build_orchestration_config(merged, config_path=config.config_path)


def _cloud_profile_store_path() -> Path:
    override = os.getenv("AI_FACTORY_CLOUD_PROFILES_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".ai-factory" / "cloud_profiles.json").resolve()


def _secure_path_permissions(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass


def _ensure_profile_store_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _secure_path_permissions(path.parent, 0o700)


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    _ensure_profile_store_parent(path)
    serialized = yaml.safe_dump(payload, sort_keys=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        tmp_path = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _secure_path_permissions(tmp_path, 0o600)
        tmp_path.replace(path)
        _secure_path_permissions(path, 0o600)
    finally:
        if tmp_path.exists() and tmp_path != path:
            try:
                tmp_path.unlink()
            except OSError:
                pass


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
    _write_yaml_atomic(path, payload)
    return path
