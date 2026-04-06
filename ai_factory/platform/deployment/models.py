from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_factory.core.schemas import DeploymentSpec, ModelArtifact


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}, ()):
            return value
    return None


def _coerce_float(value: Any, default: float, *, strict: bool = False, field_name: str = "value") -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError) as exc:
        if strict:
            raise ValueError(f"{field_name} must be numeric") from exc
        return float(default)


def _coerce_int(value: Any, default: int | None, *, strict: bool = False, field_name: str = "value") -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        if strict:
            raise ValueError(f"{field_name} must be an integer") from exc
        return default


class DeploymentRolloutStage(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    traffic_percent: float = Field(default=100.0, ge=0.0, le=100.0)
    duration_minutes: int | None = None
    pause_after_stage: bool = False
    health_check_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentVersionSummary(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    version: str
    architecture: str
    format: str
    parameter_count: int
    parameter_size_b: float
    source_path: str
    release_label: str
    summary_text: str
    lineage: dict[str, Any] = Field(default_factory=dict)
    changelog: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentRollbackReadiness(BaseModel):
    model_config = ConfigDict(strict=True)

    ready: bool = False
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    strategy: str | None = None
    previous_version: str | None = None
    fallback_artifact_path: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class DeploymentManifest(BaseModel):
    model_config = ConfigDict(strict=True)

    deployment_id: str
    created_at: str = Field(default_factory=_utc_now)
    target: str
    target_name: str
    status: str
    model_name: str
    public: bool = False
    spec: dict[str, Any] = Field(default_factory=dict)
    target_capabilities: list[str] = Field(default_factory=list)
    version_summary: DeploymentVersionSummary
    rollout_stages: list[DeploymentRolloutStage] = Field(default_factory=list)
    rollback: DeploymentRollbackReadiness = Field(default_factory=DeploymentRollbackReadiness)
    result: dict[str, Any] | None = None
    status_history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "target": self.target,
            "target_name": self.target_name,
            "status": self.status,
            "model_name": self.model_name,
            "version": self.version_summary.version,
            "release_label": self.version_summary.release_label,
            "rollout_stages": [stage.model_dump(mode="json") for stage in self.rollout_stages],
            "rollback_ready": self.rollback.ready,
        }


def summarize_model_version(model: ModelArtifact, spec: DeploymentSpec | None = None) -> DeploymentVersionSummary:
    metadata = _as_dict(model.metadata)
    spec_metadata = _as_dict(spec.metadata) if spec is not None else {}
    lineage = _as_dict(metadata.get("lineage"))
    if not lineage and isinstance(spec_metadata.get("lineage"), dict):
        lineage = dict(spec_metadata["lineage"])

    source_model = _first_non_empty(
        lineage.get("source_model"),
        metadata.get("base_model"),
        metadata.get("source_model"),
        spec_metadata.get("base_model"),
    )
    training_run_ids = _as_list(
        _first_non_empty(
            lineage.get("training_run_ids"),
            metadata.get("training_run_ids"),
            metadata.get("run_ids"),
            spec_metadata.get("training_run_ids"),
        )
    )
    dataset_ids = _as_list(
        _first_non_empty(
            lineage.get("training_dataset_ids"),
            metadata.get("dataset_ids"),
            metadata.get("training_dataset_ids"),
            spec_metadata.get("dataset_ids"),
        )
    )
    changelog = _as_list(_first_non_empty(metadata.get("changelog"), spec_metadata.get("changelog")))
    tags = []
    for tag in _as_list(_first_non_empty(metadata.get("tags"), spec_metadata.get("tags"))):
        cleaned = str(tag).strip().lower()
        if cleaned and cleaned not in tags:
            tags.append(cleaned)

    release_label = str(
        _first_non_empty(
            metadata.get("release_label"),
            spec_metadata.get("release_label"),
            f"{model.name}:{model.version}",
        )
    )
    parameter_size_b = round(model.parameters / 1_000_000_000, 4)
    summary_bits = [f"{model.name}@{model.version}", model.architecture, f"{parameter_size_b:.4f}B"]
    if source_model:
        summary_bits.append(f"base={source_model}")
    if training_run_ids:
        summary_bits.append(f"runs={','.join(str(item) for item in training_run_ids[:3])}")
    if dataset_ids:
        summary_bits.append(f"datasets={len(dataset_ids)}")
    if metadata.get("artifact_sha256"):
        summary_bits.append(f"sha256={metadata['artifact_sha256']}")

    lineage_summary = {
        "source_model": source_model,
        "training_run_ids": training_run_ids,
        "training_dataset_ids": dataset_ids,
        "artifact_sha256": _first_non_empty(metadata.get("artifact_sha256"), spec_metadata.get("artifact_sha256")),
        "config_path": _first_non_empty(metadata.get("config_path"), spec_metadata.get("config_path")),
        "commit_sha": _first_non_empty(
            metadata.get("commit_sha"), metadata.get("git_sha"), spec_metadata.get("git_sha")
        ),
        "artifact_path": model.path,
        "lineage": lineage,
    }

    return DeploymentVersionSummary(
        name=model.name,
        version=model.version,
        architecture=model.architecture,
        format=model.format,
        parameter_count=model.parameters,
        parameter_size_b=parameter_size_b,
        source_path=model.path,
        release_label=release_label,
        summary_text=" | ".join(summary_bits),
        lineage=lineage_summary,
        changelog=[str(item) for item in changelog],
        tags=tags,
        metadata={
            "domain": model.domain,
            "base_model": source_model,
            "public": bool(spec.public) if spec is not None else False,
            "model_metadata": metadata,
        },
    )


def _rollout_config(spec: DeploymentSpec) -> dict[str, Any]:
    rollout = _as_dict(spec.metadata.get("rollout"))
    if rollout:
        return rollout
    rollout = _as_dict(spec.config.get("rollout"))
    if rollout:
        return rollout
    if "rollout" in spec.metadata or "rollout" in spec.config:
        return rollout
    return {}


def _rollback_config(spec: DeploymentSpec) -> dict[str, Any]:
    rollback = _as_dict(spec.metadata.get("rollback"))
    if rollback:
        return rollback
    rollback = _as_dict(spec.config.get("rollback"))
    if rollback:
        return rollback
    if "rollback" in spec.metadata or "rollback" in spec.config:
        return rollback
    return {}


def build_rollout_stages(spec: DeploymentSpec, *, strict: bool = False) -> list[DeploymentRolloutStage]:
    rollout = _rollout_config(spec)
    raw_stages = rollout.get("stages")
    stages: list[DeploymentRolloutStage] = []

    if isinstance(raw_stages, list) and raw_stages:
        for index, stage in enumerate(raw_stages):
            payload = _as_dict(stage)
            stages.append(
                DeploymentRolloutStage(
                    name=str(payload.get("name") or f"stage-{index + 1}"),
                    traffic_percent=_coerce_float(
                        payload.get("traffic_percent", payload.get("traffic")),
                        100.0,
                        strict=strict,
                        field_name=f"rollout stage {index + 1} traffic_percent",
                    ),
                    duration_minutes=_coerce_int(
                        payload.get("duration_minutes"),
                        None,
                        strict=strict,
                        field_name=f"rollout stage {index + 1} duration_minutes",
                    ),
                    pause_after_stage=bool(payload.get("pause_after_stage", False)),
                    health_check_path=payload.get("health_check_path"),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k
                        not in {
                            "name",
                            "traffic_percent",
                            "traffic",
                            "duration_minutes",
                            "pause_after_stage",
                            "health_check_path",
                        }
                    },
                )
            )
        return stages

    traffic_percent = _coerce_float(
        _first_non_empty(rollout.get("traffic_percent"), rollout.get("traffic"), 100.0),
        100.0,
        strict=strict,
        field_name="rollout traffic_percent",
    )
    stages.append(
        DeploymentRolloutStage(
            name=str(_first_non_empty(rollout.get("name"), "full")),
            traffic_percent=_coerce_float(
                traffic_percent,
                100.0,
                strict=strict,
                field_name="rollout traffic_percent",
            ),
            duration_minutes=_coerce_int(
                rollout.get("duration_minutes"),
                None,
                strict=strict,
                field_name="rollout duration_minutes",
            ),
            pause_after_stage=bool(rollout.get("pause_after_stage", False)),
            health_check_path=rollout.get("health_check_path"),
            metadata={
                k: v
                for k, v in rollout.items()
                if k
                not in {
                    "name",
                    "traffic_percent",
                    "traffic",
                    "duration_minutes",
                    "pause_after_stage",
                    "health_check_path",
                    "stages",
                }
            },
        )
    )
    return stages


def validate_rollout_configuration(spec: DeploymentSpec) -> list[str]:
    errors: list[str] = []
    rollout = _rollout_config(spec)
    raw_stages = rollout.get("stages")
    if raw_stages is not None and not isinstance(raw_stages, list):
        errors.append("rollout.stages must be a list of stage objects.")
    elif isinstance(raw_stages, list):
        for index, stage in enumerate(raw_stages):
            if not isinstance(stage, dict):
                errors.append(f"rollout.stages[{index}] must be an object.")

    try:
        stages = build_rollout_stages(spec, strict=True)
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    errors.extend(validate_rollout_stages(stages))
    return errors


def validate_rollout_stages(stages: list[DeploymentRolloutStage]) -> list[str]:
    errors: list[str] = []
    seen_names: set[str] = set()
    previous_percent = 0.0

    for index, stage in enumerate(stages):
        if not stage.name.strip():
            errors.append(f"Rollout stage {index + 1} has an empty name.")
        if stage.name in seen_names:
            errors.append(f"Rollout stage '{stage.name}' is duplicated.")
        seen_names.add(stage.name)
        if stage.traffic_percent <= 0:
            errors.append(f"Rollout stage '{stage.name}' must target more than 0% traffic.")
        if stage.traffic_percent < previous_percent:
            errors.append(f"Rollout stage '{stage.name}' must not reduce traffic below the prior stage.")
        previous_percent = stage.traffic_percent

    if stages and abs(stages[-1].traffic_percent - 100.0) > 0.01:
        errors.append("The final rollout stage must reach 100% traffic for a complete deployment.")

    return errors


def assess_rollback_readiness(
    model: ModelArtifact,
    spec: DeploymentSpec,
    *,
    rollout_stages: list[DeploymentRolloutStage],
    target_supports_rollback: bool,
) -> DeploymentRollbackReadiness:
    rollout = _rollout_config(spec)
    rollback = _rollback_config(spec)
    metadata = _as_dict(model.metadata)

    previous_version = (
        str(
            _first_non_empty(
                rollback.get("previous_version"),
                rollout.get("previous_version"),
                metadata.get("previous_version"),
                metadata.get("rollback_version"),
                spec.metadata.get("previous_version"),
            )
        )
        if _first_non_empty(
            rollback.get("previous_version"),
            rollout.get("previous_version"),
            metadata.get("previous_version"),
            metadata.get("rollback_version"),
            spec.metadata.get("previous_version"),
        )
        is not None
        else None
    )

    fallback_artifact_path = (
        str(
            _first_non_empty(
                rollback.get("artifact_path"),
                rollback.get("fallback_artifact_path"),
                rollout.get("artifact_path"),
                metadata.get("rollback_artifact_path"),
                metadata.get("previous_artifact_path"),
                spec.metadata.get("rollback_artifact_path"),
            )
        )
        if _first_non_empty(
            rollback.get("artifact_path"),
            rollback.get("fallback_artifact_path"),
            rollout.get("artifact_path"),
            metadata.get("rollback_artifact_path"),
            metadata.get("previous_artifact_path"),
            spec.metadata.get("rollback_artifact_path"),
        )
        is not None
        else None
    )

    strategy = (
        str(
            _first_non_empty(
                rollback.get("strategy"),
                rollout.get("strategy"),
                "previous_version" if previous_version else None,
            )
        )
        if _first_non_empty(
            rollback.get("strategy"), rollout.get("strategy"), "previous_version" if previous_version else None
        )
        is not None
        else None
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if not target_supports_rollback:
        blockers.append("The selected deployment target does not advertise rollback support.")
    if not (previous_version or fallback_artifact_path):
        blockers.append("No rollback baseline is defined in model metadata or deployment configuration.")
    if rollout_stages and rollout_stages[-1].traffic_percent < 100.0:
        warnings.append("The rollout plan does not reach full traffic; rollback coverage may be partial.")
    if len(rollout_stages) > 1 and rollout_stages[0].traffic_percent < 5.0:
        warnings.append("The first rollout stage is very small; monitor closely before promotion.")

    return DeploymentRollbackReadiness(
        ready=len(blockers) == 0,
        blockers=blockers,
        warnings=warnings,
        strategy=strategy,
        previous_version=previous_version,
        fallback_artifact_path=fallback_artifact_path,
        summary={
            "target_supports_rollback": target_supports_rollback,
            "rollout_stage_count": len(rollout_stages),
            "strategy": strategy,
            "previous_version": previous_version,
            "fallback_artifact_path": fallback_artifact_path,
        },
    )


def build_deployment_manifest(
    *,
    deployment_id: str,
    model: ModelArtifact,
    spec: DeploymentSpec,
    target_name: str,
    target_capabilities: list[str],
    target_supports_rollback: bool,
) -> DeploymentManifest:
    rollout_stages = build_rollout_stages(spec)
    version_summary = summarize_model_version(model, spec)
    rollback = assess_rollback_readiness(
        model,
        spec,
        rollout_stages=rollout_stages,
        target_supports_rollback=target_supports_rollback,
    )
    return DeploymentManifest(
        deployment_id=deployment_id,
        target=spec.target,
        target_name=target_name,
        status="pending",
        model_name=model.name,
        public=bool(spec.public),
        spec=spec.model_dump(mode="json"),
        target_capabilities=list(target_capabilities),
        version_summary=version_summary,
        rollout_stages=rollout_stages,
        rollback=rollback,
        metadata={
            "spec_metadata": _as_dict(spec.metadata),
            "model_metadata": _as_dict(model.metadata),
            "rollout_strategy": _rollout_config(spec).get("strategy"),
        },
    )
