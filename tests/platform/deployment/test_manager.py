from __future__ import annotations

from pathlib import Path

import pytest

from ai_factory.core.schemas import DeploymentSpec, ModelArtifact
from ai_factory.platform.deployment import DeploymentManager
from ai_factory.platform.deployment.targets import DeploymentTarget


class _FakeDeploymentTarget(DeploymentTarget):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Fake Target"
        self.description = "Test deployment target"
        self.capabilities = ["versioning", "rollback"]
        self.supports_staged_rollout = True
        self.supports_rollback = True

    async def prepare_model(self, model: ModelArtifact, spec: DeploymentSpec) -> ModelArtifact:
        metadata = dict(model.metadata)
        metadata["prepared_for"] = spec.target
        return model.model_copy(update={"metadata": metadata})

    async def deploy(self, model: ModelArtifact, spec: DeploymentSpec) -> dict[str, object]:
        return {
            "endpoint": f"https://deployments.local/{spec.target}/{model.version}",
            "revision": model.version,
            "traffic_percent": 100,
        }

    async def get_deployment_status(self, deployment_id: str) -> dict[str, object]:
        return {"deployment_id": deployment_id, "status": "serving", "traffic_percent": 100}

    async def cancel_deployment(self, deployment_id: str) -> bool:
        return True

    async def validate_spec(self, spec: DeploymentSpec) -> list[str]:
        return []


def _build_model() -> ModelArtifact:
    return ModelArtifact(
        name="atlas-math",
        version="1.2.3",
        path="artifacts/models/atlas-math",
        architecture="qwen2",
        parameters=2_000_000_000,
        format="safetensors",
        metadata={
            "base_model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "artifact_sha256": "abc123",
            "lineage": {
                "source_model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
                "training_run_ids": ["run-123"],
                "training_dataset_ids": ["dataset-a", "dataset-b"],
            },
            "tags": ["math", "production"],
            "changelog": ["initial stable release"],
            "release_label": "atlas-math-1.2.3",
        },
    )


def _build_spec() -> DeploymentSpec:
    return DeploymentSpec(
        target="fake",
        model_name="atlas-math",
        public=False,
        config={
            "rollout": {
                "strategy": "canary",
                "stages": [
                    {"name": "canary", "traffic_percent": 5, "duration_minutes": 30},
                    {"name": "ramp", "traffic_percent": 50, "duration_minutes": 60},
                    {"name": "full", "traffic_percent": 100},
                ],
            },
            "rollback": {
                "previous_version": "1.2.2",
                "artifact_path": "artifacts/models/atlas-math-1.2.2",
                "strategy": "previous_version",
            },
        },
        metadata={"release_label": "atlas-math-1.2.3"},
    )


def test_build_deployment_manifest_captures_lineage_and_rollout(tmp_path: Path) -> None:
    manager = DeploymentManager(tmp_path, targets={"fake": _FakeDeploymentTarget()})
    model = _build_model()
    spec = _build_spec()

    manifest = manager.build_deployment_manifest(
        "deploy-demo",
        model,
        spec,
        "Fake Target",
        ["versioning", "rollback"],
        True,
    )

    assert manifest.version_summary.summary_text.startswith("atlas-math@1.2.3")
    assert manifest.version_summary.lineage["training_run_ids"] == ["run-123"]
    assert len(manifest.rollout_stages) == 3
    assert manifest.rollout_stages[-1].traffic_percent == 100
    assert manifest.rollback.ready is True
    assert manifest.summary()["rollback_ready"] is True


@pytest.mark.asyncio
async def test_deploy_model_records_manifest_and_live_status(tmp_path: Path) -> None:
    manager = DeploymentManager(tmp_path, targets={"fake": _FakeDeploymentTarget()})
    deployment_id = await manager.deploy_model(_build_model(), _build_spec())

    status = await manager.get_deployment_status(deployment_id)
    manifest = manager.get_deployment_manifest(deployment_id)
    deployments = await manager.list_deployments()

    assert status["status"] == "deployed"
    assert status["manifest_summary"]["rollback_ready"] is True
    assert status["live_status"]["status"] == "serving"
    assert manifest.version_summary.release_label == "atlas-math-1.2.3"
    assert any(entry["status"] == "deployed" for entry in status["status_history"])
    assert deployments[0]["manifest_summary"]["version"] == "1.2.3"


@pytest.mark.asyncio
async def test_validate_deployment_spec_rejects_bad_rollout_metadata(tmp_path: Path) -> None:
    manager = DeploymentManager(tmp_path, targets={"fake": _FakeDeploymentTarget()})
    spec = DeploymentSpec(
        target="fake",
        model_name="atlas-math",
        config={"rollout": {"stages": [{"name": "canary", "traffic_percent": "oops"}]}},
        metadata={"release_label": "atlas-math-1.2.3"},
    )

    errors = await manager.validate_deployment_spec(spec)

    assert any("traffic_percent must be numeric" in error for error in errors)
