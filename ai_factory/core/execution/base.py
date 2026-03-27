from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_factory.core.instances.models import EnvironmentSpec, ExecutionHandle, InstanceManifest


class CommandSpec(BaseModel):
    argv: list[str]
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    expected_artifacts: dict[str, str] = Field(default_factory=dict)
    long_running: bool = False


class RunnerPayload(BaseModel):
    artifacts_dir: str
    instance_id: str
    environment: EnvironmentSpec
    command: CommandSpec


def encode_payload(payload: RunnerPayload) -> str:
    body = json.dumps(payload.model_dump(mode="json"), separators=(",", ":"))
    return base64.urlsafe_b64encode(body.encode("utf-8")).decode("utf-8")


def decode_payload(value: str) -> RunnerPayload:
    body = base64.urlsafe_b64decode(value.encode("utf-8")).decode("utf-8")
    return RunnerPayload.model_validate_json(body)


class BaseExecutor(ABC):
    backend_name: str

    @abstractmethod
    def start(
        self,
        manifest: InstanceManifest,
        command: CommandSpec,
        *,
        artifacts_dir: str | Path,
        stdout_path: str,
        stderr_path: str,
    ) -> ExecutionHandle:
        raise NotImplementedError

    def read_remote_file(self, manifest: InstanceManifest, path: str) -> str:
        raise NotImplementedError(f"{self.backend_name} does not support remote reads")
