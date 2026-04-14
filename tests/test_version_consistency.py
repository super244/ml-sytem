from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import ai_factory
from ai_factory.version import VERSION
from inference.app.config import get_settings


def test_version_matches_pyproject() -> None:
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    assert data["project"]["version"] == VERSION


def test_ai_factory_init_exports_same_version() -> None:
    assert ai_factory.__version__ == VERSION


def test_inference_settings_default_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_FACTORY_API_VERSION", raising=False)
    assert get_settings().version == VERSION
