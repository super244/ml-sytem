from __future__ import annotations

import importlib

import pytest


def _import_or_skip(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - only used when surfaces are not wired yet
        pytest.skip(f"{module_name} is not available yet: {exc}", allow_module_level=True)


ai_factory_cli = _import_or_skip("ai_factory.cli")
_import_or_skip("inference.app.services.instance_service")
_import_or_skip("inference.app.routers.instances")


def test_cli_module_exposes_main():
    assert hasattr(ai_factory_cli, "main")
