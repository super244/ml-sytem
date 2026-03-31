"""Utility functions for instance management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ai_factory.core.instances.models import InstanceManifest


class _SafeTemplateDict(dict[str, Any]):
    """Safe template dictionary that returns placeholder for missing keys."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    merged = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _source_artifact_ref(manifest: InstanceManifest) -> str | None:
    """Get the source artifact reference from a manifest."""
    return (
        (manifest.artifact_refs.get("published") or {}).get("final_adapter")
        or (manifest.artifact_refs.get("published") or {}).get("merged_model")
        or manifest.artifact_refs.get("source_artifact")
    )


def _stage_for_instance_type(instance_type: str) -> str:
    """Get the stage name for an instance type."""
    mapping = {
        "prepare": "prepare",
        "train": "train",
        "finetune": "finetune",
        "evaluate": "evaluate",
        "inference": "infer",
        "deploy": "publish",
        "report": "decide",
    }
    return mapping.get(instance_type, "train")
