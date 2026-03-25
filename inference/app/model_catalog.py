from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def list_model_catalog(path: str | Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    models: list[dict[str, Any]] = []
    for item in payload.get("models", []):
        adapter_path = item.get("adapter_path")
        models.append(
            {
                "name": item["name"],
                "label": item.get("label") or item["name"],
                "description": item.get("description"),
                "base_model": item["base_model"],
                "adapter_path": adapter_path,
                "available": adapter_path is None or Path(adapter_path).exists(),
                "tags": item.get("tags", []),
            }
        )
    return models
