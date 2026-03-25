from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CATALOG_PATH = Path("data/catalog.json")
DEFAULT_PACK_SUMMARY_PATH = Path("data/processed/pack_summary.json")


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        return {"generated_at": None, "datasets": [], "summary": {}}
    return json.loads(catalog_path.read_text())


def list_catalog_entries(kind: str | None = None, path: str | Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    entries = load_catalog(path).get("datasets", [])
    if kind is None:
        return entries
    return [entry for entry in entries if entry.get("kind") == kind]


def list_sample_prompts(limit: int = 12, path: str | Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for entry in load_catalog(path).get("datasets", []):
        for preview in entry.get("preview_examples", []):
            prompts.append(
                {
                    "dataset_id": entry.get("id"),
                    "dataset_title": entry.get("title"),
                    "question": preview.get("question"),
                    "difficulty": preview.get("difficulty"),
                    "topic": preview.get("topic"),
                }
            )
    return prompts[:limit]


def load_pack_summary(path: str | Path = DEFAULT_PACK_SUMMARY_PATH) -> dict[str, Any]:
    summary_path = Path(path)
    if not summary_path.exists():
        return {"packs": []}
    return json.loads(summary_path.read_text())
