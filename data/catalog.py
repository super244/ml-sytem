from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = Path("data/catalog.json")
DEFAULT_PROCESSED_MANIFEST_PATH = Path("data/processed/manifest.json")
DEFAULT_PACK_SUMMARY_PATH = Path("data/processed/pack_summary.json")
DEFAULT_LINEAGE_SUMMARY_PATH = Path("data/processed/lineage_summary.json")


def _resolve_path(path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute() or repo_root is None:
        return resolved
    return Path(repo_root) / resolved


def _load_json(path: str | Path, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    resolved = _resolve_path(path, repo_root=repo_root)
    if not resolved.exists():
        return {}
    data = json.loads(resolved.read_text())
    return data if isinstance(data, dict) else {}


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    catalog = _load_json(path, repo_root=repo_root)
    if not catalog:
        return {"generated_at": None, "datasets": [], "summary": {}}
    return catalog


def load_processed_manifest(
    path: str | Path = DEFAULT_PROCESSED_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    return _load_json(path, repo_root=repo_root)


def list_catalog_entries(
    kind: str | None = None,
    path: str | Path = DEFAULT_CATALOG_PATH,
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    entries = load_catalog(path, repo_root=repo_root).get("datasets", [])
    if kind is None:
        return entries if isinstance(entries, list) else []
    return [entry for entry in entries if entry.get("kind") == kind]


def list_sample_prompts(
    limit: int = 12,
    path: str | Path = DEFAULT_CATALOG_PATH,
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for entry in load_catalog(path, repo_root=repo_root).get("datasets", []):
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


def load_pack_summary(path: str | Path = DEFAULT_PACK_SUMMARY_PATH, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    summary = _load_json(path, repo_root=repo_root)
    if not summary:
        return {"packs": []}
    return summary


def load_lineage_summary(
    path: str | Path = DEFAULT_LINEAGE_SUMMARY_PATH,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    return _load_json(path, repo_root=repo_root)


def load_pack_manifests(
    path: str | Path = DEFAULT_PACK_SUMMARY_PATH,
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    summary = load_pack_summary(path, repo_root=repo_root)
    manifests: list[dict[str, Any]] = []
    for pack in summary.get("packs", []):
        manifest_path = pack.get("manifest_path")
        if not manifest_path and pack.get("path"):
            manifest_path = str(Path(str(pack["path"])).with_name("manifest.json"))
        if not manifest_path:
            manifests.append({})
            continue
        manifest = _load_json(manifest_path, repo_root=repo_root)
        manifests.append(manifest)
    return manifests


def load_dataset_provenance(
    *,
    repo_root: str | Path | None = None,
    processed_manifest_path: str | Path = DEFAULT_PROCESSED_MANIFEST_PATH,
    pack_summary_path: str | Path = DEFAULT_PACK_SUMMARY_PATH,
    lineage_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    processed_manifest = load_processed_manifest(processed_manifest_path, repo_root=repo_root)
    resolved_lineage_summary_path = lineage_summary_path
    if resolved_lineage_summary_path is None:
        resolved_lineage_summary_path = (
            processed_manifest.get("metadata", {}).get("lineage_summary_path")
            or DEFAULT_LINEAGE_SUMMARY_PATH
        )
    return {
        "processed_manifest": processed_manifest,
        "pack_summary": load_pack_summary(pack_summary_path, repo_root=repo_root),
        "pack_manifests": load_pack_manifests(pack_summary_path, repo_root=repo_root),
        "lineage_summary": load_lineage_summary(resolved_lineage_summary_path, repo_root=repo_root),
    }
