from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
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


def inspect_json_asset(path: str | Path, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    resolved = _resolve_path(path, repo_root=repo_root)
    if not resolved.exists():
        return {"ok": False, "kind": "missing", "path": str(resolved), "detail": "file is missing"}

    raw = resolved.read_text()
    stripped = raw.strip()
    if not stripped:
        return {"ok": False, "kind": "empty", "path": str(resolved), "detail": "file is empty"}
    if stripped.startswith("version https://git-lfs.github.com/spec/v1"):
        return {
            "ok": False,
            "kind": "git_lfs_pointer",
            "path": str(resolved),
            "detail": "file is a Git LFS pointer; run `git lfs pull` or regenerate the asset locally",
        }

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "kind": "invalid_json",
            "path": str(resolved),
            "detail": f"invalid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})",
        }

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "kind": "unexpected_type",
            "path": str(resolved),
            "detail": f"expected a JSON object, found {type(payload).__name__}",
        }

    return {"ok": True, "kind": "json_object", "path": str(resolved), "detail": "valid JSON object"}


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


def load_pack_summary(
    path: str | Path = DEFAULT_PACK_SUMMARY_PATH, *, repo_root: str | Path | None = None
) -> dict[str, Any]:
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
            processed_manifest.get("metadata", {}).get("lineage_summary_path") or DEFAULT_LINEAGE_SUMMARY_PATH
        )
    return {
        "processed_manifest": processed_manifest,
        "pack_summary": load_pack_summary(pack_summary_path, repo_root=repo_root),
        "pack_manifests": load_pack_manifests(pack_summary_path, repo_root=repo_root),
        "lineage_summary": load_lineage_summary(resolved_lineage_summary_path, repo_root=repo_root),
    }


def compute_record_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    difficulty_counts = Counter(record.get("difficulty", "unknown") for record in records)
    topic_counts = Counter(record.get("topic", "unknown") for record in records)
    source_counts = Counter(record.get("source", "unknown") for record in records)
    pack_counts = Counter(record.get("pack_id", "unknown") for record in records)
    reasoning_counts = Counter(record.get("reasoning_style", "mixed") for record in records)
    quality_scores = [float(record.get("quality_score", 0.0) or 0.0) for record in records]

    def _quantile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return round(values[0], 4)
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        if lower == upper:
            return round(ordered[lower], 4)
        weight = position - lower
        return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 4)

    return {
        "num_records": len(records),
        "difficulty_counts": dict(difficulty_counts),
        "topic_counts": dict(topic_counts),
        "source_counts": dict(source_counts),
        "pack_counts": dict(pack_counts),
        "reasoning_style_counts": dict(reasoning_counts),
        "failure_case_count": sum(1 for record in records if record.get("failure_case")),
        "verification_ready_count": sum(1 for record in records if record.get("step_checks")),
        "avg_quality_score": round(mean(quality_scores), 4) if quality_scores else 0.0,
        "quality_score_summary": {
            "min": round(min(quality_scores), 4) if quality_scores else 0.0,
            "max": round(max(quality_scores), 4) if quality_scores else 0.0,
            "p50": _quantile(quality_scores, 0.5),
            "p90": _quantile(quality_scores, 0.9),
        },
        "contaminated_count": sum(
            1
            for record in records
            if (record.get("contamination") or {}).get("exact_match")
            or (record.get("contamination") or {}).get("near_match")
        ),
    }


__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_LINEAGE_SUMMARY_PATH",
    "DEFAULT_PACK_SUMMARY_PATH",
    "DEFAULT_PROCESSED_MANIFEST_PATH",
    "compute_record_stats",
    "inspect_json_asset",
    "list_catalog_entries",
    "list_sample_prompts",
    "load_catalog",
    "load_dataset_provenance",
    "load_lineage_summary",
    "load_pack_manifests",
    "load_pack_summary",
    "load_processed_manifest",
]
