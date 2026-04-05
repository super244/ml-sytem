from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_factory.core.datasets import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_LINEAGE_SUMMARY_PATH,
    DEFAULT_PACK_SUMMARY_PATH,
    DEFAULT_PROCESSED_MANIFEST_PATH,
)
from ai_factory.core.datasets import (
    list_sample_prompts as _list_sample_prompts,
)
from ai_factory.core.datasets import (
    load_catalog as _load_catalog,
)
from ai_factory.core.datasets import (
    load_dataset_provenance as _load_dataset_provenance,
)
from ai_factory.core.datasets import (
    load_lineage_summary as _load_lineage_summary,
)
from ai_factory.core.datasets import (
    load_pack_manifests as _load_pack_manifests,
)
from ai_factory.core.datasets import (
    load_pack_summary as _load_pack_summary,
)
from ai_factory.core.datasets import (
    load_processed_manifest as _load_processed_manifest,
)


def _build_catalog_profile_summary(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    kind_counts = Counter(str(entry.get("kind", "unknown")) for entry in datasets)
    family_counts = Counter(str(entry.get("family", "unknown")) for entry in datasets)
    topic_counts = Counter(str(entry.get("topic", "unknown")) for entry in datasets)
    size_bytes = sum(int(entry.get("size_bytes", 0) or 0) for entry in datasets)
    rows = sum(int(entry.get("num_rows", 0) or 0) for entry in datasets)
    return {
        "num_datasets": len(datasets),
        "kind_counts": dict(kind_counts),
        "family_counts": dict(family_counts),
        "topic_counts": dict(topic_counts),
        "total_rows": rows,
        "total_bytes": size_bytes,
        "top_kinds": kind_counts.most_common(5),
        "top_families": family_counts.most_common(5),
        "top_topics": topic_counts.most_common(5),
    }


class CatalogPreview(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    question: str
    difficulty: str
    topic: str
    final_answer: str | None = None


class CatalogEntry(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    title: str
    kind: str
    family: str
    topic: str
    path: str
    num_rows: int
    size_bytes: int
    description: str
    reasoning_style: str | None = None
    sha256: str | None = None  # cryptographic hash
    lineage_data: dict[str, Any] | None = None  # explicit lineage data
    preview_examples: list[CatalogPreview] = Field(default_factory=list)
    profile_summary: dict[str, Any] = Field(default_factory=dict)
    quality_summary: dict[str, Any] = Field(default_factory=dict)


class CatalogSummary(BaseModel):
    model_config = ConfigDict(strict=True)
    num_datasets: int = 0
    custom_datasets: int = 0
    public_datasets: int = 0
    total_bytes: int = 0
    total_rows: int = 0
    profile_summary: dict[str, Any] = Field(default_factory=dict)


class CatalogModel(BaseModel):
    model_config = ConfigDict(strict=True)
    generated_at: str | None = None
    summary: CatalogSummary = Field(default_factory=CatalogSummary)
    datasets: list[CatalogEntry] = Field(default_factory=list)


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH, *, repo_root: str | Path | None = None) -> CatalogModel:
    data = _load_catalog(path, repo_root=repo_root)
    datasets = data.get("datasets") or []
    if "summary" not in data or not data["summary"]:
        data["summary"] = {
            "num_datasets": len(datasets),
            "custom_datasets": sum(1 for entry in datasets if entry.get("kind") == "custom"),
            "public_datasets": sum(1 for entry in datasets if entry.get("kind") == "public"),
            "total_bytes": sum(int(entry.get("size_bytes", 0) or 0) for entry in datasets),
            "total_rows": sum(int(entry.get("num_rows", 0) or 0) for entry in datasets),
        }
    summary = data["summary"]
    if "profile_summary" not in summary or not summary["profile_summary"]:
        summary["profile_summary"] = _build_catalog_profile_summary(datasets)
    for entry in datasets:
        entry.setdefault("profile_summary", {})
        entry.setdefault("quality_summary", {})
    return CatalogModel.model_validate(data)


def list_catalog_entries(
    kind: str | None = None,
    path: str | Path = DEFAULT_CATALOG_PATH,
    *,
    repo_root: str | Path | None = None,
) -> list[CatalogEntry]:
    catalog = load_catalog(path, repo_root=repo_root)
    entries = catalog.datasets
    if kind is None:
        return entries
    return [e for e in entries if e.kind == kind]


def list_sample_prompts(
    limit: int = 12,
    path: str | Path = DEFAULT_CATALOG_PATH,
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    return _list_sample_prompts(limit, path, repo_root=repo_root)


def load_dataset_provenance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _load_dataset_provenance(*args, **kwargs)


def load_lineage_summary(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _load_lineage_summary(*args, **kwargs)


def load_pack_manifests(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return _load_pack_manifests(*args, **kwargs)


def load_pack_summary(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _load_pack_summary(*args, **kwargs)


def load_processed_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _load_processed_manifest(*args, **kwargs)


__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_LINEAGE_SUMMARY_PATH",
    "DEFAULT_PACK_SUMMARY_PATH",
    "DEFAULT_PROCESSED_MANIFEST_PATH",
    "CatalogModel",
    "CatalogEntry",
    "CatalogSummary",
    "CatalogPreview",
    "list_catalog_entries",
    "list_sample_prompts",
    "load_catalog",
    "load_dataset_provenance",
    "load_lineage_summary",
    "load_pack_manifests",
    "load_pack_summary",
    "load_processed_manifest",
]
