from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_SCALE_TAGS = {
    "baseline",
    "specialist",
    "scaleup",
    "local",
    "fast_iteration",
    "long_context",
    "pretraining",
}
_PARAMETER_LABEL_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<suffix>[bk])", re.IGNORECASE)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _parse_parameter_size_b(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1_000_000:
            return numeric / 1_000_000_000.0
        return numeric

    normalized = str(value).strip().lower().replace("_", "")
    if not normalized:
        return None

    match = _PARAMETER_LABEL_RE.fullmatch(normalized) or _PARAMETER_LABEL_RE.search(normalized)
    if match:
        numeric = float(match.group("value"))
        suffix = match.group("suffix")
        if suffix == "b":
            return numeric
        if suffix == "k":
            return numeric / 1_000_000.0

    try:
        numeric = float(normalized)
    except ValueError:
        return None
    if numeric > 1_000_000:
        return numeric / 1_000_000_000.0
    return numeric


def _format_parameter_size_label(value: float | None) -> str | None:
    if value is None:
        return None
    if value.is_integer():
        return f"{int(value)}B"
    return f"{value:g}B"


def _infer_quantization(item: Mapping[str, Any]) -> str:
    explicit = item.get("quantization")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if item.get("load_in_4bit") is True:
        return "4bit"
    if item.get("load_in_8bit") is True:
        return "8bit"
    if item.get("use_full_precision") is True:
        return "16bit"
    dtype = str(item.get("dtype") or "").strip().lower()
    if dtype in {"bf16", "bfloat16", "fp16", "float16", "float32", "fp32"}:
        return "16bit"
    return "4bit"


def _infer_parameter_size_b(item: Mapping[str, Any]) -> float | None:
    for key in ("parameter_size_b", "parameter_size", "target_parameters"):
        parsed = _parse_parameter_size_b(item.get(key))
        if parsed is not None:
            return parsed
    for candidate in (
        item.get("label"),
        item.get("name"),
        item.get("base_model"),
        item.get("base_model_name"),
    ):
        if not candidate:
            continue
        parsed = _parse_parameter_size_b(candidate)
        if parsed is not None:
            return parsed
    return None


def _infer_tier(item: Mapping[str, Any], tags: list[str], parameter_size_b: float | None) -> str | None:
    explicit = item.get("tier")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    for candidate in tags:
        lowered = candidate.lower()
        if lowered in _SCALE_TAGS:
            return candidate
    if parameter_size_b is not None and parameter_size_b >= 7:
        return "scaleup"
    if parameter_size_b is not None and parameter_size_b <= 1:
        return "fast_iteration"
    return None


def _availability_context(item: Mapping[str, Any], *, available: bool) -> dict[str, Any]:
    adapter_path = item.get("adapter_path")
    if adapter_path:
        path = Path(str(adapter_path))
        exists = path.exists()
        return {
            "state": "available" if available else "missing",
            "detail": "adapter path present" if exists else f"adapter path missing: {path}",
            "adapter_path": str(path),
            "adapter_path_exists": exists,
            "source": "adapter",
        }
    base_model = item.get("base_model") or item.get("base_model_name")
    return {
        "state": "available" if available else "missing",
        "detail": f"served directly from {base_model}" if base_model else "served directly from base weights",
        "adapter_path": None,
        "adapter_path_exists": None,
        "source": "base_model",
    }


def normalize_model_record(
    item: Mapping[str, Any], *, source: str = "registry", available: bool | None = None
) -> dict[str, Any]:
    record = dict(item)
    tags = _dedupe([str(tag) for tag in (item.get("tags") or []) if isinstance(tag, str)])
    parameter_size_b = _infer_parameter_size_b(item)
    parameter_size_label = item.get("parameter_size_label")
    if not isinstance(parameter_size_label, str) or not parameter_size_label.strip():
        parameter_size_label = _format_parameter_size_label(parameter_size_b)
    quantization = _infer_quantization(item)
    tier = _infer_tier(item, tags, parameter_size_b)
    scale_tags = _dedupe(
        [
            *([str(tag) for tag in (item.get("scale_tags") or []) if isinstance(tag, str)]),
            *([tier] if tier else []),
        ]
    )
    resolved_available = available
    if resolved_available is None:
        adapter_path = item.get("adapter_path")
        resolved_available = not adapter_path or Path(str(adapter_path)).exists()

    record.update(
        {
            "label": item.get("label") or item.get("name"),
            "description": item.get("description"),
            "tags": tags,
            "scale_tags": scale_tags,
            "parameter_size_b": parameter_size_b,
            "parameter_size_label": parameter_size_label,
            "quantization": quantization,
            "tier": tier,
            "availability_context": _availability_context(item, available=bool(resolved_available)),
            "available": bool(resolved_available),
            "source": source,
        }
    )
    return record


def summarize_model_catalog(models: list[dict[str, Any]]) -> dict[str, Any]:
    quantization_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    scale_tags: set[str] = set()
    parameter_sizes_b: list[float] = []
    ready = 0

    for model in models:
        if model.get("available"):
            ready += 1
        quantization = str(model.get("quantization") or "unknown")
        quantization_counts[quantization] = quantization_counts.get(quantization, 0) + 1
        tier = model.get("tier")
        if isinstance(tier, str) and tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        for tag in model.get("scale_tags", []) or []:
            if isinstance(tag, str) and tag:
                scale_tags.add(tag)
        size = model.get("parameter_size_b")
        if isinstance(size, (int, float)):
            parameter_sizes_b.append(float(size))

    return {
        "total": len(models),
        "ready": ready,
        "missing": len(models) - ready,
        "quantization_counts": quantization_counts,
        "tier_counts": tier_counts,
        "scale_tags": sorted(scale_tags),
        "parameter_sizes_b": sorted(parameter_sizes_b),
    }


def list_model_catalog(path: str | Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return [normalize_model_record(item) for item in payload.get("models", [])]
