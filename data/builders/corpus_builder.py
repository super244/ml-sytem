from __future__ import annotations

from collections.abc import Iterable
import csv
from dataclasses import dataclass, field
import glob
import io
import json
import math
from pathlib import Path
import random
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

from ai_factory.core.artifacts import current_git_sha, sha256_file, sha256_text, write_json, write_jsonl, write_markdown
from ai_factory.core.hashing import normalize_text, stable_question_fingerprint
from ai_factory.core.schemas import DatasetBuildInfo, DatasetFileInfo, DatasetManifest, MathRecord, PackagedMathRecord, SourceLineage, StepCheck
from data.builders.pack_registry import build_derived_packs
from data.quality.contamination import apply_contamination_status, deduplicate_near_duplicates
from data.quality.difficulty import difficulty_score, estimate_difficulty, normalize_difficulty
from data.quality.scoring import estimate_quality_score
from data.quality.stats import compute_record_stats
from data.reports.cards import pack_card_text, size_report_markdown


QUESTION_KEYS = ("question", "problem", "prompt", "instruction")
SOLUTION_KEYS = ("solution", "rationale", "response", "gold_solution", "answer_explanation", "output")
FINAL_ANSWER_KEYS = ("final_answer", "answer", "target", "reference_answer")
TOPIC_KEYS = ("topic", "subject", "category")
SOURCE_KEYS = ("source", "dataset", "competition")


@dataclass
class ProcessingConfig:
    seed: int = 42
    eval_ratio: float = 0.15
    test_ratio: float = 0.05
    dataset_version: str = "v2"
    processing_version: str = "v1"
    source_spec_version: str = "v1"
    min_difficulty: str = "hard"
    max_samples: int | None = None
    hard_only: bool = False
    sources: list[Any] | None = None
    failure_logs: list[str] | None = None
    topic_allowlist: list[str] | None = None
    source_allowlist: list[str] | None = None
    contamination_sources: list[str] | None = None
    output_dir: str = "data/processed"
    system_prompt: str = (
        "You are an elite competition mathematician. Solve carefully, show your reasoning, "
        "and finish with `Final Answer: ...`."
    )
    build_id: str | None = None
    require_final_answer: bool = True
    output_subdir: str | None = None
    derived_packs: list[str] | None = None


@dataclass
class SourceSpec:
    id: str
    kind: str = "local"
    path: str | None = None
    split: str = "train"
    revision: str | None = None
    cache_dir: str | None = None
    sample_ratio: float | None = None
    version: str | None = None
    optional: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def load_processing_config(path: str | Path) -> ProcessingConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return ProcessingConfig(**raw)


def resolve_source_paths(paths: list[str] | None) -> list[Path]:
    if not paths:
        return []
    resolved: list[Path] = []
    for path_str in paths:
        path = Path(path_str)
        if any(char in path_str for char in "*?[]"):
            resolved.extend(sorted(Path(match) for match in glob.glob(path_str, recursive=True)))
            continue
        if path.is_dir():
            resolved.extend(sorted(child for child in path.iterdir() if child.is_file()))
            continue
        resolved.append(path)
    return resolved


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                yield from payload["data"]
                return
            yield payload
            return
        if isinstance(payload, list):
            yield from payload
            return
    if suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            yield from reader
        return
    raise ValueError(f"Unsupported file type: {path}")


def _slug_source_id(value: str | None, index: int) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "")).strip("_").lower()
    return cleaned or f"source_{index}"


def _coerce_ratio(value: Any) -> float | None:
    if value is None:
        return None
    ratio = float(value)
    if ratio < 0:
        return 0.0
    return min(ratio, 1.0)


def _source_kind_from_value(value: str | None) -> str:
    cleaned = str(value or "").strip().lower()
    if cleaned.startswith("s3://"):
        return "s3"
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return "web"
    if cleaned in {"huggingface", "hf", "dataset"}:
        return "huggingface"
    if cleaned in {"local", "composite", "mix", "group", "s3", "web"}:
        return cleaned
    return "local"


def _parse_source_entry(entry: Any, index: int) -> list[SourceSpec]:
    if isinstance(entry, str):
        return [
            SourceSpec(
                id=_slug_source_id(Path(entry).stem or entry, index),
                kind=_source_kind_from_value(entry),
                path=entry,
            )
        ]
    if not isinstance(entry, dict):
        raise TypeError(f"Unsupported source spec: {entry!r}")

    kind = _source_kind_from_value(entry.get("kind") or entry.get("loader") or entry.get("path") or entry.get("url") or entry.get("uri"))
    if kind in {"composite", "mix", "group"}:
        child_specs: list[SourceSpec] = []
        for child_index, child in enumerate(entry.get("sources") or []):
            child_specs.extend(_parse_source_entry(child, child_index))
        parent_ratio = _coerce_ratio(entry.get("sample_ratio") or entry.get("ratio") or entry.get("mix_ratio"))
        inherited_version = entry.get("version")
        inherited_optional = bool(entry.get("optional", False))
        inherited_metadata = {k: v for k, v in entry.items() if k not in {"kind", "loader", "sources", "sample_ratio", "ratio", "mix_ratio"}}
        for child_spec in child_specs:
            if parent_ratio is not None:
                child_spec.sample_ratio = _coerce_ratio((child_spec.sample_ratio or 1.0) * parent_ratio)
            if inherited_version and not child_spec.version:
                child_spec.version = str(inherited_version)
            child_spec.optional = child_spec.optional or inherited_optional
            child_spec.metadata = {**inherited_metadata, **child_spec.metadata}
        return child_specs

    path = entry.get("path") or entry.get("url") or entry.get("uri")
    source_id = entry.get("id") or _slug_source_id(path or f"source_{index}", index)
    metadata = dict(entry.get("metadata") or {})
    for key in ("filters", "loader_kwargs", "headers", "query_params"):
        if key in entry and entry[key] is not None:
            metadata[key] = entry[key]
    return [
        SourceSpec(
            id=str(source_id),
            kind=kind,
            path=path,
            split=str(entry.get("split", "train")),
            revision=entry.get("revision"),
            cache_dir=entry.get("cache_dir"),
            sample_ratio=_coerce_ratio(entry.get("sample_ratio") or entry.get("ratio") or entry.get("mix_ratio")),
            version=str(entry["version"]) if entry.get("version") is not None else None,
            optional=bool(entry.get("optional", False)),
            metadata=metadata,
        )
    ]


def coerce_source_specs(raw_sources: list[Any] | None) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    for index, entry in enumerate(raw_sources or []):
        specs.extend(_parse_source_entry(entry, index))
    return specs


def _read_records_from_text(text: str, *, suffix: str = "", content_type: str = "") -> list[dict[str, Any]]:
    body = text.strip()
    if not body:
        return []
    suffix = suffix.lower()
    content_type = content_type.lower()
    if suffix == ".csv" or "csv" in content_type:
        return list(csv.DictReader(io.StringIO(text)))
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                return [dict(item) for item in payload["data"]]
            return [payload]
    if suffix == ".jsonl" or "jsonl" in content_type:
        return [json.loads(line) for line in body.splitlines() if line.strip()]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in body.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [dict(item) for item in payload["data"]]
        return [payload]
    return []


def _load_local_source_rows(spec: SourceSpec) -> list[dict[str, Any]]:
    if not spec.path:
        return []
    paths = resolve_source_paths([spec.path])
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(dict(row) for row in read_records(path))
    return rows


def _load_huggingface_source_rows(spec: SourceSpec) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            f"Source '{spec.id}' requires the optional 'datasets' dependency. Install it or mark the source optional."
        ) from exc
    dataset = load_dataset(spec.path, split=spec.split, cache_dir=spec.cache_dir, revision=spec.revision)
    return [dict(row) for row in dataset]


def _load_web_source_rows(spec: SourceSpec) -> list[dict[str, Any]]:
    if not spec.path:
        return []
    request = Request(spec.path, headers={"User-Agent": "ai-factory-data-loader/1.0"})
    with urlopen(request) as response:  # nosec B310 - controlled loader utility
        body = response.read().decode(response.headers.get_content_charset() or "utf-8")
        content_type = response.headers.get("Content-Type", "")
    suffix = Path(urlparse(spec.path).path).suffix
    return _read_records_from_text(body, suffix=suffix, content_type=content_type)


def _load_s3_source_rows(spec: SourceSpec) -> list[dict[str, Any]]:
    if not spec.path:
        return []
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            f"Source '{spec.id}' requires the optional 'boto3' dependency. Install it or mark the source optional."
        ) from exc
    parsed = urlparse(spec.path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {spec.path}")
    client = boto3.client("s3")
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    content_type = response.get("ContentType", "")
    suffix = Path(key).suffix
    return _read_records_from_text(body, suffix=suffix, content_type=content_type)


def load_source_rows(spec: SourceSpec) -> list[dict[str, Any]]:
    kind = spec.kind.lower()
    if kind == "local":
        return _load_local_source_rows(spec)
    if kind == "huggingface":
        return _load_huggingface_source_rows(spec)
    if kind == "web":
        return _load_web_source_rows(spec)
    if kind == "s3":
        return _load_s3_source_rows(spec)
    raise ValueError(f"Unsupported source kind: {spec.kind}")


def _sample_source_rows(rows: list[dict[str, Any]], spec: SourceSpec, seed: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    ratio = spec.sample_ratio
    if ratio is None or ratio >= 1.0:
        return rows
    if ratio <= 0:
        return []
    sample_size = min(len(rows), max(1, math.ceil(len(rows) * ratio)))
    rng = random.Random(f"{seed}:{spec.id}")
    chosen_indexes = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in chosen_indexes]


def load_source_records(
    specs: list[SourceSpec],
    *,
    seed: int,
) -> tuple[list[tuple[SourceSpec, dict[str, Any]]], list[dict[str, Any]], list[str]]:
    loaded_rows: list[tuple[SourceSpec, dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    warnings: list[str] = []
    for spec in specs:
        try:
            raw_rows = load_source_rows(spec)
            selected_rows = _sample_source_rows(raw_rows, spec, seed)
        except Exception as exc:  # noqa: BLE001
            if spec.optional:
                warnings.append(f"Skipped optional source '{spec.id}': {exc}")
                summaries.append(
                    {
                        "id": spec.id,
                        "kind": spec.kind,
                        "path": spec.path,
                        "version": spec.version,
                        "sample_ratio": spec.sample_ratio,
                        "optional": True,
                        "status": "skipped",
                        "rows_loaded": 0,
                        "rows_selected": 0,
                        "error": str(exc),
                    }
                )
                continue
            raise
        summaries.append(
            {
                "id": spec.id,
                "kind": spec.kind,
                "path": spec.path,
                "version": spec.version,
                "sample_ratio": spec.sample_ratio,
                "optional": spec.optional,
                "status": "loaded",
                "rows_loaded": len(raw_rows),
                "rows_selected": len(selected_rows),
            }
        )
        loaded_rows.extend((spec, row) for row in selected_rows)
    return loaded_rows, summaries, warnings


def _source_lineage_notes(spec: SourceSpec) -> list[str]:
    notes = [f"kind={spec.kind}"]
    if spec.version:
        notes.append(f"version={spec.version}")
    if spec.sample_ratio is not None:
        notes.append(f"sample_ratio={spec.sample_ratio}")
    return notes


def _source_metadata(spec: SourceSpec, summary: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "source_spec_id": spec.id,
        "source_kind": spec.kind,
        "source_path": spec.path,
        "source_split": spec.split,
        "source_version": spec.version,
        "source_sample_ratio": spec.sample_ratio,
    }
    metadata.update(summary)
    metadata.update(spec.metadata)
    return {key: value for key, value in metadata.items() if value is not None}


def _build_source_input_infos(specs: list[SourceSpec], summaries: list[dict[str, Any]]) -> list[DatasetFileInfo]:
    summary_lookup = {
        (summary["id"], summary.get("path"), summary.get("version")): summary for summary in summaries
    }
    inputs: list[DatasetFileInfo] = []
    for spec in specs:
        if spec.kind == "local" and spec.path:
            local_paths = resolve_source_paths([spec.path])
            if local_paths:
                inputs.extend(_build_file_info(path) for path in local_paths)
                continue
        summary = summary_lookup.get((spec.id, spec.path, spec.version))
        selected_rows = summary.get("rows_selected", 0) if summary else 0
        inputs.append(
            DatasetFileInfo(
                path=spec.path or spec.id,
                sha256="",
                size_bytes=0,
                num_rows=selected_rows,
            )
        )
    return inputs


def first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def normalize_topic(value: str | None) -> str:
    if not value:
        return "general"
    cleaned = normalize_text(value).lower()
    topic_map = {
        "nt": "number theory",
        "number_theory": "number theory",
        "calc": "calculus",
        "geo": "geometry",
        "comb": "combinatorics",
    }
    return topic_map.get(cleaned, cleaned)


def _coerce_step_checks(step_checks: Any) -> list[dict[str, Any]]:
    if not step_checks:
        return []
    if isinstance(step_checks, str):
        step_checks = [segment.strip() for segment in step_checks.split("||") if segment.strip()]
    normalized: list[dict[str, Any]] = []
    for item in step_checks:
        if isinstance(item, dict):
            normalized.append(StepCheck.model_validate(item).model_dump())
        else:
            normalized.append(StepCheck(kind="substring", value=str(item)).model_dump())
    return normalized


def normalize_record(
    record: dict[str, Any],
    default_source: str,
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    question = first_text(record, QUESTION_KEYS)
    solution = first_text(record, SOLUTION_KEYS)
    if not question or not solution:
        return None
    topic = normalize_topic(first_text(record, TOPIC_KEYS))
    source = first_text(record, SOURCE_KEYS) or default_source
    final_answer = first_text(record, FINAL_ANSWER_KEYS)
    difficulty = normalize_difficulty(record.get("difficulty") or first_text(record, ("level", "competition_level")))
    reasoning_style = str(record.get("reasoning_style") or ("verification" if record.get("failure_case") else "mixed"))
    loader_kind = str((source_meta or {}).get("source_kind") or "local")
    origin_path = str((source_meta or {}).get("source_path") or record.get("origin_path") or default_source)
    lineage = record.get("lineage") or SourceLineage(
        dataset_id=source,
        dataset_family=source,
        origin_path=origin_path,
        loader=loader_kind,
        source_url=(source_meta or {}).get("source_path") if loader_kind in {"web", "s3"} else None,
    ).model_dump()
    if not isinstance(lineage, dict):
        lineage = SourceLineage.model_validate(lineage).model_dump()
    if source_meta:
        notes = list(lineage.get("notes", []))
        notes.extend(
            _source_lineage_notes(
                SourceSpec(
                    id=default_source,
                    kind=loader_kind,
                    path=origin_path,
                    sample_ratio=source_meta.get("source_sample_ratio"),
                    version=source_meta.get("source_version"),
                )
            )
        )
        lineage["notes"] = list(dict.fromkeys(notes))
        lineage["loader"] = loader_kind
        lineage.setdefault("origin_path", origin_path)
    normalized = {
        "schema_version": "v2",
        "id": record.get("id") or stable_question_fingerprint(question),
        "question": question.strip(),
        "solution": solution.strip(),
        "difficulty": difficulty or estimate_difficulty(question, solution),
        "topic": topic,
        "source": source,
        "final_answer": final_answer,
        "step_checks": _coerce_step_checks(record.get("step_checks") or record.get("key_steps")),
        "tags": record.get("tags", []),
        "dataset_split": record.get("dataset_split", "unspecified"),
        "subtopic": record.get("subtopic"),
        "quality_score": float(record.get("quality_score", 0.0) or 0.0),
        "reasoning_style": reasoning_style,
        "failure_case": bool(record.get("failure_case", False)),
        "pack_id": record.get("pack_id") or source,
        "contamination": record.get("contamination") or {
            "checked_against": [],
            "exact_match": False,
            "near_match": False,
            "max_similarity": 0.0,
            "notes": [],
        },
        "lineage": lineage,
        "generator": record.get("generator"),
        "metadata": {
            "original_source": source,
            **({key: value for key, value in (source_meta or {}).items() if value is not None}),
            **(record.get("metadata") or {}),
        },
    }
    if normalized["quality_score"] <= 0.0:
        normalized["quality_score"] = estimate_quality_score(normalized)
    return MathRecord.model_validate(normalized).model_dump()


def filter_records(records: list[dict[str, Any]], config: ProcessingConfig) -> list[dict[str, Any]]:
    threshold = difficulty_score(config.min_difficulty)
    topic_allowlist = {normalize_topic(item) for item in (config.topic_allowlist or [])}
    source_allowlist = {normalize_text(item).lower() for item in (config.source_allowlist or [])}
    kept: list[dict[str, Any]] = []
    for record in records:
        score = difficulty_score(record["difficulty"])
        if config.hard_only and score < difficulty_score("hard"):
            continue
        if score < threshold:
            continue
        if topic_allowlist and record["topic"] not in topic_allowlist:
            continue
        if source_allowlist and normalize_text(record["source"]).lower() not in source_allowlist:
            continue
        kept.append(record)
    return kept


def stratified_split(
    records: list[dict[str, Any]],
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        bucket_key = f"{record['topic']}::{record['difficulty']}"
        buckets.setdefault(bucket_key, []).append(record)
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        test_count = max(1, int(len(bucket) * test_ratio)) if len(bucket) >= 10 else 0
        eval_count = max(1, int(len(bucket) * eval_ratio)) if len(bucket) >= 4 else 0
        test_records.extend(bucket[:test_count])
        eval_records.extend(bucket[test_count : test_count + eval_count])
        train_records.extend(bucket[test_count + eval_count :])
    for record in train_records:
        record["dataset_split"] = "train"
    for record in eval_records:
        record["dataset_split"] = "eval"
    for record in test_records:
        record["dataset_split"] = "test"
    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    rng.shuffle(test_records)
    return train_records, eval_records, test_records


def build_messages(record: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    difficulty_line = f"Difficulty target: {record['difficulty']}.\n" if record.get("difficulty") else ""
    topic_line = f"Topic: {record['topic']}.\n" if record.get("topic") else ""
    reasoning_line = (
        f"Reasoning style: {record['reasoning_style']}.\n" if record.get("reasoning_style") else ""
    )
    failure_line = (
        "This problem comes from a prior model failure case, so be extra careful about correctness.\n"
        if record.get("failure_case")
        else ""
    )
    user_prompt = (
        "Solve the following math problem.\n"
        f"{topic_line}"
        f"{difficulty_line}"
        f"{reasoning_line}"
        f"{failure_line}"
        "Show the reasoning step by step and end with `Final Answer: ...`.\n\n"
        f"Problem:\n{record['question']}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": record["solution"]},
    ]


def package_record(record: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    packaged = dict(record)
    packaged["messages"] = build_messages(record, system_prompt)
    packaged["weight"] = round(
        1.0
        + (0.4 if record.get("failure_case") else 0.0)
        + (0.2 if record.get("difficulty") == "olympiad" else 0.0)
        + (0.2 if record.get("reasoning_style") == "proof" else 0.0),
        3,
    )
    return PackagedMathRecord.model_validate(packaged).model_dump()


def _build_file_info(path: Path) -> DatasetFileInfo:
    num_rows = 0
    if path.exists() and path.suffix.lower() == ".jsonl":
        num_rows = sum(1 for line in path.read_text().splitlines() if line.strip())
    return DatasetFileInfo(
        path=str(path),
        sha256=sha256_file(path) if path.exists() else "",
        size_bytes=path.stat().st_size if path.exists() else 0,
        num_rows=num_rows,
    )


def build_corpus(config: ProcessingConfig, config_path: str | Path) -> dict[str, Any]:
    all_records: list[dict[str, Any]] = []
    source_specs = coerce_source_specs(config.sources)
    source_rows, source_summaries, source_warnings = load_source_records(source_specs, seed=config.seed)
    summary_lookup = {
        (summary["id"], summary.get("path"), summary.get("version")): summary for summary in source_summaries
    }
    for spec, raw in source_rows:
        source_meta = _source_metadata(
            spec,
            {
                **summary_lookup.get((spec.id, spec.path, spec.version), {}),
                "dataset_version": config.dataset_version,
                "processing_version": config.processing_version,
                "source_spec_version": config.source_spec_version,
            },
        )
        normalized = normalize_record(raw, default_source=spec.id, source_meta=source_meta)
        if normalized:
            all_records.append(normalized)
    for path in resolve_source_paths(config.failure_logs):
        for raw in read_records(path):
            normalized = normalize_record(dict(raw, failure_case=True), default_source=f"failure_log::{path.stem}")
            if normalized:
                normalized["failure_case"] = True
                normalized["reasoning_style"] = "verification"
                normalized["tags"] = sorted(set(normalized.get("tags", []) + ["failure_replay"]))
                normalized["quality_score"] = max(normalized.get("quality_score", 0.0), 0.85)
                all_records.append(normalized)

    all_records = filter_records(all_records, config)
    all_records, dedupe_report = deduplicate_near_duplicates(all_records)

    benchmark_records: list[dict[str, Any]] = []
    for path in resolve_source_paths(config.contamination_sources):
        for raw in read_records(path):
            normalized = normalize_record(raw, default_source=path.stem)
            if normalized:
                benchmark_records.append(normalized)
    if benchmark_records:
        all_records = apply_contamination_status(all_records, benchmark_records)
        all_records = [record for record in all_records if not record["contamination"]["exact_match"]]

    if config.max_samples:
        all_records = all_records[: config.max_samples]

    train_records, eval_records, test_records = stratified_split(
        records=all_records,
        eval_ratio=config.eval_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    normalized_all = [package_record(item, config.system_prompt) for item in all_records]
    train_packaged = [package_record(item, config.system_prompt) for item in train_records]
    eval_packaged = [package_record(item, config.system_prompt) for item in eval_records]
    test_packaged = [package_record(item, config.system_prompt) for item in test_records]

    output_dir = Path(config.output_dir)
    if config.output_subdir:
        output_dir = output_dir / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = output_dir / "normalized_all.jsonl"
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    test_path = output_dir / "test.jsonl"
    write_jsonl(normalized_path, normalized_all)
    write_jsonl(train_path, train_packaged)
    write_jsonl(eval_path, eval_packaged)
    write_jsonl(test_path, test_packaged)

    stats = {
        "all": compute_record_stats(normalized_all),
        "train": compute_record_stats(train_packaged),
        "eval": compute_record_stats(eval_packaged),
        "test": compute_record_stats(test_packaged),
        "dedupe": dedupe_report,
    }
    stats_path = output_dir / "stats.json"
    write_json(stats_path, stats)

    repo_root = Path(__file__).resolve().parents[2]
    config_text = Path(config_path).read_text()
    build = DatasetBuildInfo(
        build_id=config.build_id or sha256_text(config_text)[:12],
        git_sha=current_git_sha(repo_root),
        config_path=str(config_path),
        config_sha256=sha256_text(config_text),
        seed=config.seed,
        notes=["atlas_math_lab_v2", f"processing_version={config.processing_version}", f"dataset_version={config.dataset_version}"],
    )
    manifest = DatasetManifest(
        manifest_type="dataset",
        build=build,
        pack_id="core_train_mix",
        description="Canonical processed mixture for Atlas Math Lab.",
        inputs=[
            *_build_source_input_infos(source_specs, source_summaries),
            *[_build_file_info(path) for path in resolve_source_paths(config.failure_logs)],
        ],
        outputs=[_build_file_info(path) for path in [normalized_path, train_path, eval_path, test_path, stats_path]],
        source_lineage=[SourceLineage.model_validate(record["lineage"]) for record in normalized_all[:1000]],
        stats=stats,
        metadata={
            "system_prompt": config.system_prompt,
            "dataset_version": config.dataset_version,
            "processing_version": config.processing_version,
            "source_spec_version": config.source_spec_version,
            "source_summaries": source_summaries,
            "source_warnings": source_warnings,
        },
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest.model_dump())

    card_body = pack_card_text(
        "core_train_mix",
        "Canonical mixed-source processed dataset for training, evaluation, and benchmark pack derivation.",
        normalized_all,
    )
    card_path = output_dir / "card.md"
    write_markdown(card_path, card_body)

    size_report_path = output_dir / "size_report.md"
    write_markdown(
        size_report_path,
        size_report_markdown(
            "Processed Dataset Size Report",
            [file_info.model_dump() for file_info in manifest.outputs],
        ),
    )

    derived_pack_summaries = build_derived_packs(
        normalized_all,
        output_dir / "packs",
        build=build,
        pack_ids=config.derived_packs,
    )
    write_json(output_dir / "pack_summary.json", {"packs": derived_pack_summaries})
    return {
        "output_dir": str(output_dir),
        "stats": stats,
        "manifest_path": str(manifest_path),
        "derived_packs": derived_pack_summaries,
        "source_summaries": source_summaries,
        "source_warnings": source_warnings,
    }
