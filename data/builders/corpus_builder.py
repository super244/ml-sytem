from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

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
    min_difficulty: str = "hard"
    max_samples: int | None = None
    hard_only: bool = False
    sources: list[str] | None = None
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
            resolved.extend(sorted(Path().glob(path_str)))
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


def normalize_record(record: dict[str, Any], default_source: str) -> dict[str, Any] | None:
    question = first_text(record, QUESTION_KEYS)
    solution = first_text(record, SOLUTION_KEYS)
    if not question or not solution:
        return None
    topic = normalize_topic(first_text(record, TOPIC_KEYS))
    source = first_text(record, SOURCE_KEYS) or default_source
    final_answer = first_text(record, FINAL_ANSWER_KEYS)
    difficulty = normalize_difficulty(record.get("difficulty") or first_text(record, ("level", "competition_level")))
    reasoning_style = str(record.get("reasoning_style") or ("verification" if record.get("failure_case") else "mixed"))
    lineage = record.get("lineage") or SourceLineage(
        dataset_id=source,
        dataset_family=source,
        origin_path=str(record.get("origin_path") or default_source),
        loader="local",
    ).model_dump()
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
    for path in resolve_source_paths(config.sources):
        default_source = path.stem
        for raw in read_records(path):
            normalized = normalize_record(raw, default_source=default_source)
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
        notes=["atlas_math_lab_v2"],
    )
    manifest = DatasetManifest(
        manifest_type="dataset",
        build=build,
        pack_id="core_train_mix",
        description="Canonical processed mixture for Atlas Math Lab.",
        inputs=[_build_file_info(path) for path in resolve_source_paths(config.sources) + resolve_source_paths(config.failure_logs)],
        outputs=[_build_file_info(path) for path in [normalized_path, train_path, eval_path, test_path, stats_path]],
        source_lineage=[SourceLineage.model_validate(record["lineage"]) for record in normalized_all[:1000]],
        stats=stats,
        metadata={"system_prompt": config.system_prompt},
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
    }
