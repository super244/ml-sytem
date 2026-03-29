from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Callable

from ai_factory.core.hashing import normalize_text
from ai_factory.core.io import write_json, write_jsonl, write_markdown
from ai_factory.core.schemas import DatasetBuildInfo, DatasetFileInfo, DatasetManifest
from ai_factory.core.hashing import sha256_file
from data.reports.cards import pack_card_text


Predicate = Callable[[dict[str, Any]], bool]


def _is_calculus_like(record: dict[str, Any]) -> bool:
    topic = normalize_text(record.get("topic", "")).lower()
    return any(token in topic for token in ("calculus", "differential equations"))


DEFAULT_PACK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "core_train_mix": {
        "description": "Primary mixed-source training corpus across all supported math reasoning families.",
        "predicate": lambda record: record.get("dataset_split") in {"train", "eval", "test", "benchmark", "unspecified"},
    },
    "calculus_hard_pack": {
        "description": "Hard and olympiad calculus-focused examples for specialist adaptation.",
        "predicate": lambda record: _is_calculus_like(record) and record.get("difficulty") in {"hard", "olympiad"},
    },
    "olympiad_reasoning_pack": {
        "description": "Proof-style olympiad reasoning examples spanning algebra, number theory, and combinatorics.",
        "predicate": lambda record: record.get("reasoning_style") == "proof" or "olympiad" in (record.get("tags") or []),
    },
    "failure_replay_pack": {
        "description": "Failure-driven replay corpus collected from previous evaluations.",
        "predicate": lambda record: bool(record.get("failure_case")),
    },
    "verification_pack": {
        "description": "Examples with typed step checks suited for verifier-assisted training and evaluation.",
        "predicate": lambda record: bool(record.get("step_checks")),
    },
    "benchmark_holdout_pack": {
        "description": "Held-out benchmark/test slice reserved for evaluation and contamination checks.",
        "predicate": lambda record: record.get("dataset_split") in {"test", "benchmark"},
    },
}


def build_derived_packs(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    build: DatasetBuildInfo,
    pack_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = pack_ids or list(DEFAULT_PACK_DEFINITIONS)
    results: list[dict[str, Any]] = []
    for pack_id in selected_ids:
        definition = DEFAULT_PACK_DEFINITIONS[pack_id]
        predicate: Predicate = definition["predicate"]
        pack_records = [dict(record, pack_id=pack_id) for record in records if predicate(record)]
        pack_dir = base_dir / pack_id
        pack_dir.mkdir(parents=True, exist_ok=True)
        records_path = pack_dir / "records.jsonl"
        write_jsonl(records_path, pack_records)
        card_path = pack_dir / "card.md"
        write_markdown(card_path, pack_card_text(pack_id, definition["description"], pack_records))
        manifest = DatasetManifest(
            manifest_type="pack",
            build=build,
            pack_id=pack_id,
            description=definition["description"],
            outputs=[
                DatasetFileInfo(
                    path=str(records_path),
                    sha256=sha256_file(records_path) if records_path.exists() else "",
                    size_bytes=records_path.stat().st_size if records_path.exists() else 0,
                    num_rows=len(pack_records),
                )
            ],
            stats={
                "num_records": len(pack_records),
            },
            metadata={"card_path": str(card_path)},
        )
        manifest_path = pack_dir / "manifest.json"
        write_json(manifest_path, manifest.model_dump())
        results.append(
            {
                "id": pack_id,
                "description": definition["description"],
                "num_rows": len(pack_records),
                "size_bytes": records_path.stat().st_size if records_path.exists() else 0,
                "path": str(records_path),
            }
        )
    return results
