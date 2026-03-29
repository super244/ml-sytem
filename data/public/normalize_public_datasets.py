from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.artifacts import sha256_file
from ai_factory.core.io import write_json, write_jsonl, write_markdown
from ai_factory.core.schemas import DatasetBuildInfo, DatasetFileInfo, DatasetManifest
from data.adapters.base import iter_source_rows, load_public_registry, matches_filters, normalize_public_record
from data.quality.scoring import estimate_quality_score
from data.reports.cards import dataset_card_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize public math datasets into the repo schema.")
    parser.add_argument("--registry", default="data/public/registry.yaml")
    parser.add_argument("--dataset-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", default="data/raw/public")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = set(args.dataset_id or [])
    entries = [entry for entry in load_public_registry(args.registry) if not wanted or entry.id in wanted]
    if not entries:
        print("No datasets matched the requested ids.")
        return
    for entry in entries:
        normalized_rows: list[dict] = []
        for raw_row in iter_source_rows(entry, cache_dir=args.cache_dir):
            row = dict(raw_row)
            if not matches_filters(row, entry):
                continue
            normalized = normalize_public_record(row, entry)
            if normalized is None:
                continue
            normalized["quality_score"] = estimate_quality_score(normalized)
            normalized_rows.append(normalized)
            if args.limit and len(normalized_rows) >= args.limit:
                break
        output_path = Path(entry.output_file)
        write_jsonl(output_path, normalized_rows)
        entry_payload = {
            "id": entry.id,
            "title": entry.title,
            "kind": "public",
            "family": entry.dataset_family,
            "topic": entry.expected_topic,
            "reasoning_style": entry.reasoning_style,
            "path": str(output_path),
            "num_rows": len(normalized_rows),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "description": entry.notes,
            "usage": entry.usage,
            "default_weight": entry.default_weight,
            "benchmark_tags": entry.benchmark_tags,
            "preview_examples": [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "difficulty": row["difficulty"],
                    "topic": row["topic"],
                    "final_answer": row.get("final_answer"),
                }
                for row in normalized_rows[:5]
            ],
        }
        write_markdown(output_path.with_suffix(".md"), dataset_card_text(entry_payload))
        manifest = DatasetManifest(
            manifest_type="dataset",
            build=DatasetBuildInfo(build_id=f"public-{entry.id}", config_path=args.registry),
            pack_id=entry.id,
            description=entry.notes,
            outputs=[
                DatasetFileInfo(
                    path=str(output_path),
                    sha256=sha256_file(output_path) if output_path.exists() else "",
                    size_bytes=output_path.stat().st_size if output_path.exists() else 0,
                    num_rows=len(normalized_rows),
                )
            ],
            source_lineage=[
                row.get("lineage") for row in normalized_rows[:20]
            ],
            stats={"num_rows": len(normalized_rows)},
            metadata={
                "usage": entry.usage,
                "default_weight": entry.default_weight,
                "benchmark_tags": entry.benchmark_tags,
                "expected_topic": entry.expected_topic,
            },
        )
        write_json(output_path.with_suffix(".manifest.json"), manifest.model_dump())
        print(f"Wrote {len(normalized_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
