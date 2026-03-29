from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.artifacts import current_git_sha, sha256_file, sha256_text, write_json, write_jsonl
from ai_factory.core.io import read_jsonl
from ai_factory.core.schemas import DatasetBuildInfo, DatasetFileInfo, DatasetManifest
from data.adapters.base import load_public_registry
from data.synthesis import DatasetSpec, build_catalog_entry, generate_records
from data.synthesis.registry import write_dataset_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sizeable calculus and olympiad datasets for the repo.")
    parser.add_argument("--config", default="data/configs/generation.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-size-mb", type=float, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _build_file_info(path: Path) -> DatasetFileInfo:
    num_rows = len(read_jsonl(path)) if path.exists() and path.suffix == ".jsonl" else 0
    return DatasetFileInfo(
        path=str(path),
        sha256=sha256_file(path) if path.exists() else "",
        size_bytes=path.stat().st_size if path.exists() else 0,
        num_rows=num_rows,
    )


def _load_public_entries(registry_path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in load_public_registry(registry_path):
        output_path = Path(item.output_file)
        rows = read_jsonl(output_path)[:5] if output_path.exists() else []
        entries.append(
            {
                "id": item.id,
                "title": item.title,
                "kind": "public",
                "family": item.dataset_family,
                "topic": item.expected_topic,
                "reasoning_style": item.reasoning_style,
                "path": str(output_path),
                "num_rows": len(read_jsonl(output_path)) if output_path.exists() else 0,
                "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
                "description": item.notes,
                "usage": item.usage,
                "default_weight": item.default_weight,
                "benchmark_tags": item.benchmark_tags,
                "preview_examples": [
                    {
                        "id": row.get("id"),
                        "question": row.get("question"),
                        "difficulty": row.get("difficulty"),
                        "topic": row.get("topic"),
                        "final_answer": row.get("final_answer"),
                    }
                    for row in rows
                ],
            }
        )
    return entries


def write_catalog(path: Path, datasets: list[dict[str, Any]]) -> None:
    summary = {
        "num_datasets": len(datasets),
        "custom_datasets": sum(1 for item in datasets if item["kind"] == "custom"),
        "public_datasets": sum(1 for item in datasets if item["kind"] == "public"),
        "total_bytes": sum(item.get("size_bytes", 0) for item in datasets),
        "total_rows": sum(item.get("num_rows", 0) for item in datasets),
    }
    payload = {"generated_at": "local-build", "summary": summary, "datasets": datasets}
    write_json(path, payload)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    target_size_mb = args.target_size_mb or config.get("target_size_mb", 3.1)
    target_size_bytes = int(target_size_mb * 1024 * 1024)

    output_dir = Path(config.get("output_dir", "data/custom"))
    catalog_path = Path(config.get("catalog_path", "data/catalog.json"))
    registry_path = str(config.get("public_registry_path", "data/public/registry.yaml"))
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[dict[str, Any]] = []
    repo_root = Path(__file__).resolve().parents[2]
    for offset, spec_payload in enumerate(config.get("dataset_specs", [])):
        spec = DatasetSpec(**spec_payload)
        records = generate_records(spec, target_size_bytes=target_size_bytes, seed=seed + offset * 101)
        output_path = output_dir / f"{spec.id}.jsonl"
        write_jsonl(output_path, records)
        entry = build_catalog_entry(spec, output_path, records)
        datasets.append(entry)
        write_dataset_card(entry, output_path.with_suffix(".md"))
        manifest = DatasetManifest(
            manifest_type="dataset",
            build=DatasetBuildInfo(
                build_id=f"{spec.id}-{seed + offset * 101}",
                git_sha=current_git_sha(repo_root),
                config_path=args.config,
                config_sha256=sha256_text(Path(args.config).read_text()),
                seed=seed + offset * 101,
            ),
            pack_id=spec.id,
            description=" / ".join(spec.pedagogical_focus),
            outputs=[_build_file_info(output_path)],
            stats={"num_rows": len(records), "topic": spec.topic, "family": spec.family},
            metadata={"card_path": str(output_path.with_suffix(".md"))},
        )
        write_json(output_path.with_suffix(".manifest.json"), manifest.model_dump())
        print(f"Generated {entry['id']} with {entry['num_rows']} rows at {entry['path']}")

    datasets.extend(_load_public_entries(registry_path))
    write_catalog(catalog_path, datasets)
    print(f"Wrote dataset catalog to {catalog_path}")


if __name__ == "__main__":
    main()
