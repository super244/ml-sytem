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
    parser.add_argument("--custom-total-size-gb", type=float, default=None)
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


def _resolve_dataset_targets(
    specs: list[DatasetSpec], config: dict[str, Any], args: argparse.Namespace
) -> dict[str, int]:
    explicit_targets = {}
    for spec in specs:
        if spec.target_size_bytes is not None:
            explicit_targets[spec.id] = int(spec.target_size_bytes)
        elif spec.target_size_mb is not None:
            explicit_targets[spec.id] = int(spec.target_size_mb * 1024 * 1024)
    if explicit_targets:
        fallback_bytes = int((args.target_size_mb or config.get("target_size_mb", 3.1)) * 1024 * 1024)
        return {spec.id: explicit_targets.get(spec.id, fallback_bytes) for spec in specs}

    total_size_gb = args.custom_total_size_gb
    if total_size_gb is None:
        total_size_gb = config.get("custom_total_size_gb")
    if total_size_gb is not None:
        total_size_bytes = int(float(total_size_gb) * 1024 * 1024 * 1024)
    else:
        total_size_mb = config.get("custom_total_size_mb")
        total_size_bytes = int(float(total_size_mb) * 1024 * 1024) if total_size_mb is not None else None

    if total_size_bytes is None:
        legacy_bytes = int((args.target_size_mb or config.get("target_size_mb", 3.1)) * 1024 * 1024)
        return {spec.id: legacy_bytes for spec in specs}

    weights = [float(spec.target_share or 1.0) for spec in specs]
    total_weight = sum(weights) or float(len(specs) or 1)
    targets: dict[str, int] = {}
    allocated = 0
    for index, spec in enumerate(specs):
        if index == len(specs) - 1:
            target_bytes = total_size_bytes - allocated
        else:
            target_bytes = max(1, int(round(total_size_bytes * (weights[index] / total_weight))))
            allocated += target_bytes
        targets[spec.id] = target_bytes
    return targets


def write_catalog(path: Path, datasets: list[dict[str, Any]]) -> None:
    kind_counts = {}
    family_counts = {}
    topic_counts = {}
    for item in datasets:
        kind = str(item.get("kind", "unknown"))
        family = str(item.get("family", "unknown"))
        topic = str(item.get("topic", "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    summary = {
        "num_datasets": len(datasets),
        "custom_datasets": kind_counts.get("custom", 0),
        "public_datasets": kind_counts.get("public", 0),
        "total_bytes": sum(item.get("size_bytes", 0) for item in datasets),
        "total_rows": sum(item.get("num_rows", 0) for item in datasets),
        "profile_summary": {
            "kind_counts": kind_counts,
            "family_counts": family_counts,
            "topic_counts": topic_counts,
            "top_kinds": sorted(kind_counts.items(), key=lambda item: item[1], reverse=True)[:5],
            "top_families": sorted(family_counts.items(), key=lambda item: item[1], reverse=True)[:5],
            "top_topics": sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)[:5],
        },
    }
    payload = {"generated_at": "local-build", "summary": summary, "datasets": datasets}
    write_json(path, payload)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 42)

    output_dir = Path(config.get("output_dir", "data/custom"))
    catalog_path = Path(config.get("catalog_path", "data/catalog.json"))
    registry_path = str(config.get("public_registry_path", "data/public/registry.yaml"))
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [DatasetSpec(**payload) for payload in config.get("dataset_specs", [])]
    target_sizes = _resolve_dataset_targets(specs, config, args)

    datasets: list[dict[str, Any]] = []
    repo_root = Path(__file__).resolve().parents[2]
    for offset, spec in enumerate(specs):
        target_size_bytes = target_sizes[spec.id]
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
            stats={
                "num_rows": len(records),
                "topic": spec.topic,
                "family": spec.family,
                "profile_summary": entry["profile_summary"],
            },
            metadata={
                "card_path": str(output_path.with_suffix(".md")),
                "profile_summary": entry["profile_summary"],
            },
        )
        write_json(output_path.with_suffix(".manifest.json"), manifest.model_dump())
        print(
            f"Generated {entry['id']} with {entry['num_rows']} rows at {entry['path']} "
            f"(target={target_size_bytes} bytes)"
        )

    datasets.extend(_load_public_entries(registry_path))
    write_catalog(catalog_path, datasets)
    print(f"Wrote dataset catalog to {catalog_path}")


if __name__ == "__main__":
    main()
