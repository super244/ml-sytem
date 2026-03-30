from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.adapters.base import iter_source_rows, load_public_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public math datasets declared in the registry.")
    parser.add_argument("--registry", default="data/public/registry.yaml")
    parser.add_argument("--dataset-id", action="append", default=None, help="Optional dataset id(s) to download.")
    parser.add_argument("--cache-dir", default="data/raw/public")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_filter = set(args.dataset_id or [])
    datasets = [
        entry for entry in load_public_registry(args.registry) if not dataset_filter or entry.id in dataset_filter
    ]
    if not datasets:
        print("No datasets matched the requested ids.")
        return
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for entry in datasets:
        print(f"Downloading {entry.id} from {entry.path} split={entry.split}")
        dataset = iter_source_rows(entry, cache_dir=str(cache_dir))
        marker = cache_dir / f"{entry.id}.downloaded.txt"
        marker.write_text(f"Downloaded {entry.path} split={entry.split} with {len(dataset)} rows.\n")
        print(f"Saved cache marker to {marker}")


if __name__ == "__main__":
    main()
