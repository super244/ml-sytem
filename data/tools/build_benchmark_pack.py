from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.io import read_jsonl
from ai_factory.core.schemas import DatasetBuildInfo
from data.builders.pack_registry import build_derived_packs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build derived benchmark/verification packs from a processed dataset.")
    parser.add_argument("--input", default="data/processed/normalized_all.jsonl")
    parser.add_argument("--output-dir", default="data/processed/packs")
    parser.add_argument("--pack-id", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    summaries = build_derived_packs(
        rows,
        args.output_dir,
        build=DatasetBuildInfo(build_id="manual-pack-build"),
        pack_ids=args.pack_id,
    )
    print(json.dumps({"packs": summaries}, indent=2))


if __name__ == "__main__":
    main()
