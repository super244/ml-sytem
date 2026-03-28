from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.builders.corpus_builder import build_corpus, load_processing_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and package math reasoning data.")
    parser.add_argument("--config", type=str, default="data/configs/processing.yaml")
    parser.add_argument("--source", action="append", default=None, help="Extra input file(s), URLs, or dataset specs.")
    parser.add_argument("--failure-log", action="append", default=None, help="Extra failure log(s).")
    parser.add_argument("--contamination-source", action="append", default=None, help="Holdout/benchmark file(s) for contamination checks.")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_processing_config(args.config)
    if args.source:
        config.sources = (config.sources or []) + args.source
    if args.failure_log:
        config.failure_logs = (config.failure_logs or []) + args.failure_log
    if args.contamination_source:
        config.contamination_sources = (config.contamination_sources or []) + args.contamination_source
    if args.output_dir:
        config.output_dir = args.output_dir
    result = build_corpus(config, args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
