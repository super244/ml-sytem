"""
Module for preparing datasets for the AI-Factory training pipeline.

This script acts as the entry point for normalizing, processing, and packaging
math reasoning data into a structured corpus suitable for training models.
It supports overriding configurations via CLI arguments and utilizes the
corpus_builder backend to construct the final datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.builders.corpus_builder import build_corpus, load_processing_config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset preparation.

    Returns:
        argparse.Namespace: The parsed command-line arguments containing paths
        for config, sources, failure logs, and output directories.
    """
    parser = argparse.ArgumentParser(description="Normalize and package math reasoning data.")
    parser.add_argument("--config", type=str, default="data/configs/processing.yaml")
    parser.add_argument("--source", action="append", default=None, help="Extra input file(s), URLs, or dataset specs.")
    parser.add_argument("--failure-log", action="append", default=None, help="Extra failure log(s).")
    parser.add_argument(
        "--contamination-source",
        action="append",
        default=None,
        help="Holdout/benchmark file(s) for contamination checks.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--source-load-workers", type=int, default=None, help="Concurrent source loader workers.")
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for dataset preparation.

    Loads the processing configuration, applies CLI argument overrides,
    and invokes the corpus builder to process and package the datasets.
    Finally, prints the build results as a JSON string and logs the execution time.
    """
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
    if args.source_load_workers is not None:
        config.source_load_workers = args.source_load_workers
    start_time = time.time()
    result = build_corpus(config, args.config)
    print(json.dumps(result, indent=2))
    print(
        f"[Titan Performance Engine] Dataset processing accelerated and wait time reduced. Finished in {time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()
