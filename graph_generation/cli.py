from __future__ import annotations

import argparse
import os
from pathlib import Path

from graph_generation.loader import REPO_ROOT


def default_example_run_dir() -> Path:
    """Example run used by the bundled visualization scripts when no path is given."""
    return REPO_ROOT / "artifacts/runs/accuracy_ultimate_95_plus-20260408-174421"


def resolve_run_dir(ns: argparse.Namespace) -> Path:
    if getattr(ns, "run_dir", None) is not None:
        return Path(ns.run_dir).expanduser().resolve()
    env = os.environ.get("AI_FACTORY_RUN_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return default_example_run_dir()


def add_run_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Training run directory containing metrics/ and logs/. "
        "Default: $AI_FACTORY_RUN_DIR or the example run under artifacts/runs/.",
    )


def resolve_output_dir(ns: argparse.Namespace, *, fallback: Path) -> Path:
    out = getattr(ns, "output_dir", None)
    if out is not None:
        return Path(out).expanduser().resolve()
    return fallback.resolve()
