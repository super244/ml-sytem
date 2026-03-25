from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_factory.core.artifacts import ensure_latest_pointer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a run's final adapter into the model registry.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    adapter_dir = run_dir / "published" / "final_adapter"
    if not adapter_dir.exists():
        raise SystemExit(f"No final adapter found at {adapter_dir}")
    model_dir = Path("artifacts/models") / args.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    ensure_latest_pointer(model_dir / "LATEST_MODEL.json", adapter_dir, metadata={"run_dir": str(run_dir)})
    print(json.dumps({"model_name": args.model_name, "adapter_dir": str(adapter_dir)}, indent=2))


if __name__ == "__main__":
    main()
