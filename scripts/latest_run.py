from __future__ import annotations

import argparse

from common import emit_payload, repo_root

from ai_factory.core.discovery import latest_training_run, list_training_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact summary for the latest Atlas training run.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = list_training_runs(str(repo_root() / "artifacts"))
    latest = latest_training_run(runs)
    if latest is None:
        emit_payload({"status": "no_runs_found"}, as_json=args.json)
        return
    payload = {
        "run_name": latest.get("run_name"),
        "profile_name": latest.get("profile_name"),
        "base_model": latest.get("base_model"),
        "output_dir": latest.get("output_dir"),
        "trainable_ratio": latest.get("model_report", {}).get("trainable_ratio"),
        "eval_loss": latest.get("metrics", {}).get("eval_loss"),
    }
    emit_payload(payload, as_json=args.json)


if __name__ == "__main__":
    main()
