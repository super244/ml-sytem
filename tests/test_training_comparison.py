from __future__ import annotations

import json

from training.src.comparison import compare_runs, format_comparison_report, load_run_summary


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_run(base, *, run_name, eval_loss, accuracy):
    _write_json(
        base / "manifests" / "run_manifest.json",
        {
            "run_id": f"{run_name}-id",
            "run_name": run_name,
            "profile_name": "baseline",
            "base_model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "model_name": "atlas",
            "metadata": {"mode": "train"},
        },
    )
    _write_json(
        base / "metrics" / "metrics.json",
        {
            "eval_loss": eval_loss,
            "eval_accuracy": accuracy,
        },
    )
    _write_json(base / "metrics" / "dataset_report.json", {"train": {"num_rows": 10}})
    _write_json(base / "metrics" / "tracking_summary.json", {"status": "completed"})


def test_compare_runs_reports_winner_and_markdown(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, run_name="left-run", eval_loss=0.42, accuracy=0.81)
    _write_run(right, run_name="right-run", eval_loss=0.31, accuracy=0.84)

    report = compare_runs(left, right, primary_metric="eval_loss")

    assert report["winner"] == "right"
    assert report["delta"]["eval_loss"]["delta"] == -0.11
    assert report["shared_metrics"] == ["eval_accuracy", "eval_loss"]
    assert load_run_summary(left)["status"] == "completed"

    markdown = format_comparison_report(report)
    assert "# Run Comparison" in markdown
    assert "Metric Deltas" in markdown
