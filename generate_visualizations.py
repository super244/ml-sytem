#!/usr/bin/env python3
"""Ultimate multifunction visualization generator for AI-Factory.

This script consolidates all prior visualization scripts into one entrypoint:
- Aggregate analytics across all runs/evaluations (`aggregate`)
- Single-run dashboard generation (`run`)
- Evaluation summary comparison charts (`eval`)
- Combined execution (`all`, default)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from graph_generation.loader import load_training_run

DEFAULT_OUTPUT_DIR = Path("evaluation/results/visualizations")
RUNS_ROOT = Path("artifacts/runs")
EVAL_RESULTS_ROOT = Path("evaluation/results")
IGNORED_EVAL_DIRS = {"visualizations"}
TOP_RUNS = 6


@dataclass
class TrainingRun:
    run_id: str
    run_name: str
    profile_name: str
    created_at: datetime
    path: Path
    final_loss: float | None
    loss_history: list[float]
    train_runtime_min: float | None
    train_samples_per_second: float | None
    train_steps_per_second: float | None
    eval_loss: float | None
    eval_runtime: float | None
    epoch: float | None
    learning_rate: float | None
    grad_accum: int | None
    total_parameters: int | None


@dataclass
class EvaluationEntry:
    label: str
    accuracy: float | None
    avg_latency: float | None
    avg_quality: float | None
    formatting_failure_rate: float | None
    step_correctness: float | None
    run_name: str
    side: str


def safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_training_metrics(log_path: Path) -> dict[str, Any]:
    result = {
        "loss_history": [],
        "final_loss": None,
        "train_runtime": None,
        "train_samples_per_second": None,
        "train_steps_per_second": None,
    }
    if not log_path.exists():
        return result
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        loss_value = None
        if isinstance(row.get("train_loss"), (int, float)):
            loss_value = float(row["train_loss"])
        elif isinstance(row.get("loss"), (int, float)):
            loss_value = float(row["loss"])
        if loss_value is not None:
            result["loss_history"].append(loss_value)
            result["final_loss"] = loss_value
        if row.get("train_runtime") is not None:
            result["train_runtime"] = row["train_runtime"]
        if row.get("train_samples_per_second") is not None:
            result["train_samples_per_second"] = row["train_samples_per_second"]
        if row.get("train_steps_per_second") is not None:
            result["train_steps_per_second"] = row["train_steps_per_second"]
    return result


def collect_training_runs() -> list[TrainingRun]:
    runs: list[TrainingRun] = []
    if not RUNS_ROOT.exists():
        return runs

    for candidate in sorted(RUNS_ROOT.iterdir()):
        if not candidate.is_dir():
            continue
        manifest = safe_load_json(candidate / "manifests" / "run_manifest.json")
        if not manifest:
            continue
        metrics_data = parse_training_metrics(candidate / "logs" / "training_metrics.jsonl")
        config_report = safe_load_json(candidate / "manifests" / "config_report.json") or {}
        training_cfg = config_report.get("training") or {}
        metrics_json = safe_load_json(candidate / "metrics" / "metrics.json") or {}
        model_report = safe_load_json(candidate / "metrics" / "model_report.json") or {}
        created_at = parse_iso_datetime(manifest.get("created_at")) or datetime.fromtimestamp(
            candidate.stat().st_mtime
        )
        runs.append(
            TrainingRun(
                run_id=manifest.get("run_id", candidate.name),
                run_name=manifest.get("run_name", candidate.name),
                profile_name=manifest.get("profile_name", ""),
                created_at=created_at,
                path=candidate,
                final_loss=metrics_data["final_loss"],
                loss_history=metrics_data["loss_history"],
                train_runtime_min=(metrics_data["train_runtime"] / 60.0 if metrics_data["train_runtime"] else None),
                train_samples_per_second=metrics_data["train_samples_per_second"],
                train_steps_per_second=metrics_data["train_steps_per_second"],
                eval_loss=metrics_json.get("eval_loss"),
                eval_runtime=metrics_json.get("eval_runtime"),
                epoch=metrics_json.get("epoch"),
                learning_rate=training_cfg.get("learning_rate"),
                grad_accum=training_cfg.get("gradient_accumulation_steps"),
                total_parameters=model_report.get("total_parameters"),
            )
        )
    return runs


def collect_evaluation_entries() -> list[EvaluationEntry]:
    entries: list[EvaluationEntry] = []
    if not EVAL_RESULTS_ROOT.exists():
        return entries
    for summary_path in sorted(EVAL_RESULTS_ROOT.rglob("summary.json")):
        candidate = summary_path.parent
        if any(part in IGNORED_EVAL_DIRS for part in candidate.parts):
            continue
        summary = safe_load_json(summary_path)
        if not summary or not isinstance(summary.get("primary"), dict):
            continue
        labels = summary.get("labels") or {}
        run_name = str(candidate.relative_to(EVAL_RESULTS_ROOT))
        for side in ("primary", "secondary"):
            section = summary.get(side)
            if not isinstance(section, dict):
                continue
            entries.append(
                EvaluationEntry(
                    label=labels.get(side) or f"{candidate.name}-{side}",
                    accuracy=section.get("accuracy"),
                    avg_latency=section.get("avg_latency_s"),
                    avg_quality=section.get("avg_quality_score"),
                    formatting_failure_rate=section.get("formatting_failure_rate"),
                    step_correctness=section.get("step_correctness"),
                    run_name=run_name,
                    side=side,
                )
            )
    return entries


def friendly_label(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split()).title()


def normalize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() else " " for ch in value).lower()


def accuracy_for_run(run: TrainingRun, entries: Iterable[EvaluationEntry]) -> float | None:
    needle = normalize_label(run.run_name)
    candidates: list[EvaluationEntry] = []
    for entry in entries:
        normalized = normalize_label(entry.label)
        if needle and (needle in normalized or normalized in needle):
            candidates.append(entry)
    if not candidates:
        return None
    return max((entry.accuracy or 0.0) for entry in candidates)


def plot_aggregate_charts(output_dir: Path, runs: list[TrainingRun], entries: list[EvaluationEntry]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = sorted([run for run in runs if run.final_loss is not None], key=lambda run: run.final_loss or 0.0)[:TOP_RUNS]
    if selected:
        labels = [friendly_label(run.run_name) for run in selected]
        losses = [run.final_loss or 0.0 for run in selected]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(labels, losses, color=plt.cm.viridis(np.linspace(0, 0.8, len(selected))))
        for bar, loss in zip(bars, losses, strict=True):
            ax.text(loss + max(losses) * 0.01, bar.get_y() + bar.get_height() / 2, f"{loss:.4f}", va="center")
        ax.set_xlabel("Final Loss (lower is better)")
        ax.set_title("Final Loss Comparison")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "01_aggregate_final_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if entries:
        usable = [entry for entry in entries if entry.accuracy is not None]
        if usable:
            usable.sort(key=lambda item: item.accuracy or 0.0, reverse=True)
            labels = [item.label for item in usable[:12]]
            accs = [(item.accuracy or 0.0) * 100 for item in usable[:12]]
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(range(len(labels)), accs, marker="o", linewidth=2.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Top Evaluation Accuracies")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "02_aggregate_accuracy_line.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if runs:
        timeline = sorted([run for run in runs if run.final_loss is not None], key=lambda run: run.created_at)
        if timeline:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot([r.created_at for r in timeline], [r.final_loss or 0.0 for r in timeline], "o-")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            ax.set_title("Run Timeline vs Final Loss")
            ax.set_ylabel("Final Loss")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "03_aggregate_timeline.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if runs and entries:
        radar_runs = [run for run in runs if run.total_parameters and run.loss_history][:3]
        if radar_runs:
            categories = ["Accuracy", "Speed", "LossEff", "EvalStability", "Size"]
            matrix: list[list[float]] = []
            for run in radar_runs:
                matrix.append(
                    [
                        accuracy_for_run(run, entries) or 0.0,
                        run.train_steps_per_second or run.train_samples_per_second or 0.0,
                        1 / (run.final_loss or 1.0),
                        1 / (run.eval_loss or 1.0),
                        (run.total_parameters or 0) / 1_000_000_000,
                    ]
                )
            values = np.array(matrix, dtype=float)
            mins = values.min(axis=0)
            spans = values.max(axis=0) - mins
            spans[spans == 0] = 1
            norm = (values - mins) / spans
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
            for i, run in enumerate(radar_runs):
                row = norm[i].tolist() + [norm[i][0]]
                ax.plot(angles, row, linewidth=2, label=friendly_label(run.run_name))
                ax.fill(angles, row, alpha=0.2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.set_title("Run Radar Comparison (normalized)")
            plt.tight_layout()
            plt.savefig(output_dir / "04_aggregate_radar.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def plot_run_dashboard(run_dir: Path, output_dir: Path, style: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_training_run(run_dir)
    df = bundle.metrics_df.copy()

    if "loss" not in df.columns:
        raise ValueError("training_metrics.jsonl does not contain 'loss' column")

    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)
    if "learning_rate" not in df.columns:
        df["learning_rate"] = 0.0
    if "grad_norm" not in df.columns:
        df["grad_norm"] = 0.0

    df["loss_smooth"] = df["loss"].rolling(window=50, min_periods=1).mean()
    df["perplexity"] = np.exp(df["loss"])

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df["epoch"], df["loss"], alpha=0.25, linewidth=1, label="Raw loss")
    ax1.plot(df["epoch"], df["loss_smooth"], linewidth=2.5, label="Smoothed loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df["epoch"], df["perplexity"], color="tab:orange", linewidth=2.2)
    ax2.set_title("Perplexity")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(df["epoch"], df["learning_rate"], color="tab:green", linewidth=2.2)
    ax3.set_title("Learning Rate")
    ax3.set_yscale("log")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(df["epoch"], df["grad_norm"], color="tab:red", linewidth=2.2)
    ax4.set_title("Gradient Norm")
    ax4.set_yscale("log")
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[2, :])
    difficulties = ["easy", "medium", "hard", "olympiad"]
    train_diff = [bundle.dataset_data["train"]["stats"]["difficulty_counts"].get(d, 0) for d in difficulties]
    eval_diff = [bundle.dataset_data["eval"]["stats"]["difficulty_counts"].get(d, 0) for d in difficulties]
    x = np.arange(len(difficulties))
    w = 0.35
    ax5.bar(x - w / 2, train_diff, w, label="Train")
    ax5.bar(x + w / 2, eval_diff, w, label="Eval")
    ax5.set_xticks(x)
    ax5.set_xticklabels([d.title() for d in difficulties])
    ax5.set_title("Dataset Difficulty Distribution")
    ax5.grid(axis="y", alpha=0.3)
    ax5.legend()

    if style in {"advanced", "ultra"}:
        for ax in (ax1, ax2, ax3, ax4, ax5):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Run Dashboard: {bundle.run_dir.name} | eval_loss={bundle.final_metrics.get('eval_loss', 'n/a')}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"run_dashboard_{style}.png", dpi=220 if style == "ultra" else 170, bbox_inches="tight")
    plt.close(fig)


def plot_eval_summary_charts(output_dir: Path, eval_dirs: dict[str, Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    for label, directory in eval_dirs.items():
        payload = safe_load_json(directory / "summary.json")
        if payload and isinstance(payload.get("primary"), dict):
            summaries[label] = payload

    if not summaries:
        print("No evaluation summaries found for eval mode.")
        return

    models = list(summaries.keys())
    primary_acc = [summaries[name]["primary"].get("accuracy", 0.0) * 100 for name in models]
    secondary_acc = [summaries[name]["secondary"].get("accuracy", 0.0) * 100 for name in models]
    delta_acc = [summaries[name].get("delta_accuracy", 0.0) * 100 for name in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - width / 2, primary_acc, width, label="Primary", color="#2ecc71")
    ax.bar(x + width / 2, secondary_acc, width, label="Secondary", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Evaluation Accuracy Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eval_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#2ecc71" if value > 0 else "#e74c3c" for value in delta_acc]
    bars = ax.barh(models, delta_acc, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Delta Accuracy (%)")
    ax.set_title("Primary vs Secondary Accuracy Delta")
    ax.grid(axis="x", alpha=0.3)
    for bar, value in zip(bars, delta_acc, strict=True):
        ax.text(value + (0.3 if value >= 0 else -0.3), bar.get_y() + bar.get_height() / 2, f"{value:+.2f}%", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "eval_delta_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def default_eval_dirs() -> dict[str, Path]:
    return {
        "Metal Shaders": Path("evaluation/results/accuracy_metal_shaders"),
        "Metal Optimized": Path("evaluation/results/metal_optimized"),
        "Accuracy Perfect": Path("evaluation/results/accuracy_perfect"),
        "Accuracy Ultimate": Path("evaluation/results/accuracy_ultimate"),
        "Accuracy Hardened": Path("evaluation/results/accuracy_hardened"),
        "Formatting Focus": Path("evaluation/results/formatting_focus_vs_base"),
    }


def parse_eval_map(pairs: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --eval-map value: {pair!r}; expected Label=path")
        label, raw_path = pair.split("=", 1)
        mapping[label.strip()] = Path(raw_path).expanduser().resolve()
    return mapping


def run_aggregate(output_dir: Path) -> None:
    runs = collect_training_runs()
    entries = collect_evaluation_entries()
    if not runs:
        print("No training runs found under artifacts/runs; skipping aggregate mode.")
        return
    plot_aggregate_charts(output_dir, runs, entries)
    summary_path = output_dir / "SUMMARY.txt"
    summary_path.write_text(
        "\n".join(
            [
                "AI-Factory Ultimate Visualization Summary",
                f"Generated At: {datetime.now().isoformat(timespec='seconds')}",
                f"Training Runs Scanned: {len(runs)}",
                f"Evaluation Entries Scanned: {len(entries)}",
            ]
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultimate multifunction visualization generator")
    parser.add_argument("--mode", choices=["all", "aggregate", "run", "eval"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=None, help="Path to a specific artifacts/runs/<run> directory")
    parser.add_argument("--style", choices=["standard", "advanced", "ultra"], default="ultra")
    parser.add_argument(
        "--eval-map",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional evaluation summary mapping; may be repeated",
    )
    return parser


def main() -> None:
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    if args.mode in {"all", "aggregate"}:
        aggregate_out = output_dir / "aggregate" if args.mode == "all" else output_dir
        run_aggregate(aggregate_out)

    if args.mode in {"all", "run"}:
        if args.run_dir is None:
            latest_runs = sorted([p for p in RUNS_ROOT.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
            if not latest_runs:
                raise SystemExit("No runs available for run mode; pass --run-dir explicitly.")
            run_dir = latest_runs[-1]
        else:
            run_dir = args.run_dir.expanduser().resolve()
        run_out = output_dir / "run" if args.mode == "all" else output_dir
        plot_run_dashboard(run_dir, run_out, args.style)

    if args.mode in {"all", "eval"}:
        eval_out = output_dir / "eval" if args.mode == "all" else output_dir
        eval_dirs = parse_eval_map(args.eval_map) if args.eval_map else default_eval_dirs()
        plot_eval_summary_charts(eval_out, eval_dirs)

    print(f"Visualization generation complete. Output root: {output_dir}")


if __name__ == "__main__":
    main()
