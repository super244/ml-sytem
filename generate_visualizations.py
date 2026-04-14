#!/usr/bin/env python3
"""
Generates training and evaluation visualizations directly from the artifacts and evaluation results produced by ai-factory runs.

New workflow:
1. Read every run under `artifacts/runs`.
2. Collect loss/train metrics plus model metadata.
3. Join the run data with evaluations under `evaluation/results/*`.
4. Emit charts that dynamically reflect the freshest data instead of hard-coded numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_DIR = Path("evaluation/results/visualizations")
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


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
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
    with log_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
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
        learning_rate = training_cfg.get("learning_rate")
        grad_accum = training_cfg.get("gradient_accumulation_steps")
        metrics_json = safe_load_json(candidate / "metrics" / "metrics.json") or {}
        model_report = safe_load_json(candidate / "metrics" / "model_report.json") or {}
        created_at = parse_iso_datetime(manifest.get("created_at")) or datetime.fromtimestamp(
            candidate.stat().st_mtime
        )
        run = TrainingRun(
            run_id=manifest.get("run_id", candidate.name),
            run_name=manifest.get("run_name", candidate.name),
            profile_name=manifest.get("profile_name", ""),
            created_at=created_at,
            path=candidate,
            final_loss=metrics_data["final_loss"],
            loss_history=metrics_data["loss_history"],
            train_runtime_min=(
                metrics_data["train_runtime"] / 60.0 if metrics_data["train_runtime"] else None
            ),
            train_samples_per_second=metrics_data["train_samples_per_second"],
            train_steps_per_second=metrics_data["train_steps_per_second"],
            eval_loss=metrics_json.get("eval_loss"),
            eval_runtime=metrics_json.get("eval_runtime"),
            epoch=metrics_json.get("epoch"),
            learning_rate=learning_rate,
            grad_accum=grad_accum,
            total_parameters=model_report.get("total_parameters"),
        )
        runs.append(run)
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
        if not summary:
            continue
        labels = summary.get("labels") or {}
        if not isinstance(summary.get("primary"), dict):
            continue
        run_name = str(candidate.relative_to(EVAL_RESULTS_ROOT))
        for side in ("primary", "secondary"):
            section = summary.get(side)
            if not isinstance(section, dict):
                continue
            label = labels.get(side) or f"{candidate.name}-{side}"
            entries.append(
                EvaluationEntry(
                    label=label,
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
    cleaned = value.replace("_", " ").replace("-", " ")
    return " ".join(segment.capitalize() for segment in cleaned.split())


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


def plot_final_loss(runs: list[TrainingRun]) -> None:
    selected = sorted(
        [run for run in runs if run.final_loss is not None],
        key=lambda run: run.final_loss or float("inf"),
    )[:TOP_RUNS]
    if not selected:
        print("No runs contain a final loss; skipping final loss chart.")
        return
    labels = [friendly_label(run.run_name) for run in selected]
    losses = [run.final_loss or 0.0 for run in selected]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(selected)))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, losses, color=colors, edgecolor="black", linewidth=1.3)
    for bar, loss in zip(bars, losses):
        ax.text(
            loss + max(losses) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{loss:.2f}",
            va="center",
            fontweight="bold",
        )
    ax.set_xlabel("Final Loss (lower is better)", fontsize=12, fontweight="bold")
    ax.set_title("Final Loss Comparison Across Training Runs", fontsize=14, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_bar_chart_final_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Graph 1: Final loss comparison")


def plot_training_time(runs: list[TrainingRun]) -> None:
    selected = [
        run for run in runs if run.train_runtime_min is not None and run.train_runtime_min > 0
    ]
    selected = sorted(selected, key=lambda run: run.train_runtime_min)[:TOP_RUNS]
    if not selected:
        print("No runs report training runtime; skipping training time chart.")
        return
    labels = [friendly_label(run.run_name) for run in selected]
    durations = [run.train_runtime_min or 0.0 for run in selected]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, durations, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(selected))))
    for bar, duration in zip(bars, durations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            duration + max(durations) * 0.02,
            f"{duration:.1f}m",
            ha="center",
            fontweight="bold",
        )
    ax.set_ylabel("Training Time (minutes)", fontsize=12, fontweight="bold")
    ax.set_title("Total Training Time for Selected Runs", fontsize=14, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_bar_chart_training_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Graph 2: Training time comparison")


def plot_loss_curves(runs: list[TrainingRun]) -> None:
    selected = [
        run for run in runs if run.loss_history and len(run.loss_history) > 1
    ][:TOP_RUNS]
    if not selected:
        print("No loss histories found; skipping loss curve chart.")
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    for run in selected:
        steps = list(range(1, len(run.loss_history) + 1))
        ax.plot(
            steps,
            run.loss_history,
            label=friendly_label(run.run_name),
            linewidth=2,
            marker="o",
            markersize=4,
            alpha=0.8,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("Training Loss Curves (log scale)", fontsize=14, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_line_graph_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Graph 3: Loss curves")


def plot_accuracy_line(entries: list[EvaluationEntry]) -> None:
    usable = [entry for entry in entries if entry.accuracy is not None]
    if not usable:
        print("No accuracy entries found; skipping accuracy line graph.")
        return
    usable.sort(key=lambda entry: entry.accuracy or 0.0)
    labels = [entry.label for entry in usable]
    accuracies = [entry.accuracy or 0.0 for entry in usable]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(labels)), [acc * 100 for acc in accuracies], marker="o", linewidth=2.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Evaluation Accuracy per Benchmark Entry", fontsize=14, fontweight="bold", pad=10)
    ax.axhline(85, color="red", linestyle="--", linewidth=1.5, label="Target (85%)")
    ax.set_ylim(0, 100)
    for idx, acc in enumerate(accuracies):
        ax.annotate(f"{acc*100:.1f}%", (idx, acc * 100), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_line_graph_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Graph 4: Accuracy progression")


def plot_heatmap(runs: list[TrainingRun]) -> None:
    selected = [
        run for run in runs if run.final_loss is not None
    ][:TOP_RUNS]
    if not selected:
        print("Heatmap skipped because no runs have the required data.")
        return
    metrics_labels = ["Final Loss", "Train Time (min)", "Eval Loss", "Epochs", "LR (x1e3)", "Grad Accum"]
    data_matrix = []
    for run in selected:
        data_matrix.append(
            [
                run.final_loss or 0.0,
                run.train_runtime_min or 0.0,
                run.eval_loss or 0.0,
                run.epoch or 0.0,
                (run.learning_rate or 0.0) * 1_000,
                float(run.grad_accum or 0),
            ]
        )
    matrix = np.array(data_matrix, dtype=float)
    mins = matrix.min(axis=0)
    spans = matrix.max(axis=0) - mins
    spans[spans == 0] = 1
    normalized = (matrix - mins) / spans

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(normalized.T, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(selected)))
    ax.set_yticks(range(len(metrics_labels)))
    ax.set_xticklabels([friendly_label(run.run_name) for run in selected], fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(metrics_labels, fontsize=10)
    for col in range(matrix.shape[0]):
        for row in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[col, row]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="Normalized Score (0-1)")
    ax.set_title("Training Configuration & Performance Heatmap", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_heatmap_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Graph 5: Performance heatmap")


def plot_timeline(runs: list[TrainingRun]) -> None:
    selected = sorted(
        [run for run in runs if run.final_loss is not None],
        key=lambda run: run.created_at,
    )
    if not selected:
        print("No timeline data found; skipping timeline chart.")
        return
    dates = [run.created_at for run in selected]
    losses = [run.final_loss or 0.0 for run in selected]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(dates, losses, "o-", linewidth=2, markersize=8)
    for date, loss, run in zip(dates, losses, selected):
        ax.annotate(
            friendly_label(run.run_name),
            (date, loss),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Run Start Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Loss", fontsize=12, fontweight="bold")
    ax.set_title("Training Timeline: Loss Reduction Over Time", fontsize=14, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_timeline_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Graph 6: Training timeline")


def plot_quadrant(entries: list[EvaluationEntry]) -> None:
    data = [
        entry
        for entry in entries
        if entry.accuracy is not None and entry.avg_latency and entry.avg_latency > 0
    ]
    if not data:
        print("No evaluation entries with latency/accuracy; skipping quadrant chart.")
        return
    fig, ax = plt.subplots(figsize=(14, 10))
    for entry in data:
        speed = 1 / entry.avg_latency if entry.avg_latency else 0
        accuracy_pct = (entry.accuracy or 0.0) * 100
        size = max(60, (entry.avg_quality or 0.4) * 400)
        label_lower = entry.label.lower()
        if "atlas" in label_lower or "boss" in label_lower:
            color = "#2E8B57"
        elif "base" in label_lower:
            color = "#4682B4"
        else:
            color = "#CD5C5C"
        scatter = ax.scatter(speed, accuracy_pct, s=size, c=color, alpha=0.65, edgecolors="black", linewidth=1.2)
        ax.annotate(
            entry.label,
            (speed, accuracy_pct),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
    ideal = mpatches.Rectangle((0, 70), 0.3, 30, linewidth=2, edgecolor="#4169E1", facecolor="#4169E1", alpha=0.15)
    ax.add_patch(ideal)
    ax.text(0.15, 98, "Ideal Zone (Fast + Accurate)", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Inference Speed Proxy (1 / avg latency)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Speed vs Accuracy Quadrant", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    legend_elements = [
        mpatches.Patch(color="#2E8B57", label="Atlas Runs"),
        mpatches.Patch(color="#4682B4", label="Base / Reference"),
        mpatches.Patch(color="#CD5C5C", label="Other Benchmarks"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_quadrant_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Graph 7: Speed vs accuracy quadrant")


def plot_radar(runs: list[TrainingRun], entries: Iterable[EvaluationEntry]) -> None:
    radar_runs = [run for run in runs if run.total_parameters and run.loss_history][:3]
    if not radar_runs:
        print("Radar chart skipped because radar runs are missing required metrics.")
        return
    categories = ["Accuracy", "Speed", "Loss Efficiency", "Eval Stability", "Size"]
    metrics_data: list[list[float]] = []
    for run in radar_runs:
        accuracy = accuracy_for_run(run, entries) or 0.0
        speed = run.train_steps_per_second or run.train_samples_per_second or 0.0
        loss_eff = 1 / (run.final_loss or 1.0)
        eval_stability = 1 / (run.eval_loss or 1.0)
        size = (run.total_parameters or 0) / 1_000_000_000
        metrics_data.append([accuracy, speed, loss_eff, eval_stability, size])
    matrix = np.array(metrics_data, dtype=float)
    mins = matrix.min(axis=0)
    spans = matrix.max(axis=0) - mins
    spans[spans == 0] = 1
    normalized = (matrix - mins) / spans

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    for run_idx, run in enumerate(radar_runs):
        values = normalized[run_idx].tolist()
        values += values[:1]
        ax.plot(angles, values, label=friendly_label(run.run_name), linewidth=2)
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_title("Multi-dimensional Run Comparison (Normalized)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_radar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Graph 8: Radar comparison")


def write_summary(runs: list[TrainingRun], entries: list[EvaluationEntry]) -> None:
    summary_path = OUTPUT_DIR / "SUMMARY.txt"
    lines = [
        "AI-Factory Visualization Summary",
        "",
        f"Generated At: {datetime.now().isoformat(timespec='seconds')}",
        f"Training Runs Scanned: {len(runs)}",
        f"Evaluation Entries Scanned: {len(entries)}",
        f"Training Root: {RUNS_ROOT}",
        f"Evaluation Root: {EVAL_RESULTS_ROOT}",
        "",
        "Regeneration:",
        "1. Optional cache cleanup: python scripts/cleanup_training_cache.py",
        "2. Refresh charts: python generate_visualizations.py",
        "",
        "Charts:",
        "- 01_bar_chart_final_loss.png",
        "- 02_bar_chart_training_time.png",
        "- 03_line_graph_loss_curves.png",
        "- 04_line_graph_accuracy.png",
        "- 05_heatmap_performance.png",
        "- 06_timeline_training.png",
        "- 07_quadrant_model_comparison.png",
        "- 08_radar_comparison.png",
        "",
        "Notes:",
        "- Evaluation summaries are discovered recursively under evaluation/results.",
        "- Aggregated suite summaries without primary/secondary sections are ignored.",
    ]
    summary_path.write_text("\n".join(lines))
    print(f"✓ Summary: {summary_path}")


def main() -> None:
    ensure_output_dir()
    runs = collect_training_runs()
    entries = collect_evaluation_entries()
    if not runs:
        print("No training runs detected under artifacts/runs; aborting.")
        return
    plot_final_loss(runs)
    plot_training_time(runs)
    plot_loss_curves(runs)
    plot_accuracy_line(entries)
    plot_heatmap(runs)
    plot_timeline(runs)
    plot_quadrant(entries)
    plot_radar(runs, entries)
    write_summary(runs, entries)
    print(f"\nVisualizations have been written to {OUTPUT_DIR}")


if __name__ == "__main__":
    sns.set_style("darkgrid")
    sns.set_palette("husl")
    main()
