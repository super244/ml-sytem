from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

from ai_factory.core.io import write_markdown
from evaluation.error_taxonomy import summarize_failure_taxonomy


def _mean(values: list[float | None]) -> float:
    filtered = [value for value in values if value is not None]
    return mean(filtered) if filtered else 0.0


def aggregate_metrics(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    entries = [result[key] for result in results]
    return {
        "accuracy": sum(1 for item in entries if item["correct"]) / len(entries) if entries else 0.0,
        "solve_rate": sum(1 for item in entries if item["solve"]) / len(entries) if entries else 0.0,
        "parse_rate": _mean([item["parse_rate"] for item in entries]),
        "step_correctness": _mean([item["step_correctness"] for item in entries]),
        "verifier_agreement_rate": sum(1 for item in entries if item["verifier_agreement"]) / len(entries) if entries else 0.0,
        "formatting_failure_rate": sum(1 for item in entries if item["formatting_failure"]) / len(entries) if entries else 0.0,
        "arithmetic_slip_rate": sum(1 for item in entries if item["arithmetic_slip"]) / len(entries) if entries else 0.0,
        "no_answer_rate": sum(1 for item in entries if item["no_answer"]) / len(entries) if entries else 0.0,
        "avg_latency_s": _mean([item["latency_s"] for item in entries]),
        "avg_prompt_tokens": _mean([item["prompt_tokens"] for item in entries]),
        "avg_completion_tokens": _mean([item["completion_tokens"] for item in entries]),
        "avg_estimated_cost_usd": _mean([item["estimated_cost_usd"] for item in entries]),
        "avg_candidate_agreement": _mean([item["candidate_agreement"] for item in entries]),
    }


def by_group(results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result.get(field, "unknown"))].append(result)
    return {
        group_name: {
            "primary": aggregate_metrics(group_results, "primary"),
            "secondary": aggregate_metrics(group_results, "secondary"),
        }
        for group_name, group_results in groups.items()
    }


def collect_win_cases(results: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    win_cases = [result for result in results if result["primary"]["correct"] and not result["secondary"]["correct"]]
    return win_cases[:limit]


def build_summary(results: list[dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    primary_metrics = aggregate_metrics(results, "primary")
    secondary_metrics = aggregate_metrics(results, "secondary")
    summary = {
        "num_examples": len(results),
        "labels": labels,
        "primary": primary_metrics,
        "secondary": secondary_metrics,
        "delta_accuracy": primary_metrics["accuracy"] - secondary_metrics["accuracy"],
        "delta_step_correctness": primary_metrics["step_correctness"] - secondary_metrics["step_correctness"],
        "delta_solve_rate": primary_metrics["solve_rate"] - secondary_metrics["solve_rate"],
        "by_topic": by_group(results, "topic"),
        "by_difficulty": by_group(results, "difficulty"),
        "by_source": by_group(results, "source"),
        "by_pack": by_group(results, "pack_id"),
        "by_generator_family": by_group(results, "generator_family"),
        "failure_taxonomy": summarize_failure_taxonomy(results),
        "win_cases": collect_win_cases(results),
    }
    return summary


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    primary_label = summary["labels"]["primary"]
    secondary_label = summary["labels"]["secondary"]
    lines = [
        "# Evaluation Report",
        "",
        f"- Examples: {summary['num_examples']}",
        f"- {primary_label} accuracy: {summary['primary']['accuracy']:.3f}",
        f"- {secondary_label} accuracy: {summary['secondary']['accuracy']:.3f}",
        f"- Accuracy delta: {summary['delta_accuracy']:+.3f}",
        f"- Solve-rate delta: {summary['delta_solve_rate']:+.3f}",
        f"- Step correctness delta: {summary['delta_step_correctness']:+.3f}",
        f"- {primary_label} avg latency: {summary['primary']['avg_latency_s']:.3f}s",
        f"- {secondary_label} avg latency: {summary['secondary']['avg_latency_s']:.3f}s",
        "",
        "## Failure Taxonomy",
        "",
        "```json",
        json.dumps(summary["failure_taxonomy"], indent=2),
        "```",
        "",
        "## Win Cases",
        "",
    ]
    if not summary["win_cases"]:
        lines.append("No primary-model win cases were found in this run.")
    else:
        for case in summary["win_cases"]:
            lines.extend(
                [
                    f"### {case['id']}",
                    case["question"],
                    "",
                    f"- Reference answer: `{case['reference_answer']}`",
                    f"- {primary_label}: `{case['primary']['final_answer']}`",
                    f"- {secondary_label}: `{case['secondary']['final_answer']}`",
                    "",
                ]
            )
    write_markdown(path, "\n".join(lines))
