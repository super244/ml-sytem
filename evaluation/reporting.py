from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ai_factory.core.io import write_markdown
from evaluation.error_taxonomy import summarize_failure_taxonomy


def _mean(values: list[float | None]) -> float:
    filtered = [value for value in values if value is not None]
    return mean(filtered) if filtered else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metric_entries(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in results:
        entry = result.get(key)
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _collect_component_names(entries: list[dict[str, Any]]) -> set[str]:
    component_names: set[str] = set()
    for entry in entries:
        components = entry.get("metric_components")
        if isinstance(components, dict):
            component_names.update(str(name) for name in components.keys())
    return component_names


def _summarize_metric_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "num_examples": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "solve_count": 0,
            "accuracy": 0.0,
            "solve_rate": 0.0,
            "parse_rate": 0.0,
            "step_correctness": 0.0,
            "verifier_agreement_rate": 0.0,
            "formatting_failure_rate": 0.0,
            "arithmetic_slip_rate": 0.0,
            "no_answer_rate": 0.0,
            "avg_latency_s": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_estimated_cost_usd": 0.0,
            "avg_candidate_agreement": 0.0,
            "avg_quality_score": 0.0,
            "avg_prompt_to_completion_ratio": 0.0,
            "metric_components": {},
            "error_types": {},
        }

    component_names = _collect_component_names(entries)
    error_counts = Counter(str(item.get("error_type", "unknown")) for item in entries if not item.get("correct"))
    metrics = {
        "num_examples": len(entries),
        "correct_count": sum(1 for item in entries if item.get("correct")),
        "incorrect_count": sum(1 for item in entries if not item.get("correct")),
        "solve_count": sum(1 for item in entries if item.get("solve")),
        "accuracy": _ratio(sum(1 for item in entries if item.get("correct")), len(entries)),
        "solve_rate": _ratio(sum(1 for item in entries if item.get("solve")), len(entries)),
        "parse_rate": _mean([item.get("parse_rate") for item in entries]),
        "step_correctness": _mean([item.get("step_correctness") for item in entries]),
        "verifier_agreement_rate": _ratio(
            sum(1 for item in entries if item.get("verifier_agreement")),
            len(entries),
        ),
        "formatting_failure_rate": _ratio(
            sum(1 for item in entries if item.get("formatting_failure")),
            len(entries),
        ),
        "arithmetic_slip_rate": _ratio(
            sum(1 for item in entries if item.get("arithmetic_slip")),
            len(entries),
        ),
        "no_answer_rate": _ratio(sum(1 for item in entries if item.get("no_answer")), len(entries)),
        "avg_latency_s": _mean([item.get("latency_s") for item in entries]),
        "avg_prompt_tokens": _mean([item.get("prompt_tokens") for item in entries]),
        "avg_completion_tokens": _mean([item.get("completion_tokens") for item in entries]),
        "avg_estimated_cost_usd": _mean([item.get("estimated_cost_usd") for item in entries]),
        "avg_candidate_agreement": _mean([item.get("candidate_agreement") for item in entries]),
        "avg_quality_score": _mean([item.get("quality_score") for item in entries]),
        "avg_prompt_to_completion_ratio": _mean([item.get("prompt_to_completion_ratio") for item in entries]),
        "metric_components": {
            name: _mean(
                [
                    (item.get("metric_components") or {}).get(name)
                    for item in entries
                    if isinstance(item.get("metric_components"), dict)
                ]
            )
            for name in sorted(component_names)
        },
        "error_types": dict(sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))),
    }
    return metrics


def aggregate_metrics(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return _summarize_metric_entries(_metric_entries(results, key))


def summarize_scorecards(results: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_metric_entries(results)


def build_trend_series(
    results: list[dict[str, Any]], key: str | None = None, *, window_size: int | None = None
) -> dict[str, Any]:
    entries = _metric_entries(results, key) if key else results
    if not entries:
        return {"window_size": 0, "checkpoints": []}
    checkpoint_size = window_size or max(1, len(entries) // 10)
    checkpoints: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        running.append(entry)
        if index % checkpoint_size == 0 or index == len(entries):
            summary = _summarize_metric_entries(running)
            checkpoints.append(
                {
                    "example_index": index,
                    "num_examples": summary["num_examples"],
                    "accuracy": summary["accuracy"],
                    "solve_rate": summary["solve_rate"],
                    "step_correctness": summary["step_correctness"],
                    "quality_score": summary["avg_quality_score"],
                    "avg_latency_s": summary["avg_latency_s"],
                    "error_types": summary["error_types"],
                }
            )
    return {"window_size": checkpoint_size, "checkpoints": checkpoints}


def _group_metrics(results: list[dict[str, Any]], field: str, *, include_delta: bool = False) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result.get(field) or "unknown")].append(result)
    summary: dict[str, Any] = {}
    for group_name, group_results in groups.items():
        primary = aggregate_metrics(group_results, "primary")
        secondary = aggregate_metrics(group_results, "secondary")
        group_summary: dict[str, Any] = {
            "primary": primary,
            "secondary": secondary,
        }
        if include_delta:
            group_summary["delta_accuracy"] = primary["accuracy"] - secondary["accuracy"]
            group_summary["delta_solve_rate"] = primary["solve_rate"] - secondary["solve_rate"]
            group_summary["delta_step_correctness"] = primary["step_correctness"] - secondary["step_correctness"]
        summary[group_name] = group_summary
    return summary


def by_group(results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return _group_metrics(results, field)


def build_group_comparison(results: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    groups = by_group(results, field)
    for group_name, group_metrics in groups.items():
        primary = group_metrics["primary"]
        secondary = group_metrics["secondary"]
        comparisons.append(
            {
                "group": group_name,
                "num_examples": primary["num_examples"],
                "primary_accuracy": primary["accuracy"],
                "secondary_accuracy": secondary["accuracy"],
                "delta_accuracy": primary["accuracy"] - secondary["accuracy"],
                "primary_solve_rate": primary["solve_rate"],
                "secondary_solve_rate": secondary["solve_rate"],
                "delta_solve_rate": primary["solve_rate"] - secondary["solve_rate"],
                "primary_step_correctness": primary["step_correctness"],
                "secondary_step_correctness": secondary["step_correctness"],
                "delta_step_correctness": primary["step_correctness"] - secondary["step_correctness"],
            }
        )
    return sorted(
        comparisons,
        key=lambda item: (-item["delta_accuracy"], -item["num_examples"], item["group"]),
    )


def collect_comparison_cases(
    results: list[dict[str, Any]],
    *,
    winner_key: str,
    loser_key: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    cases = []
    for result in results:
        winner = result.get(winner_key)
        loser = result.get(loser_key)
        if not isinstance(winner, dict) or not isinstance(loser, dict):
            continue
        if winner.get("correct") and not loser.get("correct"):
            cases.append(result)
    return cases[:limit]


def collect_win_cases(results: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return collect_comparison_cases(results, winner_key="primary", loser_key="secondary", limit=limit)


def collect_regression_cases(results: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return collect_comparison_cases(results, winner_key="secondary", loser_key="primary", limit=limit)


def _top_group_rows(
    comparisons: list[dict[str, Any]], *, reverse: bool = False, limit: int = 5
) -> list[dict[str, Any]]:
    if reverse:
        ordered = sorted(comparisons, key=lambda item: (item["delta_accuracy"], item["num_examples"], item["group"]))
    else:
        ordered = comparisons
    return ordered[:limit]


def build_summary(
    results: list[dict[str, Any]],
    labels: dict[str, str],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_metrics = aggregate_metrics(results, "primary")
    secondary_metrics = aggregate_metrics(results, "secondary")
    topic_comparison = build_group_comparison(results, "topic")
    difficulty_comparison = build_group_comparison(results, "difficulty")
    source_comparison = build_group_comparison(results, "source")
    pack_comparison = build_group_comparison(results, "pack_id")
    generator_comparison = build_group_comparison(results, "generator_family")
    summary = {
        "num_examples": len(results),
        "labels": labels,
        "metadata": metadata or {},
        "primary": primary_metrics,
        "secondary": secondary_metrics,
        "delta_accuracy": primary_metrics["accuracy"] - secondary_metrics["accuracy"],
        "delta_step_correctness": primary_metrics["step_correctness"] - secondary_metrics["step_correctness"],
        "delta_solve_rate": primary_metrics["solve_rate"] - secondary_metrics["solve_rate"],
        "delta_quality_score": primary_metrics["avg_quality_score"] - secondary_metrics["avg_quality_score"],
        "trend": {
            "primary": build_trend_series(results, "primary"),
            "secondary": build_trend_series(results, "secondary"),
        },
        "by_topic": by_group(results, "topic"),
        "by_difficulty": by_group(results, "difficulty"),
        "by_source": by_group(results, "source"),
        "by_pack": by_group(results, "pack_id"),
        "by_generator_family": by_group(results, "generator_family"),
        "topic_comparison": topic_comparison,
        "difficulty_comparison": difficulty_comparison,
        "source_comparison": source_comparison,
        "pack_comparison": pack_comparison,
        "generator_family_comparison": generator_comparison,
        "failure_taxonomy": summarize_failure_taxonomy(results),
        "win_cases": collect_win_cases(results),
        "regression_cases": collect_regression_cases(results),
    }
    return summary


def _format_metric_row(label: str, metrics: dict[str, Any], delta: float | None = None) -> str:
    delta_text = "0.000" if delta is None else f"{delta:+.3f}"
    return (
        f"| {label} | {metrics['accuracy']:.3f} | {metrics['solve_rate']:.3f} | "
        f"{metrics['step_correctness']:.3f} | {metrics['avg_quality_score']:.3f} | {delta_text} |"
    )


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    primary_label = summary["labels"]["primary"]
    secondary_label = summary["labels"]["secondary"]
    metadata = summary.get("metadata") or {}
    trend = summary.get("trend") or {}
    lines = [
        "# Evaluation Report",
        "",
        f"- Benchmark: {metadata.get('benchmark_id', 'unknown')}",
        f"- Benchmark path: {metadata.get('benchmark_path', 'unknown')}",
        f"- Resolved benchmark path: {metadata.get('benchmark_resolved_path', 'unknown')}",
        f"- Run label: {metadata.get('run_name', 'unknown')}",
        f"- Examples: {summary['num_examples']}",
        f"- {primary_label} examples scored: {summary['primary']['num_examples']}",
        f"- {secondary_label} examples scored: {summary['secondary']['num_examples']}",
        "",
        "## Scorecard",
        "",
        "| Model | Accuracy | Solve Rate | Step Correctness | Quality Score | Delta vs other |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        _format_metric_row(primary_label, summary["primary"], summary["delta_accuracy"]),
        _format_metric_row(secondary_label, summary["secondary"], -summary["delta_accuracy"]),
        "",
        "## Trend",
        "",
        f"- Primary trend checkpoints: {len((trend.get('primary') or {}).get('checkpoints', []))}",
        f"- Secondary trend checkpoints: {len((trend.get('secondary') or {}).get('checkpoints', []))}",
        "```json",
        json.dumps(trend, indent=2),
        "```",
        "",
        "## Failure Taxonomy",
        "",
        "```json",
        json.dumps(summary["failure_taxonomy"], indent=2),
        "```",
        "",
        "## Group Comparisons",
        "",
        "### Topic Leaders",
        "",
    ]
    topic_rows = _top_group_rows(summary.get("topic_comparison", []), limit=5)
    if not topic_rows:
        lines.append("No topic-level comparisons were available.")
    else:
        lines.extend(["| Topic | Primary | Secondary | Delta |", "| --- | ---: | ---: | ---: |"])
        lines.extend(
            f"| {row['group']} | {row['primary_accuracy']:.3f} | {row['secondary_accuracy']:.3f} | {row['delta_accuracy']:+.3f} |"
            for row in topic_rows
        )
    lines.extend(
        [
            "",
            "### Worst Regressions",
            "",
        ]
    )
    regression_rows = _top_group_rows(summary.get("topic_comparison", []), reverse=True, limit=5)
    if not regression_rows:
        lines.append("No group regressions were available.")
    else:
        lines.extend(["| Topic | Primary | Secondary | Delta |", "| --- | ---: | ---: | ---: |"])
        lines.extend(
            f"| {row['group']} | {row['primary_accuracy']:.3f} | {row['secondary_accuracy']:.3f} | {row['delta_accuracy']:+.3f} |"
            for row in regression_rows
        )
    lines.extend(
        [
            "",
            "## Win Cases",
            "",
        ]
    )
    if not summary["win_cases"]:
        lines.append("No primary-model win cases were found in this run.")
    else:
        for case in summary["win_cases"]:
            lines.extend(
                [
                    f"### {case.get('id', 'unknown')}",
                    str(case.get("question", "")),
                    "",
                    f"- Reference answer: `{case.get('reference_answer')}`",
                    f"- {primary_label}: `{(case.get('primary') or {}).get('final_answer')}`",
                    f"- {secondary_label}: `{(case.get('secondary') or {}).get('final_answer')}`",
                    "",
                ]
            )
    if summary.get("regression_cases"):
        lines.extend(["", "## Regression Cases", ""])
        for case in summary["regression_cases"]:
            lines.extend(
                [
                    f"### {case.get('id', 'unknown')}",
                    str(case.get("question", "")),
                    "",
                    f"- Reference answer: `{case.get('reference_answer')}`",
                    f"- {primary_label}: `{(case.get('primary') or {}).get('final_answer')}`",
                    f"- {secondary_label}: `{(case.get('secondary') or {}).get('final_answer')}`",
                    "",
                ]
            )
    write_markdown(path, "\n".join(lines))


def write_single_model_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics") or {}
    trend = summary.get("trend") or {}
    lines = [
        "# Generated Evaluation Report",
        "",
        f"- Questions: {summary.get('question_count', 0)}",
        f"- Accuracy: {summary.get('accuracy', 0.0):.3f}",
        f"- Correct: {summary.get('correct', 0)}",
        f"- Incorrect or blank: {summary.get('incorrect_or_blank', 0)}",
        f"- Quality score: {metrics.get('avg_quality_score', 0.0):.3f}",
        f"- Avg latency: {metrics.get('avg_latency_s', 0.0):.3f}s",
        f"- Avg candidate agreement: {metrics.get('avg_candidate_agreement', 0.0):.3f}",
        "",
        "## Trend",
        "",
        f"- Trend checkpoints: {len((trend or {}).get('checkpoints', []))}",
        "```json",
        json.dumps(trend, indent=2),
        "```",
        "",
        "## By Topic",
        "",
    ]
    by_topic = summary.get("by_topic") or {}
    if not by_topic:
        lines.append("No topic breakdown was available.")
    else:
        lines.extend(["| Topic | Questions | Accuracy |", "| --- | ---: | ---: |"])
        for topic, details in sorted(by_topic.items()):
            lines.append(f"| {topic} | {details.get('questions', 0)} | {details.get('accuracy', 0.0):.3f} |")
    lines.extend(["", "## By Difficulty", ""])
    by_difficulty = summary.get("by_difficulty") or {}
    if not by_difficulty:
        lines.append("No difficulty breakdown was available.")
    else:
        lines.extend(["| Difficulty | Questions | Accuracy |", "| --- | ---: | ---: |"])
        for difficulty, details in sorted(by_difficulty.items()):
            lines.append(f"| {difficulty} | {details.get('questions', 0)} | {details.get('accuracy', 0.0):.3f} |")
    write_markdown(path, "\n".join(lines))
