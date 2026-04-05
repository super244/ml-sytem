from __future__ import annotations

from pathlib import Path

from evaluation.reporting import build_summary, write_markdown_report


def _primary_metrics(correct: bool, final_answer: str) -> dict[str, object]:
    return {
        "correct": correct,
        "solve": True,
        "parse_rate": 1.0,
        "step_correctness": 0.75,
        "verifier_agreement": True,
        "formatting_failure": False,
        "arithmetic_slip": False,
        "no_answer": False,
        "latency_s": 1.25,
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "estimated_cost_usd": 0.01,
        "candidate_agreement": 0.8,
        "final_answer": final_answer,
    }


def test_build_summary_handles_partial_results(tmp_path: Path) -> None:
    results = [
        {
            "id": "1",
            "question": "What is 2 + 2?",
            "reference_answer": "4",
            "primary": _primary_metrics(True, "4"),
            "secondary": {
                "correct": False,
                "solve": True,
                "parse_rate": 1.0,
                "step_correctness": 0.25,
                "verifier_agreement": False,
                "formatting_failure": False,
                "arithmetic_slip": False,
                "no_answer": False,
                "latency_s": 1.5,
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "estimated_cost_usd": 0.02,
                "candidate_agreement": 0.2,
                "final_answer": "5",
                "error_type": "arithmetic",
            },
        },
        {
            "id": "2",
            "question": "What is 3 + 3?",
            "reference_answer": "6",
            "primary": {"correct": False},
            "secondary": None,
        },
    ]

    summary = build_summary(results, labels={"primary": "Model A", "secondary": "Model B"})

    assert summary["primary"]["num_examples"] == 2
    assert summary["secondary"]["num_examples"] == 1

    output_path = tmp_path / "summary.md"
    write_markdown_report(output_path, summary)

    text = output_path.read_text()
    assert "Model A examples scored: 2" in text
    assert "Model B examples scored: 1" in text
    assert "What is 2 + 2?" in text
