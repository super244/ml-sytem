from __future__ import annotations

from evaluation.analysis.analyze_failures import build_failure_analysis


def test_build_failure_analysis_summarizes_topic_counts() -> None:
    analysis = build_failure_analysis(
        [
            {
                "id": "1",
                "question": "What is 2 + 2?",
                "topic": "arithmetic",
                "difficulty": "easy",
                "source": "synthetic",
                "pack_id": "core",
                "primary": {"correct": False, "error_type": "no_answer"},
                "secondary": {"correct": True, "error_type": None},
            },
            {
                "id": "2",
                "question": "What is 3 + 3?",
                "topic": "arithmetic",
                "difficulty": "easy",
                "source": "synthetic",
                "pack_id": "core",
                "primary": {"correct": False, "error_type": "wrong_final_answer"},
                "secondary": {"correct": False, "error_type": "formatting_failure"},
            },
        ]
    )

    assert analysis["num_examples"] == 2
    assert analysis["taxonomy"]["primary"]["no_answer"] == 1
    assert analysis["by_topic"]["arithmetic"] == 2
    assert analysis["failure_examples"]
