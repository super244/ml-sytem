from evaluation.metrics import score_prediction


def test_score_prediction_reports_error_taxonomy() -> None:
    result = score_prediction(
        prediction_text="We forgot the format",
        reference_answer="3",
        step_checks=[{"kind": "substring", "value": "critical"}],
        prompt_text="Solve this",
    )
    assert result["no_answer"] is True
    assert result["error_type"] == "no_answer"
    assert result["candidate_count"] == 0
    assert result["quality_score"] >= 0.0
    assert "correctness" in result["metric_components"]
