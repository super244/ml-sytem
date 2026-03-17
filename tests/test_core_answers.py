from ai_factory.core.answers import (
    answers_equivalent,
    choose_best_candidate,
    compute_step_correctness,
    extract_final_answer,
)


def test_extract_final_answer():
    assert extract_final_answer("Work\nFinal Answer: 3/2") == "3/2"


def test_answers_equivalent_symbolically():
    assert answers_equivalent("1/2", "0.5")


def test_step_correctness_uses_typed_checks():
    score = compute_step_correctness(
        "We use u = x^2 and then compute Final Answer: 1",
        [{"kind": "substring", "value": "u = x^2", "weight": 2.0}],
    )
    assert score == 1.0


def test_choose_best_candidate_prefers_consensus():
    winner = choose_best_candidate(
        [
            {"text": "Final Answer: 2", "final_answer": "2", "verification_score": 1.0},
            {"text": "Final Answer: 2", "final_answer": "2", "verification_score": 0.5},
            {"text": "Final Answer: 3", "final_answer": "3", "verification_score": 1.0},
        ]
    )
    assert winner["final_answer"] == "2"
