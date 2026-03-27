from __future__ import annotations

from ai_factory.core.config.schema import DecisionPolicy
from ai_factory.core.instances.models import DecisionResult


def decide_next_step(summary: dict, policy: DecisionPolicy) -> DecisionResult:
    accuracy = float(summary.get("accuracy") or 0.0)
    parse_rate = float(summary.get("parse_rate") or 0.0)
    verifier = float(summary.get("verifier_agreement_rate") or 0.0)
    no_answer = float(summary.get("no_answer_rate") or 1.0)
    latency = summary.get("avg_latency_s")
    latency_value = float(latency) if isinstance(latency, (int, float)) else None
    thresholds = {
        "min_accuracy": policy.min_accuracy,
        "min_parse_rate": policy.min_parse_rate,
        "min_verifier_agreement": policy.min_verifier_agreement,
        "max_no_answer_rate": policy.max_no_answer_rate,
        "max_latency_s": policy.max_latency_s,
        "finetune_accuracy_floor": policy.finetune_accuracy_floor,
    }
    if (
        accuracy >= policy.min_accuracy
        and parse_rate >= policy.min_parse_rate
        and verifier >= policy.min_verifier_agreement
        and no_answer <= policy.max_no_answer_rate
        and (latency_value is None or latency_value <= policy.max_latency_s)
    ):
        return DecisionResult(
            action="deploy",
            rule="meets_deploy_thresholds",
            thresholds=thresholds,
            summary=summary,
            explanation="The evaluation metrics clear the deployment thresholds.",
        )
    if accuracy >= policy.finetune_accuracy_floor or parse_rate >= policy.min_parse_rate:
        return DecisionResult(
            action="finetune",
            rule="needs_iteration",
            thresholds=thresholds,
            summary=summary,
            explanation="The model shows signal worth iterating on, but is not deployment-ready yet.",
        )
    return DecisionResult(
        action="retrain",
        rule="below_iteration_floor",
        thresholds=thresholds,
        summary=summary,
        explanation="The metrics are below the iteration floor, so a broader retraining pass is recommended.",
    )
