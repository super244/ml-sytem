from __future__ import annotations

from ai_factory.core.config.schema import DecisionPolicy
from ai_factory.core.instances.models import DecisionResult, FeedbackRecommendation


def decide_next_step(summary: dict, policy: DecisionPolicy) -> DecisionResult:
    accuracy = float(summary.get("accuracy") or 0.0)
    parse_rate = float(summary.get("parse_rate") or 0.0)
    verifier = float(summary.get("verifier_agreement_rate") or 0.0)
    no_answer = float(summary.get("no_answer_rate") or 1.0)
    latency = summary.get("avg_latency_s")
    latency_value = float(latency) if isinstance(latency, (int, float)) else None
    required_metrics = {
        "accuracy": summary.get("accuracy"),
        "parse_rate": summary.get("parse_rate"),
        "verifier_agreement_rate": summary.get("verifier_agreement_rate"),
        "no_answer_rate": summary.get("no_answer_rate"),
    }
    missing_metrics = [key for key, value in required_metrics.items() if not isinstance(value, (int, float))]
    thresholds = {
        "min_accuracy": policy.min_accuracy,
        "min_parse_rate": policy.min_parse_rate,
        "min_verifier_agreement": policy.min_verifier_agreement,
        "max_no_answer_rate": policy.max_no_answer_rate,
        "max_latency_s": policy.max_latency_s,
        "finetune_accuracy_floor": policy.finetune_accuracy_floor,
    }
    if missing_metrics:
        return DecisionResult(
            action="re_evaluate",
            rule="missing_evaluation_signals",
            thresholds=thresholds,
            summary=summary,
            explanation=(
                "The evaluation summary is missing the primary signals needed for a stable decision: "
                + ", ".join(missing_metrics)
                + "."
            ),
        )
    if (
        accuracy >= policy.min_accuracy
        and parse_rate >= policy.min_parse_rate
        and verifier >= policy.min_verifier_agreement
        and no_answer <= policy.max_no_answer_rate
    ):
        if latency_value is None or latency_value > policy.max_latency_s:
            return DecisionResult(
                action="open_inference",
                rule="needs_interactive_validation",
                thresholds=thresholds,
                summary=summary,
                explanation=(
                    "Quality metrics are strong, but the serving/latency picture is incomplete or above target. "
                    "Open an inference sandbox before publishing."
                ),
            )
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


def build_feedback_recommendations(
    summary: dict,
    policy: DecisionPolicy,
    *,
    default_prepare_config: str,
    default_train_config: str,
    default_finetune_config: str,
    default_eval_config: str,
    default_inference_config: str,
    default_deploy_config: str,
    default_report_config: str,
    improvement_floor: float,
    suggest_failure_analysis: bool,
) -> list[FeedbackRecommendation]:
    recommendations: list[FeedbackRecommendation] = []
    decision = decide_next_step(summary, policy)

    accuracy = float(summary.get("accuracy") or 0.0)
    parse_rate = float(summary.get("parse_rate") or 0.0)
    verifier = float(summary.get("verifier_agreement_rate") or 0.0)
    no_answer = float(summary.get("no_answer_rate") or 1.0)
    latency = summary.get("avg_latency_s")
    latency_value = float(latency) if isinstance(latency, (int, float)) else None
    max_latency = float(policy.max_latency_s)

    if decision.action == "deploy":
        recommendations.append(
            FeedbackRecommendation(
                action="deploy",
                reason="The evaluation metrics meet deployment thresholds, so this build is ready for publishing.",
                priority=1,
                target_instance_type="deploy",
                config_path=default_deploy_config,
                deployment_target="huggingface",
                metadata={"source": "decision", "rule": decision.rule},
            )
        )
        recommendations.append(
            FeedbackRecommendation(
                action="open_inference",
                reason="Open an inference sandbox for a final interactive quality pass before publishing broadly.",
                priority=2,
                target_instance_type="inference",
                config_path=default_inference_config,
                metadata={"source": "decision", "rule": decision.rule},
            )
        )
    elif decision.action == "open_inference":
        recommendations.append(
            FeedbackRecommendation(
                action="open_inference",
                reason=(
                    "The model looks promising enough to inspect interactively. Launch an inference sandbox to check "
                    "prompt quality, latency, and edge-case behavior."
                ),
                priority=1,
                target_instance_type="inference",
                config_path=default_inference_config,
                metadata={"source": "decision", "rule": decision.rule},
            )
        )
    elif decision.action == "finetune":
        accuracy_gap = max(policy.min_accuracy - accuracy, 0.0)
        parse_gap = max(policy.min_parse_rate - parse_rate, 0.0)
        recommendations.append(
            FeedbackRecommendation(
                action="finetune",
                reason=(
                    "The evaluation is close enough to target to justify another fine-tuning pass. "
                    f"Accuracy gap={accuracy_gap:.3f}, parse-rate gap={parse_gap:.3f}."
                ),
                priority=1,
                target_instance_type="finetune",
                config_path=default_finetune_config,
                metadata={
                    "source": "decision",
                    "rule": decision.rule,
                    "improvement_floor": improvement_floor,
                },
            )
        )
    elif decision.action == "re_evaluate":
        recommendations.append(
            FeedbackRecommendation(
                action="re_evaluate",
                reason=(
                    "The evaluation summary is incomplete. Re-run the benchmark suite so the next decision is based "
                    "on a full quality and latency picture."
                ),
                priority=1,
                target_instance_type="evaluate",
                config_path=default_eval_config,
                metadata={"source": "decision", "rule": decision.rule},
            )
        )
    else:
        recommendations.append(
            FeedbackRecommendation(
                action="retrain",
                reason=(
                    "The evaluation is below the fine-tuning floor, so a broader retraining "
                    "pass is the safer next step."
                ),
                priority=1,
                target_instance_type="train",
                config_path=default_train_config,
                metadata={"source": "decision", "rule": decision.rule},
            )
        )

    if suggest_failure_analysis:
        failure_signals: list[str] = []
        if accuracy + improvement_floor < policy.min_accuracy:
            failure_signals.append("accuracy")
        if parse_rate + improvement_floor < policy.min_parse_rate:
            failure_signals.append("parse_rate")
        if verifier + improvement_floor < policy.min_verifier_agreement:
            failure_signals.append("verifier_agreement_rate")
        if no_answer - improvement_floor > policy.max_no_answer_rate:
            failure_signals.append("no_answer_rate")
        if latency_value is not None and latency_value - improvement_floor > max_latency:
            failure_signals.append("avg_latency_s")

        if failure_signals:
            recommendations.append(
                FeedbackRecommendation(
                    action="report",
                    reason="A failure-analysis report would help explain the weak metrics before the next iteration.",
                    priority=2,
                    target_instance_type="report",
                    config_path=default_report_config,
                    metadata={"signals": failure_signals, "improvement_floor": improvement_floor},
                )
            )

    if decision.action in {"finetune", "retrain"}:
        recommendations.append(
            FeedbackRecommendation(
                action="evaluate",
                reason=(
                    "Queue a follow-up evaluation after the next training cycle to "
                    "measure whether the changes helped."
                ),
                priority=3,
                target_instance_type="evaluate",
                config_path=default_eval_config,
                metadata={"source": "feedback_loop"},
            )
        )

    return recommendations
