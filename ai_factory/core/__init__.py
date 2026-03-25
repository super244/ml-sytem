from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_factory.core.answers import (
        answer_key,
        answers_equivalent,
        candidate_agreement,
        choose_best_candidate,
        classify_prediction_failure,
        compute_step_correctness,
        detect_arithmetic_slip,
        detect_formatting_failure,
        extract_final_answer,
        resolve_calculator_tags,
        split_reasoning,
        verify_prediction,
    )
    from ai_factory.core.artifacts import (
        ArtifactLayout,
        RunEnv,
        current_git_sha,
        detect_run_env,
        ensure_latest_pointer,
        prepare_run_layout,
        read_jsonl,
        sha256_file,
        sha256_text,
        write_json,
        write_jsonl,
        write_markdown,
    )
    from ai_factory.core.hashing import normalize_text, stable_question_fingerprint
    from ai_factory.core.schemas import (
        ChatMessage,
        ContaminationStatus,
        DatasetBuildInfo,
        DatasetFileInfo,
        DatasetManifest,
        GeneratorMetadata,
        MathRecord,
        MathRecordV2,
        PackagedMathRecord,
        PackagedMathRecordV2,
        RunManifest,
        SourceLineage,
        StepCheck,
    )


_EXPORT_MODULES = {
    "ArtifactLayout": "ai_factory.core.artifacts",
    "ChatMessage": "ai_factory.core.schemas",
    "ContaminationStatus": "ai_factory.core.schemas",
    "DatasetBuildInfo": "ai_factory.core.schemas",
    "DatasetFileInfo": "ai_factory.core.schemas",
    "DatasetManifest": "ai_factory.core.schemas",
    "GeneratorMetadata": "ai_factory.core.schemas",
    "MathRecord": "ai_factory.core.schemas",
    "MathRecordV2": "ai_factory.core.schemas",
    "PackagedMathRecord": "ai_factory.core.schemas",
    "PackagedMathRecordV2": "ai_factory.core.schemas",
    "RunEnv": "ai_factory.core.artifacts",
    "RunManifest": "ai_factory.core.schemas",
    "SourceLineage": "ai_factory.core.schemas",
    "StepCheck": "ai_factory.core.schemas",
    "answer_key": "ai_factory.core.answers",
    "answers_equivalent": "ai_factory.core.answers",
    "candidate_agreement": "ai_factory.core.answers",
    "choose_best_candidate": "ai_factory.core.answers",
    "classify_prediction_failure": "ai_factory.core.answers",
    "compute_step_correctness": "ai_factory.core.answers",
    "current_git_sha": "ai_factory.core.artifacts",
    "detect_arithmetic_slip": "ai_factory.core.answers",
    "detect_formatting_failure": "ai_factory.core.answers",
    "detect_run_env": "ai_factory.core.artifacts",
    "ensure_latest_pointer": "ai_factory.core.artifacts",
    "extract_final_answer": "ai_factory.core.answers",
    "normalize_text": "ai_factory.core.hashing",
    "prepare_run_layout": "ai_factory.core.artifacts",
    "read_jsonl": "ai_factory.core.artifacts",
    "resolve_calculator_tags": "ai_factory.core.answers",
    "sha256_file": "ai_factory.core.artifacts",
    "sha256_text": "ai_factory.core.artifacts",
    "split_reasoning": "ai_factory.core.answers",
    "stable_question_fingerprint": "ai_factory.core.hashing",
    "verify_prediction": "ai_factory.core.answers",
    "write_json": "ai_factory.core.artifacts",
    "write_jsonl": "ai_factory.core.artifacts",
    "write_markdown": "ai_factory.core.artifacts",
}

__all__ = [
    "ArtifactLayout",
    "ChatMessage",
    "ContaminationStatus",
    "DatasetBuildInfo",
    "DatasetFileInfo",
    "DatasetManifest",
    "GeneratorMetadata",
    "MathRecord",
    "MathRecordV2",
    "PackagedMathRecord",
    "PackagedMathRecordV2",
    "RunEnv",
    "RunManifest",
    "SourceLineage",
    "StepCheck",
    "answer_key",
    "answers_equivalent",
    "candidate_agreement",
    "choose_best_candidate",
    "classify_prediction_failure",
    "compute_step_correctness",
    "current_git_sha",
    "detect_arithmetic_slip",
    "detect_formatting_failure",
    "detect_run_env",
    "ensure_latest_pointer",
    "extract_final_answer",
    "normalize_text",
    "prepare_run_layout",
    "read_jsonl",
    "resolve_calculator_tags",
    "sha256_file",
    "sha256_text",
    "split_reasoning",
    "stable_question_fingerprint",
    "verify_prediction",
    "write_json",
    "write_jsonl",
    "write_markdown",
]


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
