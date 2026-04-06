import subprocess
import sys
from pathlib import Path

import pytest

import training.train as training_train
from training.src.config import ConfigValidationError, load_experiment_config, validate_experiment_config
from training.src.data import build_messages, build_training_text, load_jsonl
from training.src.hardware import TrainingHardware
from training.src.modeling import build_quantization_config, resolve_attention_implementation, resolve_device_map
from training.src.preflight import build_training_preflight_report
from training.src.scaling import resolve_scratch_architecture
from training.src.validation import build_validation_data_config
from training.src.workflows import build_source_specs


def test_profile_config_merges_components() -> None:
    config = load_experiment_config("training/configs/profiles/failure_aware.yaml")
    assert config.profile_name == "failure_aware"
    assert config.model.scale == "2b"
    assert config.model.base_model_name == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    assert config.data.failure_replay_boost > 1.0
    assert config.packaging.publish_model_name == "atlas-math-failure-aware"


def test_pretraining_profile_uses_scratch_model_and_text_mode() -> None:
    config = load_experiment_config("training/configs/profiles/pretraining.yaml")

    assert config.model.initialization == "scratch"
    assert config.model.model_type == "qwen2"
    assert config.model.target_parameters == "2b"
    assert config.model.architecture["vocab_size"] == 50257
    assert config.data.format == "pretraining_text"
    assert config.adapter.method == "full"


def test_local_metal_profile_uses_non_quantized_lora() -> None:
    config = load_experiment_config("training/configs/profiles/local_metal.yaml")

    assert config.adapter.method == "lora"
    assert config.model.use_4bit is False
    assert config.model.use_8bit is False
    assert config.model.use_full_precision is True


def test_resolve_scratch_architecture_from_target_parameters() -> None:
    architecture, estimate = resolve_scratch_architecture(
        model_type="qwen2",
        target_parameters="2b",
        architecture_overrides={"vocab_size": 50257, "max_position_embeddings": 4096},
    )

    assert architecture["hidden_size"] == 2560
    assert architecture["num_hidden_layers"] == 24
    assert architecture["num_attention_heads"] == 20
    assert architecture["num_key_value_heads"] == 10
    assert architecture["intermediate_size"] == 6912
    assert estimate is not None
    assert abs(estimate - 2_000_000_000) < 10_000_000


def test_resolve_large_scratch_architecture_from_target_parameters() -> None:
    architecture, estimate = resolve_scratch_architecture(
        model_type="qwen2",
        target_parameters="20b",
        architecture_overrides={"vocab_size": 50257, "max_position_embeddings": 4096},
    )

    assert architecture["hidden_size"] >= 4096
    assert architecture["num_hidden_layers"] >= 56
    assert estimate is not None
    assert abs(estimate - 20_000_000_000) < 250_000_000


def test_validation_rejects_conflicting_training_flags() -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    config.training.fp16 = True

    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)


def test_build_validation_data_config_preserves_model_type() -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    config.data.max_train_samples = 8
    config.data.max_eval_samples = 4

    validation_config = build_validation_data_config(config)

    assert validation_config.max_train_samples == 8
    assert validation_config.max_eval_samples == 4
    assert validation_config.model_dump() != {}


def test_load_jsonl_reports_invalid_records(tmp_path: Path) -> None:
    dataset_path = tmp_path / "broken.jsonl"
    dataset_path.write_text('{"question": "ok", "solution": "yes"}\nnot-json\n')

    with pytest.raises(ValueError, match="Invalid JSONL record"):
        load_jsonl(dataset_path)


def test_build_messages_requires_question_and_solution() -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml").data

    with pytest.raises(ValueError, match="question"):
        build_messages({"solution": "answer"}, config)
    with pytest.raises(ValueError, match="solution"):
        build_messages({"question": "prompt"}, config)


def test_build_training_text_uses_pretraining_document_format() -> None:
    config = load_experiment_config("training/configs/profiles/pretraining.yaml").data

    rendered = build_training_text(
        {
            "question": "Compute the derivative of x^2.",
            "solution": "Differentiate termwise to get 2x.",
            "final_answer": "2x",
            "topic": "calculus",
            "difficulty": "easy",
            "source": "custom_derivative_mastery",
        },
        config,
    )

    assert "Problem:" in rendered
    assert "Solution:" in rendered
    assert "Final Answer:" in rendered


def test_build_source_specs_rejects_missing_local_dataset(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="Local dataset not found"):
        build_source_specs(local_datasets=[str(missing_path)])


def test_resolve_attention_implementation_falls_back_when_flash_attention_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    monkeypatch.setattr("training.src.modeling.is_flash_attn_2_available", lambda: False)

    assert resolve_attention_implementation(config) is None


def test_resolve_attention_implementation_preserves_flash_attention_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    monkeypatch.setattr(
        "training.src.modeling.detect_training_hardware",
        lambda: TrainingHardware(
            backend="cuda",
            system="Linux",
            machine="x86_64",
            cuda_available=True,
            cuda_device_count=1,
            mps_available=False,
            cpu_threads=8,
            bitsandbytes_supported=True,
        ),
    )
    monkeypatch.setattr("training.src.modeling.is_flash_attn_2_available", lambda: True)

    assert resolve_attention_implementation(config) == "flash_attention_2"


def test_build_quantization_config_falls_back_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    monkeypatch.setattr(
        "training.src.modeling.detect_training_hardware",
        lambda: TrainingHardware(
            backend="mps",
            system="Darwin",
            machine="arm64",
            cuda_available=False,
            cuda_device_count=0,
            mps_available=True,
            cpu_threads=8,
            bitsandbytes_supported=False,
        ),
    )

    assert build_quantization_config(config) is None


def test_resolve_device_map_disables_auto_on_non_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    monkeypatch.setattr(
        "training.src.modeling.detect_training_hardware",
        lambda: TrainingHardware(
            backend="mps",
            system="Darwin",
            machine="arm64",
            cuda_available=False,
            cuda_device_count=0,
            mps_available=True,
            cpu_threads=8,
            bitsandbytes_supported=False,
        ),
    )

    assert resolve_device_map(config) is None


def test_build_training_arguments_uses_installed_transformers_api(tmp_path: Path) -> None:
    script = f"""
from ai_factory.core.artifacts import prepare_run_layout
from training.train import build_training_arguments
from training.src.config import load_experiment_config

config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
layout = prepare_run_layout(r"{tmp_path}", "unit-training-arguments")
arguments = build_training_arguments(config, layout)
print(arguments.eval_strategy)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert "steps" in completed.stdout.lower()


def test_resolve_precision_flags_prefers_mps_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_experiment_config("training/configs/profiles/local_metal.yaml")
    monkeypatch.setattr(training_train.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training_train, "_mps_available", lambda: True)

    flags = training_train._resolve_precision_flags(config)

    assert flags["use_cpu"] is False
    assert flags["bf16"] is False
    assert flags["fp16"] is False


def test_all_training_profiles_load_and_validate() -> None:
    profile_dir = Path("training/configs/profiles")

    for profile_path in sorted(profile_dir.glob("*.yaml")):
        config = load_experiment_config(str(profile_path))
        warnings = validate_experiment_config(config)

        assert isinstance(warnings, list), profile_path.name


def test_standard_trainer_module_imports_without_trl_dependency() -> None:
    script = """
from training.src.trainer import build_ultimate_trainer
print(callable(build_ultimate_trainer))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert "True" in completed.stdout


def test_training_preflight_flags_missing_scratch_tokenizer(tmp_path: Path) -> None:
    train_path = tmp_path / "data" / "processed" / "train.jsonl"
    eval_path = tmp_path / "data" / "processed" / "eval.jsonl"
    test_path = tmp_path / "data" / "processed" / "test.jsonl"
    manifest_path = tmp_path / "data" / "processed" / "manifest.json"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text('{"question":"q","solution":"a"}\n')
    eval_path.write_text('{"question":"q","solution":"a"}\n')
    test_path.write_text('{"question":"q","solution":"a"}\n')
    manifest_path.write_text('{"schema_version":"v2"}')

    config_path = tmp_path / "pretraining.yaml"
    config_path.write_text(
        "\n".join(
            [
                "profile_name: pretraining",
                "run_name: scratch-test",
                "seed: 1",
                "model:",
                "  initialization: scratch",
                "  model_type: qwen2",
                "  target_parameters: 1b",
                "  base_model_name: scratch/qwen2-1b",
                "  tokenizer_name: gpt2",
                "  tokenizer_path: artifacts/tokenizers/missing",
                "  trust_remote_code: false",
                "  use_4bit: false",
                "  use_8bit: false",
                "  use_full_precision: true",
                "data:",
                f"  train_file: {train_path}",
                f"  eval_file: {eval_path}",
                f"  test_file: {test_path}",
                f"  pack_manifest: {manifest_path}",
                "  format: pretraining_text",
                "training:",
                f"  artifacts_dir: {tmp_path / 'artifacts'}",
                "adapter:",
                "  method: full",
            ]
        )
    )

    report = build_training_preflight_report(str(config_path))
    tokenizer_check = next(check for check in report["checks"] if check["id"] == "tokenizer")

    assert report["status"] == "error"
    assert tokenizer_check["status"] == "error"


def test_training_preflight_flags_quantization_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_path = tmp_path / "data" / "processed" / "train.jsonl"
    eval_path = tmp_path / "data" / "processed" / "eval.jsonl"
    test_path = tmp_path / "data" / "processed" / "test.jsonl"
    manifest_path = tmp_path / "data" / "processed" / "manifest.json"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text('{"question":"q","solution":"a"}\n')
    eval_path.write_text('{"question":"q","solution":"a"}\n')
    test_path.write_text('{"question":"q","solution":"a"}\n')
    manifest_path.write_text('{"schema_version":"v2"}')

    config_path = tmp_path / "quantized.yaml"
    config_path.write_text(
        "\n".join(
            [
                "profile_name: quantized_local",
                "run_name: quantized-local-test",
                "seed: 1",
                "model:",
                "  initialization: pretrained",
                "  base_model_name: Qwen/Qwen2.5-Math-1.5B-Instruct",
                "  trust_remote_code: false",
                "  use_4bit: true",
                "  use_8bit: false",
                "  use_full_precision: false",
                "data:",
                f"  train_file: {train_path}",
                f"  eval_file: {eval_path}",
                f"  test_file: {test_path}",
                f"  pack_manifest: {manifest_path}",
                "training:",
                f"  artifacts_dir: {tmp_path / 'artifacts'}",
                "adapter:",
                "  method: qlora",
            ]
        )
    )
    monkeypatch.setattr(
        "training.src.preflight.detect_training_hardware",
        lambda: TrainingHardware(
            backend="mps",
            system="Darwin",
            machine="arm64",
            cuda_available=False,
            cuda_device_count=0,
            mps_available=True,
            cpu_threads=8,
            bitsandbytes_supported=False,
        ),
    )
    monkeypatch.setattr("training.src.preflight.torch.cuda.device_count", lambda: 0)

    report = build_training_preflight_report(str(config_path))
    quantization_check = next(check for check in report["checks"] if check["id"] == "quantization-runtime")

    assert report["status"] == "warn"
    assert quantization_check["status"] == "warn"
