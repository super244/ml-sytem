from __future__ import annotations

from pathlib import Path

from inference.app.model_catalog import list_model_catalog, summarize_model_catalog


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_list_model_catalog_normalizes_scale_metadata(tmp_path: Path) -> None:
    registry_path = tmp_path / "inference" / "configs" / "model_registry.yaml"
    _write(
        registry_path,
        (
            "models:\n"
            "  - name: base\n"
            "    label: Base 1.5B\n"
            "    base_model: Qwen/Qwen2.5-Math-1.5B-Instruct\n"
            "    tags: [baseline, local]\n"
            "  - name: fast_iteration\n"
            "    label: Fast Iteration\n"
            "    base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
            "    adapter_path: artifacts/models/fast/latest\n"
            "    load_in_8bit: true\n"
            "    tags: [specialist]\n"
        ),
    )

    models = list_model_catalog(registry_path)
    summary = summarize_model_catalog(models)

    assert models[0]["parameter_size_label"] == "1.5B"
    assert models[0]["quantization"] == "4bit"
    assert models[0]["availability_context"]["state"] == "available"
    assert models[0]["availability_context"]["source"] == "base_model"
    assert models[1]["parameter_size_label"] == "0.5B"
    assert models[1]["quantization"] == "8bit"
    assert models[1]["tier"] == "specialist"
    assert models[1]["availability_context"]["state"] == "missing"
    assert summary["total"] == 2
    assert summary["ready"] == 1
    assert summary["quantization_counts"]["4bit"] == 1
    assert summary["quantization_counts"]["8bit"] == 1
    assert "baseline" in summary["scale_tags"]
