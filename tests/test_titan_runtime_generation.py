from __future__ import annotations

from inference.app.generation import MathGenerator
from inference.app.parameters import GenerationParameters
from inference.app.prompts import PromptPreset


class DummyRegistry:
    def get_runtime(self, _name: str):  # pragma: no cover - generator path is monkeypatched below
        raise AssertionError("runtime should not be loaded in this test")


def test_generate_reports_python_fallback_when_rust_canary_not_enabled(monkeypatch) -> None:
    preset = PromptPreset(
        id="atlas_rigorous",
        title="Rigorous",
        description="Test preset",
        system_prompt="Solve carefully.",
        style_instructions="Show your work.",
    )
    generator = MathGenerator(DummyRegistry(), prompt_presets={"atlas_rigorous": preset})

    monkeypatch.setattr(
        "inference.app.generation.titan_diagnostics",
        lambda: {
            "runtime": {
                "selected": "rust-canary",
                "status_source": "rust-binary",
                "gguf_support": True,
                "kv_cache": {"enabled": True, "strategy": "paged_kv"},
                "sampler_stack": ["argmax", "top_k"],
            },
            "engine": {
                "supports_gguf": True,
                "supports_kv_cache": True,
                "sampler_stack": ["argmax", "top_k"],
            },
        },
    )
    monkeypatch.setattr(
        generator,
        "_sample_candidates",
        lambda params: (
            "prompt",
            preset,
            [
                {
                    "text": "Final Answer: 2",
                    "display_text": "Final Answer: 2",
                    "reasoning": "step",
                    "final_answer": "2",
                    "verification": None,
                    "verification_score": 1.0,
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                }
            ],
        ),
    )

    result = generator.generate(GenerationParameters(question="1+1?", model_variant="base", num_samples=1))

    assert result["runtime"]["selected"] == "rust-canary"
    assert result["runtime"]["execution_path"] == "python-fallback"
    assert result["runtime"]["canary_requested"] is True
    assert result["runtime"]["canary_active"] is False
    assert result["runtime"]["gguf_support"] is True
