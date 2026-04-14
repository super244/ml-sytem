from __future__ import annotations

import os
import time
from typing import Any, Protocol

import torch

from ai_factory.core.answers import (
    candidate_agreement,
    choose_best_candidate,
    extract_final_answer,
    resolve_calculator_tags,
    split_reasoning,
    verify_prediction,
)
from ai_factory.core.math_stack.model_loader import MathModelRegistry
from ai_factory.core.math_stack.parameters import GenerationParameters
from ai_factory.core.math_stack.prompts import DEFAULT_PROMPT_PRESET_ID, PromptPreset, build_user_prompt
from ai_factory.core.tokens import approximate_token_count
from ai_factory.titan import titan_diagnostics


class _ResponseCache(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...

    def set(self, key: str, value: dict[str, Any]) -> None: ...


class _TelemetryLogger(Protocol):
    def log_event(self, kind: str, payload: dict[str, Any]) -> str | None: ...


class MathGenerator:
    """Math generation pipeline shared across training, evaluation, and inference."""

    def __init__(
        self,
        registry: MathModelRegistry,
        prompt_presets: dict[str, PromptPreset],
        cache: _ResponseCache | None = None,
        telemetry: _TelemetryLogger | None = None,
    ):
        self.registry = registry
        self.prompt_presets = prompt_presets
        self.cache = cache
        self.telemetry = telemetry

    def _runtime_metadata(self) -> dict[str, Any]:
        diagnostics = titan_diagnostics()
        runtime = diagnostics.get("runtime") or {}
        engine = diagnostics.get("engine") or {}
        selected = str(runtime.get("selected") or "python")
        canary_requested = selected.startswith("rust")
        canary_enabled = os.getenv("AI_FACTORY_TITAN_ENABLE_CANARY_GENERATION", "0").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        execution_path = "python-transformers"
        reason = "Python Transformers path remains active."
        if canary_requested and canary_enabled:
            execution_path = "rust-canary-preflight"
            reason = "Rust canary was requested; generation remains Python-backed while Titan preflight is active."
        elif canary_requested:
            execution_path = "python-fallback"
            reason = "Rust runtime was requested but canary generation is not enabled."
        return {
            "selected": selected,
            "execution_path": execution_path,
            "source": str(runtime.get("status_source") or "python-probe"),
            "canary_requested": canary_requested,
            "canary_active": canary_requested and canary_enabled,
            "gguf_support": bool(
                runtime.get("gguf_support") or engine.get("supports_gguf") or engine.get("gguf_support")
            ),
            "kv_cache": bool(
                runtime.get("kv_cache")
                if isinstance(runtime.get("kv_cache"), bool)
                else (runtime.get("kv_cache") or {}).get(
                    "enabled", engine.get("supports_kv_cache", engine.get("kv_cache"))
                )
            ),
            "sampler_stack": list(runtime.get("sampler_stack") or engine.get("sampler_stack") or []),
            "reason": reason,
        }

    def _resolve_preset(self, preset_id: str) -> PromptPreset:
        return self.prompt_presets.get(preset_id) or self.prompt_presets[DEFAULT_PROMPT_PRESET_ID]

    def _render_prompt(self, tokenizer: Any, params: GenerationParameters) -> tuple[str, PromptPreset]:
        preset = self._resolve_preset(params.prompt_preset)
        messages = [
            {"role": "system", "content": preset.system_prompt},
            {
                "role": "user",
                "content": build_user_prompt(
                    question=params.question,
                    preset=preset,
                    difficulty_target=params.difficulty_target,
                    show_reasoning=params.show_reasoning,
                    use_calculator=params.use_calculator,
                    solver_mode=params.solver_mode,
                ),
            },
        ]
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                return (
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    preset,
                )
            except ValueError:
                pass
        prompt = "".join(f"<|{item['role']}|>\n{item['content']}\n" for item in messages)
        return prompt + "<|assistant|>\n", preset

    def _model_input_device(self, model: Any) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _sample_candidates(self, params: GenerationParameters) -> tuple[str, PromptPreset, list[dict[str, Any]]]:
        runtime = self.registry.get_runtime(params.model_variant).load()
        prompt, preset = self._render_prompt(runtime.tokenizer, params)
        model_inputs = runtime.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        model_inputs = {key: value.to(self._model_input_device(runtime.model)) for key, value in model_inputs.items()}
        prompt_length = model_inputs["input_ids"].shape[-1]

        candidate_count = max(1, params.num_samples)
        candidates: list[dict[str, Any]] = []
        sampling = (params.temperature > 0) or (candidate_count > 1)

        with torch.inference_mode():
            generate_kwargs = dict(
                **model_inputs,
                do_sample=sampling,
                top_p=params.top_p,
                max_new_tokens=params.max_new_tokens,
                repetition_penalty=1.05,
                pad_token_id=runtime.tokenizer.pad_token_id,
                eos_token_id=runtime.tokenizer.eos_token_id,
                num_return_sequences=candidate_count,
            )
            if sampling:
                generate_kwargs["temperature"] = max(1e-6, float(params.temperature))

            output_ids_batch = runtime.model.generate(**generate_kwargs)

            for output_ids in output_ids_batch:
                generated_ids = output_ids[prompt_length:]
                raw_text = runtime.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                resolved_text, calculator_trace = resolve_calculator_tags(raw_text)
                reasoning, final_answer = split_reasoning(resolved_text)
                verification = verify_prediction(
                    resolved_text,
                    reference_answer=params.reference_answer,
                    step_checks=params.step_checks,
                )
                display_text = (
                    resolved_text
                    if params.show_reasoning
                    else f"Final Answer: {final_answer or extract_final_answer(resolved_text) or resolved_text}"
                )
                candidates.append(
                    {
                        "text": resolved_text,
                        "display_text": display_text,
                        "reasoning": reasoning,
                        "final_answer": final_answer or extract_final_answer(resolved_text),
                        "calculator_trace": calculator_trace,
                        "verification": {
                            "final_answer": verification.final_answer,
                            "equivalent": verification.equivalent,
                            "step_correctness": verification.step_correctness,
                            "verifier_agreement": verification.verifier_agreement,
                            "formatting_failure": verification.formatting_failure,
                            "arithmetic_slip": verification.arithmetic_slip,
                            "error_type": verification.error_type,
                        },
                        "verification_score": (
                            (1.0 if verification.verifier_agreement else 0.0)
                            + (verification.step_correctness or 0.0)
                            - (0.2 if verification.formatting_failure else 0.0)
                        ),
                        "prompt_tokens": approximate_token_count(prompt, runtime.tokenizer),
                        "completion_tokens": approximate_token_count(resolved_text, runtime.tokenizer),
                    }
                )
        return prompt, preset, candidates

    def generate(self, params: GenerationParameters) -> dict[str, Any]:
        cache_key = params.cache_key()
        if self.cache is not None and params.use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached["cache_hit"] = True
                return cached

        start = time.perf_counter()
        runtime_metadata = self._runtime_metadata()
        prompt, preset, candidates = self._sample_candidates(params)
        winner = choose_best_candidate(candidates)
        verification = winner.get("verification")
        latency_s = time.perf_counter() - start
        structured = {
            "reasoning": winner.get("reasoning", ""),
            "final_answer": winner.get("final_answer"),
            "verification": verification,
        }
        result = {
            "model_variant": params.model_variant,
            "prompt": prompt,
            "answer": winner["display_text"],
            "raw_text": winner["text"],
            "final_answer": winner.get("final_answer"),
            "reasoning_steps": [line.strip() for line in winner.get("reasoning", "").splitlines() if line.strip()],
            "selected_score": winner["verification_score"],
            "candidates": candidates,
            "verification": verification,
            "structured": structured if params.output_format == "json" else None,
            "cache_hit": False,
            "telemetry_id": None,
            "latency_s": latency_s,
            "prompt_preset": preset.id,
            "candidate_agreement": candidate_agreement(candidates),
            "runtime": runtime_metadata,
        }
        prompt_tokens = winner.get("prompt_tokens")
        completion_tokens = winner.get("completion_tokens")
        tokens_per_s = None
        if isinstance(completion_tokens, int) and latency_s > 0:
            tokens_per_s = completion_tokens / latency_s
        if self.cache is not None and params.use_cache:
            self.cache.set(cache_key, result)
        if self.telemetry is not None:
            result["telemetry_id"] = self.telemetry.log_event(
                "generation",
                {
                    "model_variant": params.model_variant,
                    "prompt_preset": preset.id,
                    "cache_key": cache_key,
                    "latency_s": latency_s,
                    "final_answer": result["final_answer"],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_s": tokens_per_s,
                    "candidate_agreement": result["candidate_agreement"],
                    "candidate_count": len(candidates),
                    "runtime": runtime_metadata,
                },
            )
        return result
