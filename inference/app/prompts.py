from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_PROMPT_PRESET_ID = "atlas_rigorous"


@dataclass
class PromptPreset:
    id: str
    title: str
    description: str
    system_prompt: str
    style_instructions: str


def load_prompt_presets(path: str | Path) -> dict[str, PromptPreset]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return {
        item["id"]: PromptPreset(**item)
        for item in payload.get("presets", [])
    }


def build_user_prompt(
    question: str,
    preset: PromptPreset,
    difficulty_target: str | None = None,
    show_reasoning: bool = True,
    use_calculator: bool = False,
    solver_mode: str = "rigorous",
) -> str:
    if solver_mode == "exam":
        reasoning_line = (
            "Present the solution in concise olympiad-exam style with only the essential derivation."
            if show_reasoning
            else "Think carefully, but return only a short exam-style answer and the final result."
        )
    elif solver_mode == "concise":
        reasoning_line = (
            "Provide a compact derivation with only the critical steps."
            if show_reasoning
            else "Think step by step internally, but return only the final answer and one short justification."
        )
    elif solver_mode == "verification":
        reasoning_line = (
            "Expose each key algebraic or calculus step in a verifier-friendly way."
            if show_reasoning
            else "Think carefully, but return only the short verified answer."
        )
    else:
        reasoning_line = (
            "Show the full reasoning step by step."
            if show_reasoning
            else "Think step by step internally, but return only a concise explanation and the final answer."
        )
    calculator_line = (
        "If arithmetic becomes tedious, you may mark a computation as [[calc: expression]] so it can be checked."
        if use_calculator
        else ""
    )
    difficulty_line = f"Target difficulty: {difficulty_target}.\n" if difficulty_target else ""
    return (
        f"{difficulty_line}"
        f"Preset: {preset.title}.\n"
        f"Solver mode: {solver_mode}.\n"
        f"{preset.style_instructions}\n"
        f"{reasoning_line}\n"
        f"{calculator_line}\n\n"
        f"Problem:\n{question}"
    ).strip()
