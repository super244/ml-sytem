from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from training.src.config import DataConfig


def difficulty_score(level: str | None) -> int:
    mapping = {"easy": 1, "medium": 2, "hard": 3, "olympiad": 4}
    if not level:
        return 3
    return mapping.get(level.lower(), 3)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {path_obj}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path_obj.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL record in {path_obj} at line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"Expected object record in {path_obj} at line {line_number}, got {type(record).__name__}."
            )
        records.append(record)
    return records


def curriculum_sort(records: list[dict[str, Any]], phases: list[str]) -> list[dict[str, Any]]:
    phase_order = {phase: index for index, phase in enumerate(phases)}
    return sorted(
        records,
        key=lambda item: (
            phase_order.get(item.get("difficulty", "hard"), len(phases)),
            0 if item.get("failure_case") else 1,
            len(item.get("question", "")),
        ),
    )


def render_chat(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    segments = []
    for message in messages:
        segments.append(f"<|{message['role']}|>\n{message['content']}\n")
    if add_generation_prompt:
        segments.append("<|assistant|>\n")
    return "".join(segments)


def compute_sample_weight(record: dict[str, Any], data_config: DataConfig) -> float:
    weight = float(record.get("weight", 1.0) or 1.0)
    weight *= float(data_config.source_weights.get(record.get("source", ""), 1.0))
    weight *= float(data_config.difficulty_weights.get(record.get("difficulty", ""), 1.0))
    if data_config.oversample_hard_examples:
        score = difficulty_score(record.get("difficulty"))
        if score >= 4:
            weight *= max(1.0, data_config.hard_weight + 0.5)
        elif score >= 3:
            weight *= max(1.0, data_config.hard_weight)
    if record.get("failure_case"):
        weight *= data_config.failure_replay_boost
    if record.get("step_checks"):
        weight *= data_config.verification_boost
    return round(weight, 4)


def build_messages(record: dict[str, Any], data_config: DataConfig) -> list[dict[str, str]]:
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list):
            raise ValueError(f"Dataset record messages field must be a list: {record.get('id', '<unknown>')}")
        return messages
    question = record.get("question")
    solution = record.get("solution")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Dataset record is missing a non-empty question field: {record.get('id', '<unknown>')}")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError(f"Dataset record is missing a non-empty solution field: {record.get('id', '<unknown>')}")
    topic_line = f"Topic: {record.get('topic', 'general')}.\n" if data_config.include_topic_prefix else ""
    difficulty_line = f"Difficulty: {record.get('difficulty', 'hard')}.\n"
    source_line = f"Source style: {record.get('source', 'unknown')}.\n" if data_config.include_source_tag else ""
    failure_line = (
        "This is a known model failure case. Double-check the critical steps.\n"
        if data_config.include_failure_tag and record.get("failure_case")
        else ""
    )
    verification_line = (
        "The example contains intermediate verification anchors. Preserve them when helpful.\n"
        if data_config.include_verification_tag and record.get("step_checks")
        else ""
    )
    user_prompt = (
        "Solve the following math problem. Show the reasoning step by step and end with "
        "`Final Answer: ...`.\n"
        f"{topic_line}{source_line}{difficulty_line}{failure_line}{verification_line}\n"
        f"Problem:\n{question}"
    )
    return [
        {"role": "system", "content": data_config.system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": solution},
    ]


def tokenize_example(record: dict[str, Any], tokenizer: Any, data_config: DataConfig) -> dict[str, Any]:
    messages = build_messages(record, data_config)
    prompt_messages = messages[:-1]
    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)

    full_tokens = tokenizer(
        full_text,
        max_length=data_config.max_length,
        truncation=True,
        add_special_tokens=False,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        max_length=data_config.max_length,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = list(input_ids)
    prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_weight": compute_sample_weight(record, data_config),
    }


def build_dataset(file_path: str, tokenizer: Any, data_config: DataConfig, split: str) -> Dataset:
    records = load_jsonl(file_path)
    if split == "train" and data_config.max_train_samples:
        records = records[: data_config.max_train_samples]
    if split == "eval" and data_config.max_eval_samples:
        records = records[: data_config.max_eval_samples]
    if split == "train" and data_config.curriculum_learning:
        records = curriculum_sort(records, data_config.curriculum_phases)

    dataset = Dataset.from_list(records)
    return dataset.map(
        lambda example: tokenize_example(example, tokenizer, data_config),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split} dataset",
    )
