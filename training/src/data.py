from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from functools import partial
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

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
    with path_obj.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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


def load_sqlite(path: str, *, split: str | None = None) -> list[dict[str, Any]]:
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"SQLite dataset not found: {path_obj}")
    connection = sqlite3.connect(path_obj)
    try:
        query = "SELECT payload_json FROM records"
        params: list[Any] = []
        if split:
            query += " WHERE dataset_split = ?"
            params.append(split)
        query += " ORDER BY sequence_id ASC"
        return [json.loads(row[0]) for row in connection.execute(query, params).fetchall()]
    finally:
        connection.close()


def load_records(path: str, *, split: str | None = None) -> list[dict[str, Any]]:
    suffix = Path(path).expanduser().suffix.lower()
    if suffix in {".sqlite", ".db"}:
        return load_sqlite(path, split=split)
    return load_jsonl(path)


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
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except ValueError:
            # Fallback to simple template if chat template is not set
            pass
    segments = []
    for message in messages:
        segments.append(f"<|{message['role']}|>\n{message['content']}\n")
    if add_generation_prompt:
        segments.append("<|assistant|>\n")
    return "".join(segments)


def render_plain_chat(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    return render_chat(tokenizer=None, messages=messages, add_generation_prompt=add_generation_prompt)


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


def build_pretraining_text(record: dict[str, Any], data_config: DataConfig) -> str:
    raw_text = record.get("text")
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text.strip()
    question = record.get("question")
    solution = record.get("solution")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Dataset record is missing a non-empty question field: {record.get('id', '<unknown>')}")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError(f"Dataset record is missing a non-empty solution field: {record.get('id', '<unknown>')}")

    prefix_lines: list[str] = []
    if data_config.include_topic_prefix and record.get("topic"):
        prefix_lines.append(f"Topic: {record['topic']}")
    if data_config.include_source_tag and record.get("source"):
        prefix_lines.append(f"Source: {record['source']}")
    if record.get("difficulty"):
        prefix_lines.append(f"Difficulty: {record['difficulty']}")

    sections = []
    if prefix_lines:
        sections.append("\n".join(prefix_lines))
    sections.append(f"Problem:\n{question.strip()}")
    sections.append(f"Solution:\n{solution.strip()}")
    final_answer = record.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        sections.append(f"Final Answer:\n{final_answer.strip()}")
    return "\n\n".join(sections)


def build_training_text(record: dict[str, Any], data_config: DataConfig) -> str:
    if data_config.format == "pretraining_text":
        return build_pretraining_text(record, data_config)
    return render_plain_chat(build_messages(record, data_config), add_generation_prompt=False)


def _batch_to_records(batch: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not batch:
        return []
    batch_size = len(next(iter(batch.values())))
    return [{key: values[index] for key, values in batch.items()} for index in range(batch_size)]


def _tokenize_batch(batch: dict[str, list[Any]], tokenizer: Any, data_config: DataConfig) -> dict[str, list[Any]]:
    records = _batch_to_records(batch)
    if not records:
        return {"input_ids": [], "attention_mask": [], "labels": [], "sample_weight": []}

    if data_config.format == "pretraining_text":
        texts = [build_pretraining_text(record, data_config) for record in records]
        encoded = tokenizer(
            texts,
            max_length=data_config.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": [list(input_ids) for input_ids in encoded["input_ids"]],
            "sample_weight": [compute_sample_weight(record, data_config) for record in records],
        }

    message_batches = [build_messages(record, data_config) for record in records]
    prompt_message_batches = [messages[:-1] for messages in message_batches]
    full_texts = [render_chat(tokenizer, messages, add_generation_prompt=False) for messages in message_batches]
    prompt_texts = [
        render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
        for prompt_messages in prompt_message_batches
    ]
    full_tokens = tokenizer(
        full_texts,
        max_length=data_config.max_length,
        truncation=True,
        add_special_tokens=False,
    )
    prompt_tokens = tokenizer(
        prompt_texts,
        max_length=data_config.max_length,
        truncation=True,
        add_special_tokens=False,
    )

    labels: list[list[int]] = []
    for input_ids, prompt_ids in zip(full_tokens["input_ids"], prompt_tokens["input_ids"], strict=False):
        masked_labels = list(input_ids)
        prompt_length = min(len(prompt_ids), len(masked_labels))
        masked_labels[:prompt_length] = [-100] * prompt_length
        labels.append(masked_labels)

    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
        "sample_weight": [compute_sample_weight(record, data_config) for record in records],
    }


def tokenize_example(record: dict[str, Any], tokenizer: Any, data_config: DataConfig) -> dict[str, Any]:
    if data_config.format == "pretraining_text":
        text = build_pretraining_text(record, data_config)
        encoded = tokenizer(
            text,
            max_length=data_config.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"],
            "labels": list(input_ids),
            "sample_weight": compute_sample_weight(record, data_config),
        }

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


def _resolve_tokenization_num_proc(data_config: DataConfig, num_rows: int) -> int | None:
    requested = data_config.tokenization_num_proc
    if requested <= 0:
        return None
    if num_rows < 2:
        return None
    return max(1, min(requested, num_rows))


def _tokenized_cache_path(file_path: str, tokenizer: Any, data_config: DataConfig, split: str) -> Path:
    path_obj = Path(file_path).expanduser().resolve()
    stat = path_obj.stat()
    cache_root = (
        Path(data_config.tokenized_cache_dir).expanduser()
        if data_config.tokenized_cache_dir
        else path_obj.parent / ".tokenized_cache"
    )
    tokenizer_id = getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "file_path": str(path_obj),
                "file_size": stat.st_size,
                "file_mtime_ns": stat.st_mtime_ns,
                "split": split,
                "tokenizer": tokenizer_id,
                "data_config": data_config.model_dump(),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]
    return cache_root / fingerprint


def build_dataset(file_path: str, tokenizer: Any, data_config: DataConfig, split: str) -> Dataset:
    records = load_records(file_path, split=split)
    if split == "train" and data_config.max_train_samples:
        records = records[: data_config.max_train_samples]
    if split == "eval" and data_config.max_eval_samples:
        records = records[: data_config.max_eval_samples]
    if split == "train" and data_config.curriculum_learning:
        records = curriculum_sort(records, data_config.curriculum_phases)

    if data_config.use_tokenized_cache:
        cache_path = _tokenized_cache_path(file_path, tokenizer, data_config, split)
        if cache_path.exists():
            try:
                return load_from_disk(str(cache_path))
            except Exception:
                shutil.rmtree(cache_path, ignore_errors=True)

    dataset = Dataset.from_list(records)
    tokenize_batch = partial(_tokenize_batch, tokenizer=tokenizer, data_config=data_config)
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=max(1, data_config.tokenization_batch_size),
        num_proc=_resolve_tokenization_num_proc(data_config, len(records)),
        remove_columns=dataset.column_names,
        load_from_cache_file=data_config.use_tokenized_cache,
        desc=f"Tokenizing {split} dataset",
    )
    if data_config.use_tokenized_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            shutil.rmtree(cache_path)
        tokenized.save_to_disk(str(cache_path))
    return tokenized
