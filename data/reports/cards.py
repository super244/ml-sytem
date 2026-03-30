from __future__ import annotations

from collections import Counter
from typing import Any


def dataset_card_text(entry: dict[str, Any]) -> str:
    lines = [
        f"# {entry['title']}",
        "",
        f"- Dataset id: `{entry['id']}`",
        f"- Kind: `{entry['kind']}`",
        f"- Family: `{entry.get('family', 'n/a')}`",
        f"- Topic: `{entry.get('topic', 'n/a')}`",
        f"- Reasoning style: `{entry.get('reasoning_style', 'mixed')}`",
        f"- Usage: `{entry.get('usage', 'n/a')}`",
        f"- Default weight: `{entry.get('default_weight', 1.0)}`",
        f"- Rows: `{entry.get('num_rows', 0)}`",
        f"- Approx size: `{entry.get('size_bytes', 0)} bytes`",
        f"- Benchmark tags: `{entry.get('benchmark_tags', [])}`",
        "",
        "## Focus",
        "",
        entry.get("description", ""),
        "",
        "## Preview",
        "",
    ]
    previews = entry.get("preview_examples", [])
    if not previews:
        lines.append("No preview examples captured.")
    else:
        for preview in previews[:3]:
            lines.extend(
                [
                    f"### {preview['id']}",
                    preview["question"],
                    "",
                    f"- Difficulty: `{preview['difficulty']}`",
                    f"- Final answer: `{preview['final_answer']}`",
                    "",
                ]
            )
    return "\n".join(lines)


def pack_card_text(pack_id: str, description: str, records: list[dict[str, Any]]) -> str:
    topic_counts = Counter(record.get("topic", "unknown") for record in records)
    difficulty_counts = Counter(record.get("difficulty", "unknown") for record in records)
    lines = [
        f"# {pack_id}",
        "",
        description,
        "",
        f"- Rows: `{len(records)}`",
        f"- Topics: `{dict(topic_counts)}`",
        f"- Difficulties: `{dict(difficulty_counts)}`",
        "",
        "## Sample records",
        "",
    ]
    for record in records[:5]:
        lines.extend(
            [
                f"### {record.get('id', 'unknown')}",
                record.get("question", ""),
                "",
                f"- Topic: `{record.get('topic', 'unknown')}`",
                f"- Difficulty: `{record.get('difficulty', 'unknown')}`",
                f"- Source: `{record.get('source', 'unknown')}`",
                "",
            ]
        )
    return "\n".join(lines)


def size_report_markdown(title: str, items: list[dict[str, Any]]) -> str:
    lines = [f"# {title}", "", "| Item | Rows | Size Bytes |", "| --- | --- | --- |"]
    for item in items:
        lines.append(
            f"| {item.get('id', item.get('path', 'unknown'))} | {item.get('num_rows', 0)} | {item.get('size_bytes', 0)} |"
        )
    return "\n".join(lines)
