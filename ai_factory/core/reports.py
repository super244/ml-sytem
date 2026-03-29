from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ai_factory.core.io import write_markdown


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    rendered = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        rendered.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rendered)


def bullet_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def write_markdown_report(path: str | Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip(), ""])
    write_markdown(path, "\n".join(lines))
