from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.io import read_jsonl
from ai_factory.core.tokens import approximate_token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a few records from a dataset JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--fields", nargs="+", default=["question", "solution"], help="Fields to concatenate for tokenization preview.")
    parser.add_argument("--tokenizer", default=None, help="Optional Hugging Face tokenizer name or local path.")
    parser.add_argument("--token-preview-length", type=int, default=12, help="Number of preview tokens to show per field.")
    return parser.parse_args()


def load_tokenizer(tokenizer_name: str | None) -> tuple[Any | None, str]:
    if not tokenizer_name:
        return None, "approximate"
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return None, "approximate (transformers unavailable)"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:  # noqa: BLE001
        return None, "approximate (tokenizer load failed)"
    return tokenizer, "transformers"


def _token_preview(text: str, tokenizer: Any | None, preview_length: int) -> dict[str, Any]:
    if tokenizer is None:
        tokens = text.split()
        return {
            "mode": "approximate",
            "token_count": approximate_token_count(text),
            "token_preview": tokens[:preview_length],
        }
    try:
        if hasattr(tokenizer, "tokenize"):
            tokens = tokenizer.tokenize(text)
            token_count = len(tokens)
        else:
            encoded = tokenizer(text, add_special_tokens=False)
            token_ids = list(encoded.get("input_ids", []))
            token_count = len(token_ids)
            tokens = (
                tokenizer.convert_ids_to_tokens(token_ids)
                if hasattr(tokenizer, "convert_ids_to_tokens")
                else [str(token_id) for token_id in token_ids]
            )
        return {
            "mode": "transformers",
            "token_count": token_count,
            "token_preview": tokens[:preview_length],
        }
    except Exception:  # noqa: BLE001
        tokens = text.split()
        return {
            "mode": "approximate",
            "token_count": approximate_token_count(text),
            "token_preview": tokens[:preview_length],
        }


def preview_rows(
    rows: list[dict[str, Any]],
    *,
    fields: list[str],
    tokenizer: Any | None = None,
    preview_length: int = 12,
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for row in rows:
        field_previews = {}
        field_texts: list[str] = []
        for field in fields:
            value = row.get(field)
            text = "" if value is None else str(value)
            field_texts.append(text)
            field_previews[field] = _token_preview(text, tokenizer, preview_length)
        combined_text = "\n\n".join(text for text in field_texts if text)
        previews.append(
            {
                "id": row.get("id"),
                "topic": row.get("topic"),
                "difficulty": row.get("difficulty"),
                "question": row.get("question"),
                "final_answer": row.get("final_answer"),
                "tokenization": {
                    "fields": field_previews,
                    "combined": _token_preview(combined_text, tokenizer, preview_length),
                },
            }
        )
    return previews


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))[: args.limit]
    tokenizer, tokenizer_mode = load_tokenizer(args.tokenizer)
    preview = preview_rows(rows, fields=args.fields, tokenizer=tokenizer, preview_length=args.token_preview_length)
    if tokenizer_mode != "transformers":
        for row in preview:
            row["tokenization"]["mode"] = tokenizer_mode
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
