from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.io import load_json
from ai_factory.core.schemas import DatasetManifest, MathRecord, PackagedMathRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw or packaged dataset JSONL or SQLite files and optional manifests.")
    parser.add_argument("--input", required=True, help="Path/glob to JSONL file(s) or a SQLite corpus file.")
    parser.add_argument("--manifest", default=None, help="Optional manifest path to validate.")
    parser.add_argument("--schema", choices=["auto", "record", "packaged"], default="auto")
    parser.add_argument("--split", choices=["train", "eval", "test"], default=None)
    parser.add_argument("--max-errors", type=int, default=50)
    return parser.parse_args()


def resolve_inputs(pattern: str) -> list[Path]:
    if any(ch in pattern for ch in "*?[]"):
        return sorted(Path().glob(pattern))
    return [Path(pattern)]


def iter_payloads(path: Path, *, split: str | None = None):
    if path.suffix.lower() in {".sqlite", ".db"}:
        connection = sqlite3.connect(path)
        try:
            query = "SELECT payload_json FROM records"
            params: list[Any] = []
            if split:
                query += " WHERE dataset_split = ?"
                params.append(split)
            query += " ORDER BY sequence_id ASC"
            for idx, row in enumerate(connection.execute(query, params), start=1):
                yield idx, json.loads(row[0])
        finally:
            connection.close()
        return
    for idx, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if line:
            yield idx, json.loads(line)


def main() -> None:
    args = parse_args()
    inputs = resolve_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No inputs match: {args.input}")

    total = 0
    ok = 0
    errors: list[dict[str, Any]] = []
    for path in inputs:
        try:
            payloads = list(iter_payloads(path, split=args.split))
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": str(path), "line": 1, "error": str(exc)})
            continue
        for idx, payload in payloads:
            total += 1
            try:
                _validate_payload(payload, schema=args.schema)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                if len(errors) < args.max_errors:
                    errors.append({"file": str(path), "line": idx, "error": str(exc)})
    manifest_ok = None
    if args.manifest:
        try:
            DatasetManifest.model_validate(load_json(args.manifest))
            manifest_ok = True
        except Exception as exc:  # noqa: BLE001
            manifest_ok = False
            errors.append({"file": args.manifest, "line": 1, "error": str(exc)})

    summary = {
        "total": total,
        "ok": ok,
        "failed": total - ok,
        "manifest_ok": manifest_ok,
        "sample_errors": errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


def _validate_payload(payload: dict[str, Any], schema: str) -> None:
    if schema == "packaged":
        PackagedMathRecord.model_validate(payload)
        return
    if schema == "record":
        MathRecord.model_validate(payload)
        return
    if "messages" in payload:
        PackagedMathRecord.model_validate(payload)
        return
    MathRecord.model_validate(payload)


if __name__ == "__main__":
    main()
