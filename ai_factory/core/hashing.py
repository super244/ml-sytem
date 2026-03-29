from __future__ import annotations

import hashlib
import re
from pathlib import Path

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", str(text)).strip()


def stable_question_fingerprint(question: str) -> str:
    payload = normalize_text(question).lower().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
