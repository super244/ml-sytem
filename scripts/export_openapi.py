#!/usr/bin/env python3
"""Emit FastAPI OpenAPI JSON for frontend contract generation (run from repo root)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from inference.app.main import app

    schema = app.openapi()
    out_dir = root / "frontend" / "lib" / "api" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "openapi.json"
    out_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
