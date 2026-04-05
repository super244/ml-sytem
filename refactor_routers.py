import os
import re

ROUTERS_DIR = "/Users/luca/Projects/ai-factory/inference/app/routers"


def process_file(fpath: str) -> None:
    with open(fpath) as f:
        content = f.read()

    if (
        "service = get_" not in content
        and "get_instance_service()." not in content
        and "get_metadata_service()." not in content
    ):
        return

    # Add imports
    if "from typing import Any" not in content:
        content = re.sub(r"(from __future__ import annotations\n+)", r"\1from typing import Any\n", content)

    if "Depends" not in content and "from fastapi import" in content:
        content = re.sub(
            r"(from fastapi import .*?)(?=\n)",
            lambda m: m.group(1) + ", Depends" if "Depends" not in m.group(1) else m.group(1),
            content,
        )

    # Refactor direct calls
    # For def func(..., arg: type = ...): \n    service = get_service()

    # Simple strategy: Find functions that use get_service, modify signature and body
    # Using regex for function signatures: def func(args) -> return_type:
    # This can be multi-line

    # We will use a simpler approach: run my script over the remaining files

    # orchestrations.py uses get_instance_service() directly on return
    # e.g., return get_instance_service().list_orchestration_runs()
    content = re.sub(
        r"def ([a-zA-Z0-9_]+)\((.*?)\)( -> [^:]+)?:\n(\s+)try:\n(\s+)return get_([a-zA-Z0-9_]+)_service\(\)",
        r"def \1(\2, service: Any = Depends(get_\6_service))\3:\n\4try:\n\5return service",
        content,
        flags=re.DOTALL,
    )

    # For metadata.py
    content = re.sub(
        r"def ([a-zA-Z0-9_]+)\((.*?)\)( -> [^:]+)?:\n(\s+)try:\n(\s+)([a-zA-Z0-9_]+) = get_([a-zA-Z0-9_]+)_service\(\)",
        r"def \1(\2, service: Any = Depends(get_\7_service))\3:\n\4try:\n\5\6 = service",
        content,
        flags=re.DOTALL,
    )

    # For instances.py
    content = re.sub(
        r"def ([a-zA-Z0-9_]+)\((.*?)\)( -> [^:]+)?:\n(\s+)service = get_([a-zA-Z0-9_]+)_service\(\)",
        r"def \1(\2, service: Any = Depends(get_\5_service))\3:\n",
        content,
        flags=re.DOTALL,
    )

    content = re.sub(
        r"def ([a-zA-Z0-9_]+)\((.*?)\)( -> [^:]+)?:\n(\s+)return get_([a-zA-Z0-9_]+)_service\(\)",
        r"def \1(\2, service: Any = Depends(get_\5_service))\3:\n\4return service",
        content,
        flags=re.DOTALL,
    )

    # autonomous.py / lab.py: they use get_instance_service in non-route helpers
    # _loop_service() -> AutonomousLoopService:
    #    return AutonomousLoopService(get_settings(), instance_service=get_instance_service())
    # We should only refactor router endpoints. The problem says "router endpoints".
    # We can skip those helpers.

    # clean up empty commas in signatures: def func(, service: Any = Depends...) -> def func(service: Any = Depends...)
    content = content.replace("(, service: Any = Depends", "(service: Any = Depends")
    content = content.replace("(, ", "(")

    with open(fpath, "w") as f:
        f.write(content)

    print(f"Refactored {os.path.basename(fpath)}")


for fname in os.listdir(ROUTERS_DIR):
    if not fname.endswith(".py"):
        continue
    process_file(os.path.join(ROUTERS_DIR, fname))
