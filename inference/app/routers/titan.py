from __future__ import annotations

from fastapi import APIRouter

from ai_factory.titan import detect_titan_status, write_hardware_markdown
from inference.app.config import get_settings

router = APIRouter(prefix="/titan", tags=["titan"])


@router.get("/status")
def titan_status() -> dict:
    settings = get_settings()
    return detect_titan_status(settings.repo_root)


@router.post("/hardware-doc")
def generate_hardware_doc() -> dict[str, str]:
    settings = get_settings()
    output_path = write_hardware_markdown(repo_root=settings.repo_root)
    return {"status": "generated", "path": str(output_path)}
