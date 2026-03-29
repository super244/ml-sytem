from __future__ import annotations

from ai_factory.interfaces.web import main as web_main
from ai_factory.interfaces.web.main import WebInterface


def test_web_interface_injects_api_base_url_for_frontend(tmp_path, monkeypatch):
    (tmp_path / "frontend").mkdir()
    interface = WebInterface(repo_root=tmp_path)

    captured: dict[str, object] = {}

    def fake_run(cmd, cwd=None, env=None, check=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["check"] = check

    monkeypatch.setattr(web_main.subprocess, "run", fake_run)

    interface.run_frontend(port=3100)

    assert captured["cwd"] == tmp_path / "frontend"
    assert captured["env"]["NEXT_PUBLIC_API_BASE_URL"] == interface.backend_url
    assert captured["check"] is True
