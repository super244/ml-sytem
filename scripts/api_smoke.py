from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

from common import emit_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight smoke check against the local Atlas API.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the running Atlas API.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the verifier endpoint check.",
    )
    parser.add_argument(
        "--include-generate",
        action="store_true",
        help="Also run a single generation request. This may require local model assets.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    checks: dict[str, str] = {}
    details: dict[str, Any] = {}
    failed = False

    for path in (
        "/v1/health",
        "/v1/status",
        "/v1/models",
        "/v1/datasets",
        "/v1/benchmarks",
        "/v1/runs",
    ):
        endpoint = f"{base_url}{path}"
        try:
            payload = request_json("GET", endpoint)
            checks[path] = "ok"
            if path == "/v1/status":
                details["status_title"] = payload.get("title")
                details["status_version"] = payload.get("version")
            elif path == "/v1/models":
                details["num_models"] = len(payload.get("models", []))
            elif path == "/v1/datasets":
                details["num_datasets"] = payload.get("summary", {}).get("num_datasets")
            elif path == "/v1/benchmarks":
                details["num_benchmarks"] = len(payload.get("benchmarks", []))
            elif path == "/v1/runs":
                details["num_runs"] = len(payload.get("runs", []))
        except urllib.error.URLError as exc:
            checks[path] = f"failed: {exc.reason}"
            failed = True
        except Exception as exc:  # pragma: no cover - defensive CLI path
            checks[path] = f"failed: {exc}"
            failed = True

    if not args.skip_verify:
        endpoint = f"{base_url}/v1/verify"
        try:
            payload = request_json(
                "POST",
                endpoint,
                {
                    "reference_answer": "1/2",
                    "candidate_answer": "1/2",
                    "prediction_text": "Final Answer: 1/2",
                    "step_checks": [],
                },
            )
            checks["/v1/verify"] = "ok"
            details["verify_equivalent"] = payload.get("equivalent")
            details["verify_error_type"] = payload.get("error_type")
        except urllib.error.URLError as exc:
            checks["/v1/verify"] = f"failed: {exc.reason}"
            failed = True
        except Exception as exc:  # pragma: no cover - defensive CLI path
            checks["/v1/verify"] = f"failed: {exc}"
            failed = True

    if args.include_generate:
        endpoint = f"{base_url}/v1/generate"
        try:
            payload = request_json(
                "POST",
                endpoint,
                {
                    "question": "Evaluate \\int_0^1 x dx.",
                    "model_variant": "finetuned",
                    "compare_to_base": False,
                    "compare_to_model": None,
                    "prompt_preset": "atlas_rigorous",
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_new_tokens": 256,
                    "show_reasoning": True,
                    "difficulty_target": "easy",
                    "num_samples": 1,
                    "use_calculator": True,
                    "solver_mode": "rigorous",
                    "output_format": "text",
                    "use_cache": False,
                },
            )
            checks["/v1/generate"] = "ok"
            details["generated_final_answer"] = payload.get("final_answer")
        except urllib.error.URLError as exc:
            checks["/v1/generate"] = f"failed: {exc.reason}"
            failed = True
        except Exception as exc:  # pragma: no cover - defensive CLI path
            checks["/v1/generate"] = f"failed: {exc}"
            failed = True

    payload = {
        "base_url": base_url,
        "checks": checks,
        "details": details,
    }
    emit_payload(payload, as_json=args.json)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
