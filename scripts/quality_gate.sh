#!/usr/bin/env bash
set -euo pipefail

echo "Running quality gate checks..."

coverage_json="artifacts/metrics/coverage.json"
bandit_json="artifacts/metrics/bandit.json"

pytest --cov=ai_factory --cov-report="json:${coverage_json}" -q
bandit -r ai_factory -f json -o "${bandit_json}" >/dev/null

coverage=$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("artifacts/metrics/coverage.json").read_text())
print(float(payload.get("totals", {}).get("percent_covered", 0.0)))
PY
)

issues=$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("artifacts/metrics/bandit.json").read_text())
print(len(payload.get("results", [])))
PY
)

python - <<PY
coverage = float("${coverage}")
issues = int("${issues}")
if coverage < 80.0:
    raise SystemExit(f"Coverage gate failed: {coverage:.2f}% < 80.0%")
if issues > 25:
    raise SystemExit(f"Security gate failed: {issues} issues > 25")
print(f"Quality gates passed (coverage={coverage:.2f}%, security_issues={issues})")
PY

