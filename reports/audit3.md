# Audit 3 (Current Working Tree As-Is)

Date: 2026-04-04  
Workspace: `/Users/luca/Projects/ai-factory`  
Mode: Read-only audit (no file edits during audit execution)

## Scope

This audit was run against the current working tree state (including uncommitted changes), not `HEAD`.

## Progress Summary

1. Repository inventory completed.
2. Hotspot scan completed (runtime/degraded/stub/error surfaces).
3. Python static and test checks completed.
4. Frontend type checks executed.
5. Titan Rust crate tests executed.

## Inventory Snapshot

- Total files: `683`
- Python files: `282`
- TypeScript/TSX files: `52`
- Rust files: `18`
- Markdown files: `76`

## Commands Run and Outcomes

1. `ruff check .`  
Status: Pass

2. `mypy .`  
Status: Pass (`Success: no issues found in 225 source files`)

3. `pytest -q`  
Status: Pass (`239 passed`)

4. `cd frontend && npm run typecheck`  
Status: Fail (`4` TypeScript errors)

5. `cd ai_factory_titan && cargo test`  
Status: Fail (compile/test API mismatches)

## Findings (By Severity)

### P0

1. Titan crate build/test is not stable in current tree.
- Root exports do not match binary/tests usage.
- Files:
  - `ai_factory_titan/src/lib.rs`
  - `ai_factory_titan/src/bin/titan-status.rs`
  - `ai_factory_titan/tests/engine_runtime.rs`
  - `ai_factory_titan/src/sampler.rs`

2. Frontend type contracts are currently broken.
- `cluster` page reads fields not present in typed Titan runtime shape.
- `Link` typing issues from dynamic union/string `href`.
- Files:
  - `frontend/app/dashboard/cluster/page.tsx`
  - `frontend/lib/titan-schema.ts`
  - `frontend/app/dashboard/layout.tsx`
  - `frontend/app/dashboard/page.tsx`

### P1

3. Deployment targets remain intentionally degraded/stubbed.
- Multiple targets still raise runtime errors as "not wired" behaviors.
- File:
  - `ai_factory/platform/deployment/targets.py`

4. Titan runtime descriptors exist, but execution remains Python-first.
- Titan status/runtime metadata is implemented.
- Inference generation path still runs via Transformers/PyTorch.
- Files:
  - `ai_factory/titan.py`
  - `ai_factory_titan/src/runtime.rs`
  - `inference/app/model_loader.py`
  - `inference/app/generation.py`

### P2

5. High active churn in working tree.
- Large in-flight changes in Titan + prompts + build artifacts increase merge/regression risk.

## Current Risk Profile

- Python quality bar: healthy (`ruff`, `mypy`, `pytest` all green).
- Frontend release confidence: blocked by TypeScript failures.
- Titan engine confidence: blocked by Rust compile/test failures.

## Recommended Immediate Priorities

1. Restore `ai_factory_titan` compile/test health first (exports, runtime types, sampler API consistency).
2. Restore frontend typecheck by reconciling Titan schema and dashboard usage.
3. Decide whether deployment targets should remain explicit degraded contracts or move to live publisher implementations.
4. Keep a stable contract test for Titan payload shape shared across Python, API, and frontend.

## Notes

- This report reflects the repository state as audited.
- No files were modified during the audit process itself.
