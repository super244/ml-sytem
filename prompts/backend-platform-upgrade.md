# Backend Platform Upgrade Prompt

Expand and harden the AI-Factory backend so it matches the frontend smoothly, exposes clear contracts, and keeps services operational under partial failure.

## Primary Goal

Make the FastAPI and service layer feel deliberate: well-typed, well-routed, observably healthy, and easy for the frontend and agents to consume.

## Read This First

- `prompts/shared-repo-context.md`
- `inference/app/main.py`
- `inference/app/config.py`
- `inference/app/dependencies.py`
- `inference/app/services/`
- `inference/app/routers/`
- `ai_factory/titan.py`
- `ai_factory/core/orchestration/service.py`
- `tests/test_*api*.py`
- `tests/test_web_interface.py`

## Scope

- Router registration, route consistency, and degraded-mode handling
- Service boundaries, typed schemas, and response stability
- Metadata, mission control, Titan, orchestration, and instance-management integration
- Error surfaces that help the frontend recover gracefully
- Tests that lock in contracts users actually depend on

## High-Value Targets

- Remove duplicate router registration or route alias drift
- Tighten Titan and mission-control payload consistency
- Reduce backend/frontend shape mismatches
- Improve fault reporting for metadata, prompts, datasets, and orchestration state

## Definition Of Done

- The backend has a clean route surface with no accidental duplication.
- Services expose stable shapes that the frontend can trust.
- Broken dependencies degrade explicitly instead of failing opaquely.
- Relevant API tests pass.
