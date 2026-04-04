# Web Dashboard Upgrade Prompt

Upgrade the AI-Factory web dashboard until it feels production-grade: visually coherent, responsive, trustworthy, and fully wired to backend capabilities. Every important button should click through to a real effect or show a clear degraded/error state.

## Primary Goal

Make the dashboard and adjacent web surfaces feel complete, fast, and reliable rather than prototype-like.

## Read This First

- `prompts/shared-repo-context.md`
- `frontend/app/dashboard/page.tsx`
- `frontend/app/dashboard/layout.tsx`
- `frontend/components/layout/app-shell.tsx`
- `frontend/components/layout/app-nav.tsx`
- `frontend/lib/api.ts`
- `frontend/lib/routes.ts`
- `frontend/lib/titan-schema.ts`
- `frontend/hooks/use-lab-metadata.ts`
- `inference/app/main.py`
- `inference/app/routers/`
- `frontend/REDESIGN_PROMPT.md` as design inspiration, not as a binding spec

## Scope

- Dashboard UX, route consistency, loading/error states, and action wiring
- Navigation quality across dashboard, runs, datasets, benchmarks, compare, training, and workspace
- Form affordances, disabled states, optimistic updates, notices, and failure messaging
- Typed API alignment between frontend and backend responses
- Accessibility, responsive layout, and performance-sensitive rendering

## Non-Goals

- Do not introduce fake metrics or fake backend actions to make the UI look complete.
- Do not rebuild the whole visual system if targeted fixes unlock more value.

## Definition Of Done

- Critical user actions are discoverable and produce a real navigation, mutation, or explicit degraded-state response.
- No obvious dead buttons, orphan routes, or broken API contracts remain in the touched surfaces.
- Dashboard cards, metrics, and Titan/runtime displays reflect backend reality.
- Mobile and desktop layouts both remain usable.
- Relevant frontend tests or type checks pass.
