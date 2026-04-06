# Web Dashboard Upgrade Prompt

Upgrade the AI-Factory Next.js web dashboard. Your objective is to transform it from a prototype into a visually coherent, highly responsive, trustworthy, and fully-wired production interface. Every interaction must trigger a real backend effect or explicitly handle a degraded state.

## Primary Goal

Ensure the frontend feels fast, reactive, and reliable. Eradicate all mock data, hardcoded states, and "dead" buttons. The UI must be an exact reflection of the `inference/app` and `orchestration` backend state.

## Read This First (Mandatory Ingestion)

- `prompts/shared-repo-context.md`
- `frontend/app/dashboard/page.tsx` (Main entry)
- `frontend/app/dashboard/layout.tsx` (App Shell)
- `frontend/components/layout/app-shell.tsx`
- `frontend/components/layout/app-nav.tsx`
- `frontend/lib/api.ts` (API Client and fetch wrappers)
- `frontend/lib/routes.ts` (Route definitions)
- `frontend/lib/titan-schema.ts` (TypeScript bindings for Rust engine)
- `frontend/hooks/use-lab-metadata.ts`
- `inference/app/main.py`
- `inference/app/routers/`
- `frontend/REDESIGN_PROMPT.md` (Design inspiration, NOT a binding spec)

## Scope & Execution Directives

1. **State Management & Caching**:
   - Utilize React Server Components for initial data fetching where SEO/performance demands it.
   - Use strict Client Components (`"use client"`) for interactive data mutation, coupled with optimistic UI updates (e.g., via SWR or React Query).
2. **Strict Typing (Zod)**:
   - Every API response must be validated at runtime using Zod schemas. Do not blindly cast `any` to TypeScript interfaces. 
   - Ensure the TypeScript types map 1:1 to the Pydantic backend models.
3. **UX Resiliency & Form Affordances**:
   - Implement clear loading states (skeletons or spinners) for all asynchronous actions.
   - All forms must have proper validation, disabled submit states during processing, and explicit success/error toast notifications.
   - Network failures must display informative "Degraded Mode" banners, not blank white screens.
4. **Navigation & Consistency**:
   - Ensure navigation across dashboard, runs, datasets, benchmarks, compare, training, and workspace is visually seamless.
   - Do not rebuild the whole visual system (Tailwind) if targeted fixes unlock more value.

## Non-Goals

- Do not introduce fake metrics or fake backend actions to make the UI look complete. If the backend doesn't support it, gray it out or remove it.

## Definition Of Done

- Critical user actions are discoverable and wired to real `fetch` mutations.
- The UI accurately parses backend HTTP 503s into friendly degraded-state messaging.
- No obvious dead buttons, orphan routes, or `any` type casts exist in touched surfaces.
- Both mobile and desktop layouts remain fully responsive.