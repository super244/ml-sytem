# Desktop Shell

This folder is the starter shell for the AI-FACTORY desktop app.

## Why it exists

- The desktop surface is intentionally thin.
- It should reuse the same backend contracts as the CLI, TUI, and web dashboard.
- The Electron shell simply hosts the existing control center UI and API instead of introducing a second orchestration stack.

## Current behavior

- `main.js` opens the workspace route from `AI_FACTORY_DESKTOP_URL` or `http://127.0.0.1:3000/workspace`.
- `preload.js` exposes a minimal desktop capability object for future renderer-side feature checks.
- No desktop-specific orchestration logic lives here. All lifecycle actions still flow through the shared backend.

## Intended next steps

1. Add a launch workflow that starts the FastAPI backend and Next.js frontend together.
2. Replace the raw URL bootstrap with health checks and onboarding.
3. Add native desktop affordances such as tray status, notifications, and log-file opening.
