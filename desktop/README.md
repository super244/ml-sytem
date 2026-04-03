# Desktop Shell

This folder contains the desktop surfaces for AI-Factory.

## Why it exists

- The desktop surface is intentionally thin.
- It should reuse the same backend contracts as the CLI, TUI, and web dashboard.
- The Electron shell hosts the existing control center UI and API instead of introducing a second orchestration stack.
- The native macOS scaffold lives alongside it so we can move toward a SwiftUI-first shell without breaking the current path.

## Current behavior

- `main.js` opens the dashboard route from `AI_FACTORY_DESKTOP_URL` or `http://127.0.0.1:3000/dashboard`.
- `preload.js` exposes a minimal desktop capability object for future renderer-side feature checks.
- No desktop-specific orchestration logic lives here. All lifecycle actions still flow through the shared backend.

## Native macOS scaffold

- `macos/AIFactoryNative` is a standalone Swift package containing a SwiftUI control center.
- `swift run` inside that package launches the native shell on macOS.
- Set `AI_FACTORY_DESKTOP_NATIVE=1` before starting `ai_factory.interfaces.desktop.DesktopInterface.run()` to prefer the SwiftUI shell when the native package is present.
- `desktop/package.json` exposes `native:build` and `native:run` shortcuts for the Swift package.

## Intended next steps

1. Add a launch workflow that starts the FastAPI backend and Next.js frontend together.
2. Replace the raw URL bootstrap with health checks and onboarding.
3. Add native desktop affordances such as tray status, notifications, and log-file opening.
