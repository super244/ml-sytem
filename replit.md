# AI-FACTORY v2.0

## Overview
A React + TypeScript frontend application — an AI operations dashboard (Mission Control) for monitoring and managing ML training jobs, models, inference, datasets, AutoML, agents, and GPU clusters.

## Tech Stack
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite (with @vitejs/plugin-react-swc)
- **Routing**: React Router v6
- **UI Components**: Radix UI primitives + shadcn/ui
- **Styling**: Tailwind CSS
- **State/Data**: TanStack React Query
- **Forms**: React Hook Form + Zod
- **Charts**: Recharts
- **Animations**: Framer Motion

## Project Structure
- `src/pages/` — Route-level page components (Dashboard, Monitoring, ModelRegistry, InferenceChat, DatasetStudio, AutoMLExplorer, AgentMonitor, ClusterPage, Solve, SettingsPage)
- `src/components/` — Shared UI components
- `src/hooks/` — Custom React hooks
- `src/lib/` — Utility functions
- `src/data/` — Static/mock data

## Running the App
- **Dev server**: `npm run dev` (port 5000)
- **Build**: `npm run build`
- **Preview**: `npm run preview`

## Replit Configuration
- Workflow: "Start application" runs `npm run dev` on port 5000 (webview)
- Deployment: `npm run build` → `node ./dist/index.cjs`
- Vite configured with `host: "0.0.0.0"` and `allowedHosts: true` for Replit proxy compatibility
- Removed Lovable-specific `lovable-tagger` plugin from Vite config
