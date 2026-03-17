# Frontend Layer

The frontend is a multi-route research product for solving, comparing, and analyzing the math-specialist model. It is designed to make inference controls, benchmark context, and model metadata feel like first-class product features.

## Routes

- `/`: solve workspace
- `/compare`: side-by-side model comparison
- `/datasets`: dataset and pack explorer
- `/benchmarks`: benchmark summary and slice explorer
- `/runs`: training/evaluation run viewer

## Core Components

- `components/chat-shell.tsx`: primary solve workspace
- `components/compare-lab.tsx`: comparison UI
- `components/datasets-view.tsx`: dataset explorer
- `components/benchmarks-view.tsx`: benchmark explorer
- `components/runs-view.tsx`: run viewer
- `components/panels/`: reusable metric, candidate, and model UI
- `hooks/use-lab-metadata.ts`: typed metadata fetching
- `lib/api.ts`: typed API client
- `app/globals.css`: visual system and interaction styling

## Product Features

- math rendering with KaTeX
- solver-mode and prompt-preset controls
- reasoning visibility toggle
- sample-count and structured-output controls
- verifier/cache/latency badges
- candidate inspector
- dataset-backed prompt examples
- model and run metadata cards

## Example Commands

```bash
cd frontend
npm install
npm run typecheck
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```
