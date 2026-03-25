import type { Route } from "next";

type NavItem = {
  href: Route;
  label: string;
};

export const ROUTES = {
  solve: "/",
  workspace: "/workspace",
  compare: "/compare",
  datasets: "/datasets",
  benchmarks: "/benchmarks",
  runs: "/runs",
} as const satisfies Record<string, Route>;

export const NAV_ITEMS = [
  { href: ROUTES.solve, label: "Solve" },
  { href: ROUTES.workspace, label: "Workspace" },
  { href: ROUTES.compare, label: "Compare" },
  { href: ROUTES.datasets, label: "Datasets" },
  { href: ROUTES.benchmarks, label: "Benchmarks" },
  { href: ROUTES.runs, label: "Runs" },
] as const satisfies readonly NavItem[];
