import type { Route } from "next";

type NavItem = {
  href: Route;
  label: string;
};

export const ROUTES = {
  dashboard: "/dashboard",
  training: "/dashboard/training",
  monitoring: "/dashboard/monitoring",
  evaluate: "/dashboard/evaluate",
  finetune: "/dashboard/finetune",
  deploy: "/dashboard/deploy",
  inference: "/dashboard/inference",
  solve: "/solve",
  workspace: "/workspace",
  compare: "/compare",
  datasets: "/datasets",
  benchmarks: "/benchmarks",
  runs: "/runs",
} as const satisfies Record<string, Route>;

export const NAV_ITEMS = [
  { href: ROUTES.dashboard, label: "Dashboard" },
  { href: ROUTES.solve, label: "Solve" },
  { href: ROUTES.workspace, label: "Workspace" },
  { href: ROUTES.compare, label: "Compare" },
  { href: ROUTES.datasets, label: "Datasets" },
  { href: ROUTES.benchmarks, label: "Benchmarks" },
  { href: ROUTES.runs, label: "Runs" },
] as const satisfies readonly NavItem[];

export const LIFECYCLE_NAV = [
  { href: ROUTES.dashboard, label: "Overview", icon: "◈", stage: "overview" },
  { href: ROUTES.training, label: "Train", icon: "▲", stage: "train" },
  { href: ROUTES.monitoring, label: "Monitor", icon: "◉", stage: "monitor" },
  { href: ROUTES.evaluate, label: "Evaluate", icon: "◆", stage: "evaluate" },
  { href: ROUTES.finetune, label: "Finetune", icon: "⟳", stage: "finetune" },
  { href: ROUTES.deploy, label: "Deploy", icon: "⬆", stage: "deploy" },
  { href: ROUTES.inference, label: "Inference", icon: "◎", stage: "inference" },
] as const;
