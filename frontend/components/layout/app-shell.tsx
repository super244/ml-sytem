import type { ReactNode } from "react";

import { AppNav } from "@/components/layout/app-nav";

type AppShellProps = {
  children: ReactNode;
  density?: "compact" | "balanced" | "expanded";
  surfaceMode?: "focus" | "research" | "verification";
};

export function AppShell({ children, density, surfaceMode }: AppShellProps) {
  return (
    <div className="app-frame" data-density={density} data-surface-mode={surfaceMode}>
      <AppNav />
      <div className="page-stage">{children}</div>
    </div>
  );
}
