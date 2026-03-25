import type { ReactNode } from "react";

import { AppNav } from "@/components/layout/app-nav";

type AppShellProps = {
  children: ReactNode;
};

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="app-frame">
      <AppNav />
      <div className="page-stage">{children}</div>
    </div>
  );
}
