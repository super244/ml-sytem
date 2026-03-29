"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { AppNav } from "@/components/layout/app-nav";
import { LIFECYCLE_NAV, LAB_NAV } from "@/lib/routes";

type SidebarLayoutProps = {
  children: ReactNode;
};

export default function DashboardLayout({ children }: SidebarLayoutProps) {
  const pathname = usePathname();

  return (
    <div className="dashboard-app-frame">
      <AppNav />

      <div className="dashboard-body">
        {/* Lifecycle Sidebar */}
        <nav className="lifecycle-sidebar panel">
          <div className="sidebar-brand">
            <span className="sidebar-brand-icon">⬡</span>
            <div>
              <span className="sidebar-brand-kicker">AI-Factory</span>
              <span className="sidebar-brand-label">Lifecycle</span>
            </div>
          </div>

          <div className="sidebar-nav-group">
            {LIFECYCLE_NAV.map((item) => {
              const isActive =
                item.stage === "overview"
                  ? pathname === "/dashboard"
                  : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`sidebar-nav-item ${isActive ? "active" : ""}`}
                >
                  <span className="sidebar-nav-icon">{item.icon}</span>
                  <span className="sidebar-nav-label">{item.label}</span>
                  {isActive && <span className="sidebar-nav-indicator" />}
                </Link>
              );
            })}
          </div>

          <div className="sidebar-group-label" style={{ marginTop: "1.5rem", marginBottom: "0.5rem", opacity: 0.5, fontSize: "0.75rem", paddingLeft: "1.5rem", fontWeight: 600, letterSpacing: "0.05em" }}>V2 LAB</div>
          <div className="sidebar-nav-group">
            {LAB_NAV.map((item) => {
              const isActive = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`sidebar-nav-item ${isActive ? "active" : ""}`}
                >
                  <span className="sidebar-nav-icon">{item.icon}</span>
                  <span className="sidebar-nav-label">{item.label}</span>
                  {isActive && <span className="sidebar-nav-indicator" />}
                </Link>
              );
            })}
          </div>

          <div className="sidebar-footer">
            <Link href="/runs" className="sidebar-footer-link">
              All Runs ↗
            </Link>
            <Link href="/workspace" className="sidebar-footer-link">
              Workspace ↗
            </Link>
          </div>
        </nav>

        {/* Main content */}
        <main className="dashboard-main">
          {children}
        </main>
      </div>
    </div>
  );
}
