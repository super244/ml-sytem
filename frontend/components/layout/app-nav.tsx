"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { NAV_ITEMS, ROUTES } from "@/lib/routes";

export function AppNav() {
  const pathname = usePathname();

  function isActivePath(href: (typeof NAV_ITEMS)[number]["href"]): boolean {
    if (href === ROUTES.dashboard) {
      return pathname.startsWith(ROUTES.dashboard);
    }
    return pathname === href || pathname.startsWith(`${href}/`);
  }

  return (
    <nav className="app-nav">
      <div className="nav-brand-row">
        <Link className="brand-mark" href={ROUTES.solve}>
          <span className="brand-kicker">AI-Factory</span>
          <strong>Autonomous lab console</strong>
        </Link>
        <span className="nav-badge">Local-first lab</span>
      </div>

      <div className="nav-links">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`nav-link${isActivePath(item.href) ? " active" : ""}`}
          >
            {item.label}
          </Link>
        ))}
      </div>

      <div className="nav-actions">
        <Link className="ghost-button small" href={ROUTES.workspace}>
          Workspace
        </Link>
        <Link className="primary-button small" href={ROUTES.runs}>
          Runs
        </Link>
      </div>
    </nav>
  );
}
