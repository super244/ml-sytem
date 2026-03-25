"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { NAV_ITEMS, ROUTES } from "@/lib/routes";

export function AppNav() {
  const pathname = usePathname();
  return (
    <nav className="app-nav">
      <div className="nav-brand-row">
        <Link className="brand-mark" href={ROUTES.solve}>
          <span className="brand-kicker">Atlas Math Lab</span>
          <strong>Competitive reasoning platform</strong>
        </Link>
        <span className="nav-badge">Local-first specialist stack</span>
      </div>

      <div className="nav-links">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`nav-link${pathname === item.href ? " active" : ""}`}
          >
            {item.label}
          </Link>
        ))}
      </div>

      <div className="nav-actions">
        <Link className="ghost-button small" href={ROUTES.workspace}>
          Command Center
        </Link>
        <Link className="primary-button small" href={ROUTES.compare}>
          Compare Models
        </Link>
      </div>
    </nav>
  );
}
