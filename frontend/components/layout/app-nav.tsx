"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Solve" },
  { href: "/compare", label: "Compare" },
  { href: "/datasets", label: "Datasets" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/runs", label: "Runs" },
];

export function AppNav() {
  const pathname = usePathname();
  return (
    <nav className="app-nav">
      <div className="nav-brand-row">
        <Link className="brand-mark" href="/">
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
        <Link className="ghost-button small" href="/">
          New Session
        </Link>
        <Link className="primary-button small" href="/compare">
          Compare Models
        </Link>
      </div>
    </nav>
  );
}
