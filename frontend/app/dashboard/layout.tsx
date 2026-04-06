'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import type { ReactNode } from 'react';

import { AppNav } from '@/components/layout/app-nav';
import { LIFECYCLE_NAV, LAB_NAV } from '@/lib/routes';

type SidebarLayoutProps = {
  children: ReactNode;
};

export default function DashboardLayout({ children }: SidebarLayoutProps) {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="dashboard-app-frame">
      <AppNav />

      <div className={`dashboard-body ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        {/* Lifecycle Sidebar */}
        <nav className={`lifecycle-sidebar panel ${sidebarOpen ? '' : 'collapsed'}`}>
          <div className="sidebar-brand">
            <span className="sidebar-brand-icon">⬡</span>
            {sidebarOpen && (
              <div>
                <span className="sidebar-brand-kicker">AI-Factory</span>
                <span className="sidebar-brand-label">Lifecycle</span>
              </div>
            )}
          </div>

          <button
            className="sidebar-toggle-btn"
            onClick={() => setSidebarOpen((o) => !o)}
            title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? '‹' : '›'}
          </button>

          <div className="sidebar-nav-group">
            {LIFECYCLE_NAV.map((item) => {
              const isActive =
                item.stage === 'overview'
                  ? pathname === '/dashboard'
                  : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
                  title={!sidebarOpen ? item.label : undefined}
                >
                  <span className="sidebar-nav-icon">{item.icon}</span>
                  {sidebarOpen && <span className="sidebar-nav-label">{item.label}</span>}
                  {isActive && sidebarOpen && <span className="sidebar-nav-indicator" />}
                </Link>
              );
            })}
          </div>

          {sidebarOpen && (
            <div
              className="sidebar-group-label"
              style={{
                marginTop: '1.5rem',
                marginBottom: '0.5rem',
                opacity: 0.5,
                fontSize: '0.75rem',
                paddingLeft: '1.5rem',
                fontWeight: 600,
                letterSpacing: '0.05em',
              }}
            >
              V2 LAB
            </div>
          )}
          <div className="sidebar-nav-group">
            {LAB_NAV.map((item) => {
              const isActive = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
                  title={!sidebarOpen ? item.label : undefined}
                >
                  <span className="sidebar-nav-icon">{item.icon}</span>
                  {sidebarOpen && <span className="sidebar-nav-label">{item.label}</span>}
                  {isActive && sidebarOpen && <span className="sidebar-nav-indicator" />}
                </Link>
              );
            })}
          </div>

          {sidebarOpen && (
            <div className="sidebar-footer">
              <Link href="/runs" className="sidebar-footer-link">
                All Runs ↗
              </Link>
              <Link href="/workspace" className="sidebar-footer-link">
                Workspace ↗
              </Link>
            </div>
          )}
        </nav>

        {/* Main content */}
        <main className="dashboard-main">{children}</main>
      </div>
    </div>
  );
}
