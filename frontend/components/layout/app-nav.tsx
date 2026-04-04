'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { NAV_ITEMS, ROUTES } from '@/lib/routes';

export function AppNav() {
  const pathname = usePathname();

  function isActivePath(href: (typeof NAV_ITEMS)[number]['href']): boolean {
    if (href === ROUTES.dashboard) {
      return pathname.startsWith(ROUTES.dashboard);
    }
    return pathname === href || pathname.startsWith(`${href}/`);
  }

  return (
    <nav className="sticky top-4 z-50 flex items-center justify-between gap-4 px-6 py-4 mx-auto w-full bg-white/80 backdrop-blur-2xl border border-gray-200/60 shadow-sm rounded-3xl transition-all duration-300">
      <div className="flex items-center gap-4 shrink-0">
        <Link className="flex flex-col hover:opacity-80 transition-opacity" href={ROUTES.solve}>
          <span className="text-[0.68rem] font-bold uppercase tracking-widest text-emerald-700">
            AI-Factory
          </span>
          <strong className="text-sm tracking-tight text-gray-900">Autonomous lab console</strong>
        </Link>
        <span className="hidden sm:inline-flex items-center px-3 py-1 rounded-full border border-gray-200 bg-white/90 text-gray-500 text-xs font-semibold uppercase tracking-wider">
          Local-first lab
        </span>
      </div>

      <div className="hidden md:flex flex-wrap justify-center gap-2 flex-1">
        {NAV_ITEMS.map((item) => {
          const active = isActivePath(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                active
                  ? 'bg-emerald-50 text-emerald-800 border border-emerald-200 shadow-sm'
                  : 'text-gray-600 border border-transparent hover:text-gray-900 hover:bg-gray-50 hover:border-gray-200'
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </div>

      <div className="flex items-center justify-end gap-3 shrink-0">
        <Link
          className="hidden sm:inline-flex items-center justify-center px-4 py-2 text-sm font-medium text-gray-700 bg-white/60 border border-gray-200 border-dashed rounded-full hover:bg-gray-50 hover:border-emerald-200 transition-colors"
          href={ROUTES.workspace}
        >
          Workspace
        </Link>
        <Link
          className="inline-flex items-center justify-center px-5 py-2 text-sm font-semibold text-white bg-gradient-to-br from-emerald-600 to-emerald-800 rounded-full shadow-md shadow-emerald-500/20 hover:shadow-lg hover:shadow-emerald-500/30 hover:-translate-y-0.5 transition-all"
          href={ROUTES.runs}
        >
          Runs
        </Link>
      </div>
    </nav>
  );
}
