import type { ReactNode } from "react";

import { MetricBadge } from "@/components/panels/metric-badge";

type PageHeaderMetric = {
  label: string;
  value: string;
  tone?: "default" | "accent" | "secondary";
};

type PageHeaderProps = {
  eyebrow: string;
  title: string;
  description: string;
  metrics?: PageHeaderMetric[];
  actions?: ReactNode;
};

export function PageHeader({
  eyebrow,
  title,
  description,
  metrics = [],
  actions,
}: PageHeaderProps) {
  return (
    <section className="relative overflow-hidden bg-white/80 backdrop-blur-2xl border border-gray-200/60 rounded-3xl shadow-sm p-8 transition-all duration-300 hover:shadow-md hover:border-gray-300/60 group">
      <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-gray-200/80 to-transparent"></div>
      <div className="absolute -bottom-24 -right-16 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl group-hover:bg-emerald-500/15 transition-colors duration-500"></div>
      
      <div className="relative z-10 flex flex-col md:flex-row justify-between items-start gap-8">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center px-3 py-1 rounded-full bg-emerald-50 border border-emerald-100/50 text-emerald-700 text-xs font-semibold tracking-wider uppercase mb-2">
            {eyebrow}
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-gray-900 leading-tight">
            {title}
          </h1>
          <p className="text-lg text-gray-600 leading-relaxed max-w-2xl">
            {description}
          </p>
        </div>
        {actions ? (
          <div className="flex flex-wrap items-center gap-3 shrink-0">
            {actions}
          </div>
        ) : null}
      </div>
      
      {metrics.length ? (
        <div className="relative z-10 flex flex-wrap gap-4 mt-8 pt-6 border-t border-gray-100">
          {metrics.map((metric) => (
            <MetricBadge
              key={`${metric.label}-${metric.value}`}
              label={metric.label}
              value={metric.value}
              tone={metric.tone}
            />
          ))}
        </div>
      ) : null}
    </section>
  );
}
