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
    <section className="panel page-hero">
      <div className="page-hero-layout">
        <div className="page-hero-copy">
          <div className="eyebrow">{eyebrow}</div>
          <h1 className="hero-title">{title}</h1>
          <p className="hero-copy">{description}</p>
        </div>
        {actions ? <div className="hero-actions">{actions}</div> : null}
      </div>
      {metrics.length ? (
        <div className="hero-metrics">
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
