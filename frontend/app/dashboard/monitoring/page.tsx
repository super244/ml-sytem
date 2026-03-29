"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  getInstances,
  getOrchestrationSummary,
  runManagedInstanceAction,
  type InstanceSummary,
  type OrchestrationSummary,
} from "@/lib/api";

function progressPct(instance: InstanceSummary): number | null {
  const pct = instance.progress?.percent;
  return typeof pct === "number" ? Math.max(0, Math.min(100, pct * 100)) : null;
}

function statusClass(status: string): string {
  return `status-${status}`;
}

function formatTime(value?: string | null): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
}

export default function MonitoringPage() {
  const [instances, setInstances] = useState<InstanceSummary[]>([]);
  const [summary, setSummary] = useState<OrchestrationSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "running" | "completed" | "failed">("all");
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  async function refresh() {
    const [instancesRes, summaryRes] = await Promise.allSettled([
      getInstances(),
      getOrchestrationSummary(),
    ]);
    setInstances(instancesRes.status === "fulfilled" ? instancesRes.value : []);
    setSummary(summaryRes.status === "fulfilled" ? summaryRes.value : null);
    setLoading(false);
    setLastRefresh(new Date());
  }

  useEffect(() => {
    void refresh();
    const interval = setInterval(() => void refresh(), 2000);
    return () => clearInterval(interval);
  }, []);

  async function triggerAction(instance: InstanceSummary, action: string) {
    const key = `${instance.id}:${action}`;
    setBusyAction(key);
    try {
      await runManagedInstanceAction(instance.id, { action, start: true });
      await refresh();
    } finally {
      setBusyAction(null);
    }
  }

  const filtered = instances
    .filter((i) => filter === "all" || i.status === filter)
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at));

  const running = instances.filter((i) => i.status === "running").length;
  const completed = instances.filter((i) => i.status === "completed").length;
  const failed = instances.filter((i) => i.status === "failed").length;

  return (
    <div className="dashboard-content">
      {/* Header */}
      <div className="dash-page-header panel">
        <div className="dash-page-header-inner">
          <div>
            <span className="eyebrow">Lifecycle → Monitor</span>
            <h1 className="dash-page-title">Instance Monitor</h1>
            <p className="dash-page-desc">
              Real-time view of all training, evaluation, and inference instances.
              Auto-refreshes every 2 seconds.
            </p>
          </div>
          <div className="dash-header-actions">
            <span className="refresh-indicator">
              ◉ {lastRefresh.toLocaleTimeString()}
            </span>
            <button
              type="button"
              className="secondary-button small"
              onClick={() => void refresh()}
            >
              ⟳ Refresh
            </button>
          </div>
        </div>

        {/* Summary chips */}
        <div className="monitor-summary-row">
          <div className="monitor-chip total">
            <span className="monitor-chip-value">{instances.length}</span>
            <span className="monitor-chip-label">Total</span>
          </div>
          <div className="monitor-chip running">
            <span className="monitor-chip-value">{running}</span>
            <span className="monitor-chip-label">Running</span>
          </div>
          <div className="monitor-chip completed">
            <span className="monitor-chip-value">{completed}</span>
            <span className="monitor-chip-label">Completed</span>
          </div>
          <div className="monitor-chip failed">
            <span className="monitor-chip-value">{failed}</span>
            <span className="monitor-chip-label">Failed</span>
          </div>
          {summary?.task_status_counts?.running != null && (
            <div className="monitor-chip tasks">
              <span className="monitor-chip-value">{summary.task_status_counts.running}</span>
              <span className="monitor-chip-label">Active tasks</span>
            </div>
          )}
        </div>
      </div>

      {/* Filter Bar */}
      <div className="filter-bar panel">
        {(["all", "running", "completed", "failed"] as const).map((f) => (
          <button
            key={f}
            type="button"
            className={`filter-tab ${filter === f ? "active" : ""}`}
            onClick={() => setFilter(f)}
          >
            {f === "all" ? `All (${instances.length})` : `${f} (${instances.filter((i) => i.status === f).length})`}
          </button>
        ))}
      </div>

      {/* Instance Grid */}
      {loading && !instances.length && (
        <div className="dash-loading panel">
          <span className="dash-loading-icon">⟳</span>
          <span>Scanning instances…</span>
        </div>
      )}

      {!loading && !filtered.length && (
        <div className="dash-empty panel">
          <p>No {filter !== "all" ? filter : ""} instances found.</p>
          <Link href="/dashboard/training" className="primary-button small">
            ▲ Launch Training
          </Link>
        </div>
      )}

      <div className="monitor-instance-list">
        {filtered.map((instance) => {
          const pct = progressPct(instance);
          const topRec = instance.recommendations?.[0] ?? null;

          return (
            <div key={instance.id} className={`monitor-instance-card panel ${statusClass(instance.status)}`}>
              {/* Instance header */}
              <div className="monitor-instance-header">
                <div className="monitor-instance-id-group">
                  <span className={`monitor-status-dot ${statusClass(instance.status)}`} />
                  <span className="monitor-instance-type">{instance.type}</span>
                  <span className="monitor-instance-name">{instance.name}</span>
                </div>
                <div className="monitor-instance-meta">
                  <span className="monitor-learning-mode">
                    {instance.lifecycle.learning_mode ?? "—"}
                  </span>
                  <span className="monitor-env">{instance.environment.kind}</span>
                  <span className="monitor-updated">{formatTime(instance.updated_at)}</span>
                </div>
              </div>

              {/* Progress bar */}
              {pct !== null && (
                <div className="monitor-progress-track">
                  <div
                    className="monitor-progress-fill"
                    style={{ width: `${pct}%` }}
                    data-status={instance.status}
                  />
                  <span className="monitor-progress-label">{pct.toFixed(0)}%</span>
                </div>
              )}

              {/* Stage */}
              {instance.progress?.stage && (
                <div className="monitor-stage-row">
                  <span className="monitor-stage-label">
                    {instance.progress.stage}
                  </span>
                  {instance.progress.status_message && (
                    <span className="monitor-stage-msg">{instance.progress.status_message}</span>
                  )}
                </div>
              )}

              {/* Metrics */}
              {Object.keys(instance.metrics_summary).length > 0 && (
                <div className="monitor-metrics-row">
                  {Object.entries(instance.metrics_summary)
                    .filter(([, v]) => typeof v === "number")
                    .slice(0, 4)
                    .map(([key, val]) => (
                      <div key={key} className="monitor-metric">
                        <span className="monitor-metric-label">{key}</span>
                        <span className="monitor-metric-value">
                          {typeof val === "number" ? val.toFixed(3) : String(val)}
                        </span>
                      </div>
                    ))}
                </div>
              )}

              {/* Decision */}
              {instance.decision && (
                <div className="monitor-decision">
                  <span className="monitor-decision-icon">◆</span>
                  <span className="monitor-decision-action">{instance.decision.action}</span>
                  <span className="monitor-decision-rule">({instance.decision.rule})</span>
                </div>
              )}

              {/* Actions */}
              <div className="monitor-actions">
                <Link
                  href={`/runs/${instance.id}`}
                  className="secondary-button small"
                >
                  Inspect
                </Link>
                {topRec && (
                  <button
                    type="button"
                    className="primary-button small"
                    disabled={busyAction === `${instance.id}:${topRec.action}`}
                    onClick={() => void triggerAction(instance, topRec.action)}
                  >
                    {busyAction === `${instance.id}:${topRec.action}`
                      ? "Working…"
                      : topRec.action}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
