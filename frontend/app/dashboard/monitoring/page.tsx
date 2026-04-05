'use client';

import Link from 'next/link';
import { useEffect, useRef, useState, useCallback, useMemo } from 'react';

import {
  getInstances,
  getInstanceDetail,
  getOrchestrationSummary,
  getClusterNodes,
  runManagedInstanceAction,
  type ClusterNodeHardware,
  type InstanceSummary,
  type InstanceDetail,
  type OrchestrationSummary,
} from '@/lib/api';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function progressPct(instance: InstanceSummary | InstanceDetail): number | null {
  const pct = instance.progress?.percent;
  return typeof pct === 'number' ? Math.max(0, Math.min(100, pct * 100)) : null;
}

function statusColor(status: string): string {
  if (status === 'running') return 'var(--accent)';
  if (status === 'completed') return '#5da85d';
  if (status === 'failed') return 'var(--danger)';
  return 'var(--muted)';
}

function formatTime(value?: string | null): string {
  if (!value) return '—';
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
}

function elapsed(created: string): string {
  const ms = Date.now() - new Date(created).getTime();
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

// ─── Mini Sparkline (pure SVG, no lib) ────────────────────────────────────────

function Sparkline({
  values,
  color = 'var(--accent)',
  height = 40,
  width = 120,
}: {
  values: number[];
  color?: string;
  height?: number;
  width?: number;
}) {
  if (values.length < 2) return <span style={{ opacity: 0.3, fontSize: '0.75rem' }}>no data</span>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pts = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ overflow: 'visible' }}
    >
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      <circle
        cx={values.length > 0 ? ((values.length - 1) / (values.length - 1)) * width : 0}
        cy={height - ((values[values.length - 1] - min) / range) * (height - 4) - 2}
        r="2.5"
        fill={color}
      />
    </svg>
  );
}

// ─── Live Terminal Component ───────────────────────────────────────────────────

function LiveTerminal({ logText }: { logText: string }) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const lines = useMemo(() => logText
    ? logText
        .split('\n')
        .filter((line) => line.trim())
        .slice(-200)
    : ['No log output yet — instance is initializing…'], [logText]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines]);

  return (
    <div
      style={{
        background: '#0d1117',
        border: '1px solid var(--line)',
        borderRadius: '8px',
        padding: '1rem',
        height: '320px',
        overflowY: 'auto',
        fontFamily: 'var(--font-mono, monospace)',
        fontSize: '0.78rem',
        lineHeight: '1.6',
        color: '#c9d1d9',
      }}
    >
      {lines.map((line, idx) => {
        const isError = /error|exception|traceback|failed/i.test(line);
        const isWarn = /warn|token|huggingface|gated/i.test(line);
        const isGood = /loaded|success|complete|saved|done/i.test(line);
        const color = isError ? '#ff7b72' : isWarn ? '#e3b341' : isGood ? '#7ee787' : '#c9d1d9';
        return (
          <div key={idx} style={{ color, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {line}
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}

// ─── Deep Instance Detail Panel ────────────────────────────────────────────────

function InstanceDetailPanel({ instanceId, onClose }: { instanceId: string; onClose: () => void }) {
  const [detail, setDetail] = useState<InstanceDetail | null>(null);
  const [metricHistory, setMetricHistory] = useState<Record<string, number[]>>({});
  const [busyAction, setBusyAction] = useState<string | null>(null);

  const isRunning = detail?.status === 'running';

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const d = await getInstanceDetail(instanceId);
        if (!active) return;
        setDetail(d);
        // Accumulate metric history for sparklines
        setMetricHistory((prev) => {
          const next = { ...prev };
          for (const [k, v] of Object.entries(d.metrics_summary)) {
            if (typeof v === 'number') {
              next[k] = [...(prev[k] ?? []), v].slice(-60);
            }
          }
          return next;
        });
      } catch {
        // silently retry
      }
    }
    void poll();
    const interval = setInterval(() => void poll(), isRunning ? 3000 : 10_000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [instanceId, isRunning]);

  async function triggerAction(action: string) {
    setBusyAction(action);
    try {
      await runManagedInstanceAction(instanceId, { action, start: true });
      const d = await getInstanceDetail(instanceId);
      setDetail(d);
    } finally {
      setBusyAction(null);
    }
  }

  const pct = detail ? progressPct(detail) : null;
  const numericMetrics = Object.entries(detail?.metrics_summary ?? {}).filter(
    ([, v]) => typeof v === 'number',
  );
  const logText = [detail?.logs?.stdout ?? '', detail?.logs?.stderr ?? '']
    .filter(Boolean)
    .join('\n');

  return (
    <div
      className="instance-detail-panel panel"
      style={{
        marginBottom: '1.5rem',
        border: `2px solid ${detail ? statusColor(detail.status) : 'var(--line)'}`,
      }}
    >
      {/* Detail Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '1.5rem',
        }}
      >
        <div>
          {detail ? (
            <>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  marginBottom: '0.4rem',
                }}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: '50%',
                    background: statusColor(detail.status),
                    display: 'inline-block',
                    flexShrink: 0,
                  }}
                />
                <code style={{ fontSize: '0.75rem', opacity: 0.5 }}>{detail.id}</code>
                <span
                  style={{
                    fontSize: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid var(--line)',
                    borderRadius: 4,
                    padding: '0.1rem 0.4rem',
                  }}
                >
                  {detail.type}
                </span>
              </div>
              <h2 style={{ margin: 0, fontSize: '1.3rem' }}>{detail.name}</h2>
              <p style={{ margin: '0.3rem 0 0', fontSize: '0.85rem', opacity: 0.65 }}>
                {detail.lifecycle?.learning_mode ?? '—'} · {detail.environment.kind} · started{' '}
                {formatTime(detail.created_at)} · running {elapsed(detail.created_at)}
              </p>
            </>
          ) : (
            <div className="dash-loading">
              <span>⟳</span> Loading instance...
            </div>
          )}
        </div>
        <button className="ghost-button small" onClick={onClose}>
          ✕ Close
        </button>
      </div>

      {detail && (
        <>
          {/* Progress Bar */}
          <div style={{ marginBottom: '1.5rem' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: '0.5rem',
                fontSize: '0.85rem',
              }}
            >
              <span style={{ opacity: 0.7 }}>{detail.progress?.stage ?? 'initializing'}</span>
              <strong style={{ color: statusColor(detail.status) }}>
                {pct !== null ? `${pct.toFixed(1)}%` : detail.status}
              </strong>
            </div>
            <div
              style={{
                height: '10px',
                background: 'rgba(255,255,255,0.06)',
                borderRadius: '5px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: pct !== null ? `${pct}%` : detail.status === 'running' ? '100%' : '100%',
                  background: statusColor(detail.status),
                  borderRadius: '5px',
                  transition: 'width 0.5s ease',
                  animation:
                    detail.status === 'running' && pct === null
                      ? 'pulse-bar 2s ease-in-out infinite'
                      : undefined,
                }}
              />
            </div>
            {detail.progress?.status_message && (
              <p
                style={{
                  margin: '0.4rem 0 0',
                  fontSize: '0.8rem',
                  opacity: 0.6,
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {detail.progress.status_message}
              </p>
            )}
          </div>

          {/* Metric Sparklines */}
          {numericMetrics.length > 0 && (
            <div style={{ marginBottom: '1.5rem' }}>
              <h3
                style={{
                  fontSize: '0.85rem',
                  opacity: 0.6,
                  fontWeight: 600,
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                  marginBottom: '0.75rem',
                }}
              >
                Live Metrics
              </h3>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                  gap: '0.75rem',
                }}
              >
                {numericMetrics.slice(0, 6).map(([key, val]) => (
                  <div
                    key={key}
                    style={{
                      background: 'rgba(255,255,255,0.03)',
                      border: '1px solid var(--line)',
                      borderRadius: '8px',
                      padding: '0.75rem',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                        marginBottom: '0.5rem',
                      }}
                    >
                      <span
                        style={{
                          fontSize: '0.75rem',
                          opacity: 0.6,
                          textTransform: 'uppercase',
                          letterSpacing: '0.04em',
                        }}
                      >
                        {key.replace(/_/g, ' ')}
                      </span>
                      <strong style={{ fontSize: '1rem', color: statusColor(detail.status) }}>
                        {typeof val === 'number'
                          ? Math.abs(val) < 10
                            ? val.toFixed(4)
                            : val.toFixed(2)
                          : String(val)}
                      </strong>
                    </div>
                    <Sparkline
                      values={metricHistory[key] ?? [val as number]}
                      color={statusColor(detail.status)}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Decision */}
          {detail.decision && (
            <div
              style={{
                marginBottom: '1.5rem',
                background: 'rgba(15,122,97,0.08)',
                border: '1px solid rgba(15,122,97,0.3)',
                borderRadius: '8px',
                padding: '1rem',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  marginBottom: '0.25rem',
                }}
              >
                <span style={{ color: 'var(--accent)' }}>◆</span>
                <strong style={{ color: 'var(--accent)' }}>{detail.decision.action}</strong>
                <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>({detail.decision.rule})</span>
              </div>
              {detail.decision.explanation && (
                <p style={{ margin: 0, fontSize: '0.85rem', opacity: 0.8 }}>
                  {detail.decision.explanation}
                </p>
              )}
            </div>
          )}

          {/* Live Terminal */}
          <div style={{ marginBottom: '1.5rem' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                marginBottom: '0.75rem',
              }}
            >
              <h3
                style={{
                  fontSize: '0.85rem',
                  opacity: 0.6,
                  fontWeight: 600,
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                  margin: 0,
                }}
              >
                Live Output
              </h3>
              {isRunning && (
                <span
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    fontSize: '0.75rem',
                    color: 'var(--accent)',
                  }}
                >
                  <span className="monitor-status-dot status-running" /> streaming
                </span>
              )}
            </div>
            <LiveTerminal logText={logText} />
          </div>

          {/* Action Buttons */}
          {detail.available_actions && detail.available_actions.length > 0 && (
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              {detail.available_actions.slice(0, 4).map((act) => (
                <button
                  key={act.action}
                  className="secondary-button small"
                  disabled={busyAction === act.action}
                  onClick={() => void triggerAction(act.action)}
                  title={act.description}
                >
                  {busyAction === act.action ? '⟳ Working…' : act.label}
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── Main Monitoring Page ──────────────────────────────────────────────────────

export default function MonitoringPage() {
  const [instances, setInstances] = useState<InstanceSummary[]>([]);
  const [clusterNodes, setClusterNodes] = useState<ClusterNodeHardware[]>([]);
  const [summary, setSummary] = useState<OrchestrationSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'running' | 'completed' | 'failed'>('all');
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [instancesRes, summaryRes, clusterRes] = await Promise.allSettled([
      getInstances(),
      getOrchestrationSummary(),
      getClusterNodes(),
    ]);
    setInstances(instancesRes.status === 'fulfilled' ? instancesRes.value : []);
    setSummary(summaryRes.status === 'fulfilled' ? summaryRes.value : null);
    setClusterNodes(clusterRes.status === 'fulfilled' ? clusterRes.value : []);
    setLoading(false);
    setLastRefresh(new Date());
  }, []);

  useEffect(() => {
    void refresh();
    const interval = setInterval(() => void refresh(), 6000);
    return () => clearInterval(interval);
  }, [refresh]);

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
    .filter((i) => filter === 'all' || i.status === filter)
    .sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''));

  const running = instances.filter((i) => i.status === 'running').length;
  const completed = instances.filter((i) => i.status === 'completed').length;
  const failed = instances.filter((i) => i.status === 'failed').length;

  return (
    <div className="dashboard-content">
      {/* Header */}
      <div className="dash-page-header panel">
        <div className="dash-page-header-inner">
          <div>
            <span className="eyebrow">Lifecycle → Monitor</span>
            <h1 className="dash-page-title">Instance Monitor</h1>
            <p className="dash-page-desc">
              Real-time view of all training, evaluation, and inference instances. Click any
              instance to inspect logs, metrics, and progress.
            </p>
          </div>
          <div className="dash-header-actions">
            <span className="refresh-indicator">◉ {lastRefresh.toLocaleTimeString()}</span>
            <button type="button" className="secondary-button small" onClick={() => void refresh()}>
              ⟳ Refresh
            </button>
          </div>
        </div>

        {/* Summary chips */}
        <div className="monitor-summary-row">
          {[
            { label: 'Total', value: instances.length, cls: 'total' },
            { label: 'Running', value: running, cls: 'running' },
            { label: 'Completed', value: completed, cls: 'completed' },
            { label: 'Failed', value: failed, cls: 'failed' },
          ].map((c) => (
            <div key={c.label} className={`monitor-chip ${c.cls}`}>
              <span className="monitor-chip-value">{c.value}</span>
              <span className="monitor-chip-label">{c.label}</span>
            </div>
          ))}
          {summary?.task_status_counts?.running != null && (
            <div className="monitor-chip tasks">
              <span className="monitor-chip-value">{summary.task_status_counts.running}</span>
              <span className="monitor-chip-label">Active tasks</span>
            </div>
          )}
        </div>
      </div>

      {/* Cluster Health */}
      {clusterNodes.length > 0 && (
        <div className="panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
          <h2
            className="section-title"
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1rem',
            }}
          >
            <span>Cluster Node Health</span>
            <span
              style={{
                fontSize: '0.75rem',
                opacity: 0.5,
                border: '1px solid var(--line)',
                padding: '0.2rem 0.5rem',
                borderRadius: '4px',
              }}
            >
              Live
            </span>
          </h2>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '1rem',
            }}
          >
            {clusterNodes.map((node) => (
              <div
                key={node.id}
                style={{
                  border: '1px solid var(--line)',
                  borderRadius: '8px',
                  padding: '1rem',
                  background: 'rgba(255,255,255,0.02)',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginBottom: '0.4rem',
                  }}
                >
                  <strong style={{ fontSize: '0.9rem' }}>{node.name}</strong>
                  <span
                    style={{
                      fontSize: '0.75rem',
                      color: node.status === 'online' ? 'var(--accent)' : 'var(--foreground)',
                      opacity: node.status === 'online' ? 1 : 0.5,
                    }}
                  >
                    ● {node.status}
                  </span>
                </div>
                <div style={{ fontSize: '0.78rem', opacity: 0.6, marginBottom: '0.75rem' }}>
                  {node.type} · {node.memory}
                </div>
                <div className="monitor-progress-track">
                  <div
                    className="monitor-progress-fill"
                    style={{
                      width: `${node.usage}%`,
                      background: node.usage > 85 ? 'var(--danger)' : 'var(--accent)',
                    }}
                  />
                  <span className="monitor-progress-label">{node.usage}% VRAM</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Deep Instance Detail — shown when an instance is selected */}
      {selectedId && (
        <InstanceDetailPanel instanceId={selectedId} onClose={() => setSelectedId(null)} />
      )}

      {/* Filter Bar */}
      <div className="filter-bar panel">
        {(['all', 'running', 'completed', 'failed'] as const).map((f) => (
          <button
            key={f}
            type="button"
            className={`filter-tab ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all'
              ? `All (${instances.length})`
              : `${f} (${instances.filter((i) => i.status === f).length})`}
          </button>
        ))}
      </div>

      {/* Instance List */}
      {loading && !instances.length && (
        <div className="dash-loading panel">
          <span className="dash-loading-icon">⟳</span>
          <span>Scanning instances…</span>
        </div>
      )}

      {!loading && !filtered.length && (
        <div className="dash-empty panel">
          <p>No {filter !== 'all' ? filter : ''} instances found.</p>
          <Link href="/dashboard/training" className="primary-button small">
            ▲ Launch Training
          </Link>
        </div>
      )}

      <div className="monitor-instance-list">
        {filtered.map((instance) => {
          const pct = progressPct(instance);
          const topRec = instance.recommendations?.[0] ?? null;
          const isSelected = instance.id === selectedId;

          return (
            <div
              key={instance.id}
              className={`monitor-instance-card panel ${isSelected ? 'selected-instance' : ''}`}
              style={{ borderLeft: `3px solid ${statusColor(instance.status)}`, cursor: 'pointer' }}
              onClick={() => setSelectedId(isSelected ? null : instance.id)}
            >
              {/* Instance header */}
              <div className="monitor-instance-header">
                <div className="monitor-instance-id-group">
                  <span className={`monitor-status-dot status-${instance.status}`} />
                  <span className="monitor-instance-type">{instance.type}</span>
                  <span className="monitor-instance-name">{instance.name}</span>
                </div>
                <div className="monitor-instance-meta">
                  <span className="monitor-learning-mode">
                    {instance.lifecycle?.learning_mode ?? '—'}
                  </span>
                  <span className="monitor-env">{instance.environment.kind}</span>
                  <span className="monitor-updated">{formatTime(instance.updated_at)}</span>
                  <span style={{ fontSize: '0.75rem', opacity: 0.5, marginLeft: '0.5rem' }}>
                    {isSelected ? '▲ close' : '▼ inspect'}
                  </span>
                </div>
              </div>

              {/* Progress bar */}
              {pct !== null && (
                <div className="monitor-progress-track" style={{ marginTop: '0.75rem' }}>
                  <div
                    className="monitor-progress-fill"
                    style={{ width: `${pct}%`, background: statusColor(instance.status) }}
                  />
                  <span className="monitor-progress-label">{pct.toFixed(0)}%</span>
                </div>
              )}

              {/* Stage */}
              {instance.progress?.stage && (
                <div className="monitor-stage-row">
                  <span className="monitor-stage-label">{instance.progress.stage}</span>
                  {instance.progress.status_message && (
                    <span className="monitor-stage-msg">{instance.progress.status_message}</span>
                  )}
                </div>
              )}

              {/* Metrics inline */}
              {Object.keys(instance.metrics_summary).length > 0 && (
                <div className="monitor-metrics-row">
                  {Object.entries(instance.metrics_summary)
                    .filter(([, v]) => typeof v === 'number')
                    .slice(0, 4)
                    .map(([key, val]) => (
                      <div key={key} className="monitor-metric">
                        <span className="monitor-metric-label">{key}</span>
                        <span className="monitor-metric-value">
                          {typeof val === 'number' ? val.toFixed(3) : String(val)}
                        </span>
                      </div>
                    ))}
                </div>
              )}

              {/* Quick action */}
              {topRec && !isSelected && (
                <div className="monitor-actions" onClick={(e) => e.stopPropagation()}>
                  <button
                    type="button"
                    className="primary-button small"
                    disabled={busyAction === `${instance.id}:${topRec.action}`}
                    onClick={() => void triggerAction(instance, topRec.action)}
                  >
                    {busyAction === `${instance.id}:${topRec.action}` ? 'Working…' : topRec.action}
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
