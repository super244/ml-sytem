"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import {
  getInstanceDetail,
  runManagedInstanceAction,
  type InstanceActionDescriptor,
  type InstanceDetail,
} from "@/lib/api";
import { formatCount, formatFixed, formatLatency, formatPercent } from "@/lib/formatting";
import { ROUTES } from "@/lib/routes";

import { MetricsTrendChart } from "@/components/metrics-trend-chart";
import { MetricBadge } from "@/components/panels/metric-badge";
import { PageHeader } from "@/components/ui/page-header";
import { StatePanel } from "@/components/ui/state-panel";

function formatTimestamp(value?: string | null) {
  if (!value) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function lifecycleValue(value?: string | null) {
  if (!value) {
    return "n/a";
  }
  return value.replace(/_/g, " ");
}

export function InstanceDetailView({ instanceId }: { instanceId: string }) {
  const router = useRouter();
  const [detail, setDetail] = useState<InstanceDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [stream, setStream] = useState<"stdout" | "stderr">("stdout");
  const [busyAction, setBusyAction] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadDetail() {
      try {
        const payload = await getInstanceDetail(instanceId);
        if (!active) {
          return;
        }
        setDetail(payload);
        setError(null);
      } catch (nextError) {
        if (!active) {
          return;
        }
        setError(nextError instanceof Error ? nextError.message : "Failed to load instance detail.");
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadDetail();
    const intervalId = window.setInterval(() => {
      void loadDetail();
    }, 5000);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [instanceId]);

  const latestEvents = useMemo(
    () => (detail?.events ?? []).slice(-8).reverse(),
    [detail?.events],
  );

  async function triggerAction(action: InstanceActionDescriptor) {
    if (!detail) {
      return;
    }
    setBusyAction(action.label);
    setNotice(null);
    setError(null);
    try {
      const nextDetail = await runManagedInstanceAction(detail.id, {
        action: action.action,
        config_path: action.config_path ?? undefined,
        deployment_target: action.deployment_target ?? undefined,
        start: true,
      });
      setNotice(`Created ${nextDetail.type} instance ${nextDetail.name}.`);
      router.push(`/runs/${nextDetail.id}`);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to launch the action.");
    } finally {
      setBusyAction(null);
    }
  }

  if (loading && !detail) {
    return (
      <StatePanel
        eyebrow="Loading"
        title="Instance detail is loading."
        description="The control center is gathering lifecycle detail, logs, metrics, and follow-up actions."
        tone="loading"
      />
    );
  }

  if (error && !detail) {
    return (
      <StatePanel
        eyebrow="Unavailable"
        title="The instance detail could not be loaded."
        description={error}
        tone="error"
      />
    );
  }

  if (!detail) {
    return null;
  }

  const metrics = detail.metrics.summary ?? {};
  const logText = stream === "stdout" ? detail.logs?.stdout ?? "" : detail.logs?.stderr ?? "";

  return (
    <section className="route-stack">
      <PageHeader
        eyebrow="Instance Detail"
        title={detail.name}
        description="Inspect lifecycle intent, logs, metric streams, lineage, and managed follow-up actions from one place."
        metrics={[
          { label: "Type", value: detail.type },
          { label: "Status", value: detail.status, tone: "secondary" },
          {
            label: "Lifecycle",
            value: lifecycleValue(detail.lifecycle.stage ?? detail.progress?.stage ?? detail.type),
            tone: "accent",
          },
          { label: "Children", value: formatCount(detail.children.length) },
          { label: "Accuracy", value: formatPercent(Number(metrics.accuracy ?? NaN), 1) },
        ]}
        actions={
          <>
            <Link className="ghost-button small" href={ROUTES.runs}>
              Back to runs
            </Link>
            {detail.type === "inference" ? (
              <Link className="primary-button small" href={ROUTES.solve}>
                Open chat workspace
              </Link>
            ) : null}
          </>
        }
      />

      {notice ? (
        <StatePanel eyebrow="Action" title="Managed follow-up launched." description={notice} />
      ) : null}

      {error ? (
        <StatePanel eyebrow="Action Error" title="A control action failed." description={error} tone="error" />
      ) : null}

      <section className="detail-grid">
        <article className="panel detail-panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Lifecycle</div>
              <h2>Branch summary</h2>
            </div>
          </div>
          <div className="badge-row">
            <MetricBadge label="Stage" value={lifecycleValue(detail.lifecycle.stage)} />
            <MetricBadge label="Origin" value={lifecycleValue(detail.lifecycle.origin)} tone="secondary" />
            <MetricBadge label="Mode" value={lifecycleValue(detail.lifecycle.learning_mode)} tone="accent" />
            <MetricBadge label="Environment" value={detail.environment.kind} />
          </div>
          <div className="preview-block subtle">
            <strong>Source model</strong>
            <p>{detail.lifecycle.source_model ?? "Inherited from the config/profile."}</p>
          </div>
          {detail.lifecycle.architecture?.family ? (
            <div className="preview-block subtle">
              <strong>Architecture</strong>
              <p>
                {detail.lifecycle.architecture.family}
                {detail.lifecycle.architecture.hidden_size
                  ? ` • hidden ${detail.lifecycle.architecture.hidden_size}`
                  : ""}
                {detail.lifecycle.architecture.num_layers
                  ? ` • layers ${detail.lifecycle.architecture.num_layers}`
                  : ""}
                {detail.lifecycle.architecture.num_attention_heads
                  ? ` • heads ${detail.lifecycle.architecture.num_attention_heads}`
                  : ""}
              </p>
            </div>
          ) : null}
          <div className="preview-block subtle">
            <strong>Timing</strong>
            <p>Created {formatTimestamp(detail.created_at)}</p>
            <p>Updated {formatTimestamp(detail.updated_at)}</p>
            <p>
              Parent{" "}
              {detail.parent_instance_id ? (
                <Link href={`/runs/${detail.parent_instance_id}`}>{detail.parent_instance_id}</Link>
              ) : (
                "root"
              )}
            </p>
          </div>
        </article>

        <article className="panel detail-panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Control Actions</div>
              <h2>Managed next steps</h2>
            </div>
          </div>
          <div className="action-row">
            {detail.available_actions.map((action) => (
              <button
                key={`${action.action}-${action.label}-${action.deployment_target ?? "default"}`}
                className={action.action === "deploy" ? "secondary-button small" : "primary-button small"}
                type="button"
                disabled={busyAction === action.label}
                onClick={() => void triggerAction(action)}
              >
                {busyAction === action.label ? "Working..." : action.label}
              </button>
            ))}
          </div>
          <div className="recommendation-stack">
            {(detail.recommendations ?? []).map((item) => (
              <article key={`${item.action}-${item.reason}`} className="recommendation-card">
                <div className="message-meta">
                  <span>{item.action}</span>
                  <span className="status-pill">priority {item.priority}</span>
                </div>
                <p>{item.reason}</p>
              </article>
            ))}
          </div>
        </article>
      </section>

      <section className="detail-grid">
        <article className="panel detail-panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Metric Streams</div>
              <h2>Live learning and evaluation traces</h2>
            </div>
          </div>
          <div className="badge-row">
            <MetricBadge label="Accuracy" value={formatPercent(Number(metrics.accuracy ?? NaN), 1)} />
            <MetricBadge
              label="Parse rate"
              value={formatPercent(Number(metrics.parse_rate ?? NaN), 1)}
              tone="secondary"
            />
            <MetricBadge
              label="Latency"
              value={formatLatency(Number(metrics.avg_latency_s ?? NaN))}
              tone="accent"
            />
            <MetricBadge label="Latest step" value={formatFixed(Number(metrics.latest_step ?? NaN), 0)} />
          </div>
          <MetricsTrendChart points={detail.metrics.points ?? []} />
        </article>

        <article className="panel detail-panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Lineage</div>
              <h2>Child workflows and follow-ups</h2>
            </div>
          </div>
          {!detail.children.length ? (
            <p className="hero-copy">No child instances have been launched from this branch yet.</p>
          ) : (
            <div className="lineage-list">
              {detail.children.map((child) => (
                <Link key={child.id} className="lineage-card" href={`/runs/${child.id}`}>
                  <div className="message-meta">
                    <span>{child.type}</span>
                    <span className="status-pill">{child.status}</span>
                  </div>
                  <strong>{child.name}</strong>
                  <p>
                    {lifecycleValue(child.lifecycle.stage)} • {child.environment.kind} • updated{" "}
                    {formatTimestamp(child.updated_at)}
                  </p>
                </Link>
              ))}
            </div>
          )}
        </article>
      </section>

      <section className="detail-grid">
        <article className="panel detail-panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Logs</div>
              <h2>Runtime output</h2>
            </div>
            <div className="action-row">
              <button
                className={`secondary-button small${stream === "stdout" ? " active-button" : ""}`}
                type="button"
                onClick={() => setStream("stdout")}
              >
                Stdout
              </button>
              <button
                className={`secondary-button small${stream === "stderr" ? " active-button" : ""}`}
                type="button"
                onClick={() => setStream("stderr")}
              >
                Stderr
              </button>
            </div>
          </div>
          <pre className="log-shell">
            <code>{logText || "No log output yet."}</code>
          </pre>
        </article>

        <article className="panel detail-panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Events</div>
              <h2>Control-plane event tail</h2>
            </div>
          </div>
          <div className="event-list">
            {latestEvents.length ? (
              latestEvents.map((event, index) => (
                <article key={`${String(event.id ?? "event")}-${index}`} className="event-card">
                  <div className="message-meta">
                    <span>{String(event.type ?? event.event_type ?? "event")}</span>
                    <span className="status-pill">{formatTimestamp(String(event.created_at ?? ""))}</span>
                  </div>
                  <p>{String(event.message ?? "No message available.")}</p>
                </article>
              ))
            ) : (
              <p className="hero-copy">No control-plane events recorded yet.</p>
            )}
          </div>
        </article>
      </section>
    </section>
  );
}
