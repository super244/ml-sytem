"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getMissionControl, type MissionControlSnapshot } from "@/lib/api";
import { formatCount } from "@/lib/formatting";
import { ROUTES } from "@/lib/routes";

type ClusterState = {
  mission: MissionControlSnapshot | null;
  loading: boolean;
  error: string | null;
};

function nodeTone(status: string) {
  if (status === "online") {
    return "var(--accent)";
  }
  if (status === "idle") {
    return "var(--secondary)";
  }
  return "var(--danger)";
}

export default function ClusterPage() {
  const [state, setState] = useState<ClusterState>({
    mission: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const mission = await getMissionControl();
        if (!active) {
          return;
        }
        setState({
          mission,
          loading: false,
          error: null,
        });
      } catch (error) {
        if (!active) {
          return;
        }
        setState({
          mission: null,
          loading: false,
          error: error instanceof Error ? error.message : "Cluster state could not be loaded.",
        });
      }
    }

    void load();
    const interval = setInterval(() => void load(), 10_000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const mission = state.mission;
  const nodes = mission?.watchlist.cluster_nodes ?? [];
  const runningInstances = mission?.watchlist.running_instances ?? [];

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div className="dash-page-header-inner">
          <div>
            <span className="eyebrow">V2 Lab → Cluster</span>
            <h1 className="dash-page-title">Cluster Orchestration</h1>
            <p className="dash-page-desc">
              Track node availability, see where active work is running, and move quickly into training or monitoring when capacity opens up.
            </p>
          </div>
          <div className="dash-header-actions">
            <Link href={ROUTES.training} className="primary-button small">
              Launch Training
            </Link>
            <Link href={ROUTES.monitoring} className="secondary-button small">
              Open Monitor
            </Link>
          </div>
        </div>
      </div>

      {state.error ? <div className="dash-error-banner panel">⚠ {state.error}</div> : null}

      <div className="workspace-summary-grid">
        {[
          { label: "Nodes", value: mission?.summary.cluster_nodes },
          { label: "Running Jobs", value: mission?.summary.running_instances },
          { label: "Open Circuits", value: mission?.summary.open_circuits },
          { label: "Telemetry Backlog", value: mission?.summary.telemetry_backlog },
          { label: "Agents", value: mission?.summary.active_agents },
          { label: "Sweeps", value: mission?.summary.running_sweeps },
        ].map((item) => (
          <div key={item.label} className="workspace-summary-card panel">
            <span className="workspace-summary-value">{formatCount(item.value)}</span>
            <span className="workspace-summary-label">{item.label}</span>
          </div>
        ))}
      </div>

      <div className="workspace-section-grid">
        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Node Health</h2>
            <p className="control-label">
              Local and remote capacity visible from the same surface.
            </p>
          </div>

          {state.loading && !nodes.length ? (
            <div className="dash-loading"><span>⟳</span> Loading cluster nodes…</div>
          ) : nodes.length === 0 ? (
            <div className="dash-empty" style={{ padding: "2rem 1rem" }}>
              <p>No cluster nodes reported yet.</p>
            </div>
          ) : (
            <div className="resource-list">
              {nodes.map((node) => (
                <article key={node.id} className="resource-card">
                  <div className="model-chip-header">
                    <strong>{node.name}</strong>
                    <span style={{ color: nodeTone(node.status), fontWeight: 700 }}>
                      {node.status}
                    </span>
                  </div>
                  <p>{node.type} · {node.memory}</p>
                  <div className="monitor-progress-track" style={{ marginTop: "0.65rem" }}>
                    <div
                      className="monitor-progress-fill"
                      data-status={node.status === "offline" ? "failed" : node.status === "idle" ? "completed" : "running"}
                      style={{ width: `${Math.max(0, Math.min(100, node.usage))}%` }}
                    />
                    <span className="monitor-progress-label">{node.usage}% load</span>
                  </div>
                  <div className="badge-row">
                    <span className="status-pill">{node.activeJobs} active jobs</span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Running Workloads</h2>
            <p className="control-label">
              Active branches currently occupying cluster capacity.
            </p>
          </div>

          {runningInstances.length === 0 ? (
            <div className="dash-empty" style={{ padding: "2rem 1rem" }}>
              <p>No running workloads right now.</p>
            </div>
          ) : (
            <div className="resource-list">
              {runningInstances.map((instance) => (
                <Link key={instance.id} href={`/runs/${instance.id}`} className="resource-card">
                  <div className="model-chip-header">
                    <strong>{instance.name}</strong>
                    <span>{instance.type}</span>
                  </div>
                  <p>
                    {instance.environment.kind} · {instance.lifecycle.learning_mode ?? "managed"} · {instance.progress?.stage ?? instance.status}
                  </p>
                  {typeof instance.progress?.percent === "number" ? (
                    <div className="monitor-progress-track" style={{ marginTop: "0.65rem" }}>
                      <div
                        className="monitor-progress-fill"
                        data-status="running"
                        style={{ width: `${Math.max(0, Math.min(100, instance.progress.percent * 100))}%` }}
                      />
                      <span className="monitor-progress-label">
                        {(instance.progress.percent * 100).toFixed(0)}%
                      </span>
                    </div>
                  ) : null}
                </Link>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
