"use client";

import type { Route } from "next";
import Link from "next/link";
import { useEffect, useState } from "react";

import {
  getMissionControl,
  type MissionControlRecommendation,
  type MissionControlSnapshot,
} from "@/lib/api";
import { ROUTES } from "@/lib/routes";
import { formatCount } from "@/lib/formatting";

type DashboardState = {
  mission: MissionControlSnapshot | null;
  loading: boolean;
  error: string | null;
};

function statusColor(status: string) {
  if (status === "running") return "var(--accent)";
  if (status === "completed") return "var(--secondary)";
  if (status === "failed") return "var(--danger)";
  return "var(--muted)";
}

function InstanceStatusDot({ status }: { status: string }) {
  return (
    <span
      className="status-dot"
      style={{ background: statusColor(status) }}
      title={status}
    />
  );
}

function severityTone(severity: MissionControlRecommendation["severity"]) {
  if (severity === "critical") {
    return {
      label: "Critical",
      color: "var(--danger)",
      bg: "rgba(191, 90, 64, 0.08)",
      border: "rgba(191, 90, 64, 0.18)",
    };
  }
  if (severity === "warning") {
    return {
      label: "Warning",
      color: "#c47b20",
      bg: "rgba(196, 123, 32, 0.08)",
      border: "rgba(196, 123, 32, 0.18)",
    };
  }
  if (severity === "opportunity") {
    return {
      label: "Opportunity",
      color: "var(--accent)",
      bg: "rgba(15, 122, 97, 0.08)",
      border: "rgba(15, 122, 97, 0.18)",
    };
  }
  return {
    label: "Info",
    color: "var(--secondary)",
    bg: "rgba(37, 95, 155, 0.08)",
    border: "rgba(37, 95, 155, 0.18)",
  };
}

const LIFECYCLE_STAGES = [
  {
    stage: "train",
    label: "Training",
    icon: "▲",
    href: ROUTES.training,
    description: "Launch and manage training runs",
    color: "var(--accent)",
    bg: "rgba(15, 122, 97, 0.08)",
    border: "rgba(15, 122, 97, 0.18)",
    instanceTypes: ["train"],
  },
  {
    stage: "monitor",
    label: "Monitoring",
    icon: "◉",
    href: ROUTES.monitoring,
    description: "Real-time instance progress",
    color: "var(--secondary)",
    bg: "rgba(37, 95, 155, 0.08)",
    border: "rgba(37, 95, 155, 0.18)",
    instanceTypes: [],
  },
  {
    stage: "evaluate",
    label: "Evaluation",
    icon: "◆",
    href: ROUTES.evaluate,
    description: "Benchmark results and comparison",
    color: "#8857c4",
    bg: "rgba(136, 87, 196, 0.07)",
    border: "rgba(136, 87, 196, 0.16)",
    instanceTypes: ["evaluate"],
  },
  {
    stage: "finetune",
    label: "Finetuning",
    icon: "⟳",
    href: ROUTES.finetune,
    description: "LoRA, QLoRA and full finetune",
    color: "#c47b20",
    bg: "rgba(196, 123, 32, 0.07)",
    border: "rgba(196, 123, 32, 0.16)",
    instanceTypes: ["finetune"],
  },
  {
    stage: "deploy",
    label: "Deployment",
    icon: "⬆",
    href: ROUTES.deploy,
    description: "HuggingFace, Ollama, LM Studio",
    color: "var(--danger)",
    bg: "rgba(191, 90, 64, 0.07)",
    border: "rgba(191, 90, 64, 0.16)",
    instanceTypes: ["deploy"],
  },
  {
    stage: "inference",
    label: "Inference",
    icon: "◎",
    href: ROUTES.inference,
    description: "Chat with your deployed models",
    color: "#2ea8a8",
    bg: "rgba(46, 168, 168, 0.07)",
    border: "rgba(46, 168, 168, 0.16)",
    instanceTypes: ["inference"],
  },
];

const LAB_SURFACES = [
  {
    stage: "datasets",
    label: "Datasets",
    icon: "▤",
    href: ROUTES.dashboard_datasets,
    description: "Curate telemetry and dispatch synthesis jobs",
    color: "var(--accent)",
    bg: "rgba(15, 122, 97, 0.08)",
    border: "rgba(15, 122, 97, 0.18)",
  },
  {
    stage: "agents",
    label: "Agents",
    icon: "⍾",
    href: ROUTES.dashboard_agents,
    description: "Adjust swarm roles and watch orchestration logs",
    color: "var(--secondary)",
    bg: "rgba(37, 95, 155, 0.08)",
    border: "rgba(37, 95, 155, 0.18)",
  },
  {
    stage: "automl",
    label: "AutoML",
    icon: "⎈",
    href: ROUTES.dashboard_automl,
    description: "Launch sweeps and inspect the best trial fast",
    color: "#a26e1f",
    bg: "rgba(162, 110, 31, 0.08)",
    border: "rgba(162, 110, 31, 0.18)",
  },
  {
    stage: "cluster",
    label: "Cluster",
    icon: "▦",
    href: ROUTES.dashboard_cluster,
    description: "Track node health, load, and distributed capacity",
    color: "#6a589b",
    bg: "rgba(106, 88, 155, 0.08)",
    border: "rgba(106, 88, 155, 0.18)",
  },
];

export default function DashboardPage() {
  const [state, setState] = useState<DashboardState>({
    mission: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const mission = await getMissionControl();
        if (!active) return;
        setState({
          mission,
          loading: false,
          error: null,
        });
      } catch (error) {
        if (!active) return;
        setState({
          mission: null,
          loading: false,
          error: error instanceof Error ? error.message : "Mission control could not be loaded.",
        });
      }
    }
    void load();
    const interval = setInterval(() => void load(), 8000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const mission = state.mission;
  const instances = mission?.watchlist.instances ?? [];
  const recent = instances.slice(0, 6);
  const running = mission?.summary.running_instances ?? 0;
  const failed = mission?.summary.failed_instances ?? 0;
  const totalInstances = mission?.summary.instances ?? 0;
  const completed = Math.max(totalInstances - running - failed, 0);
  const openCircuits = mission?.summary.open_circuits ?? 0;
  const priorityQueue = mission?.recommendations ?? [];
  const criticality = mission?.criticality.level ?? "info";
  const labSurfaceCounts = {
    datasets: mission?.summary.telemetry_backlog ?? 0,
    agents: mission?.summary.active_agents ?? 0,
    automl: mission?.summary.running_sweeps ?? 0,
    cluster: mission?.summary.cluster_nodes ?? 0,
  } as const;

  return (
    <div className="dashboard-content">
      {/* Hero */}
      <div className="dash-hero panel">
        <div className="dash-hero-inner">
          <div className="dash-hero-copy">
            <span className="eyebrow">V2 Autonomous Loop</span>
            <h1 className="dash-hero-title">Operate the lab, not just the run.</h1>
            <p className="dash-hero-desc">
              Launch training, promote failures into new data, configure agents, and keep the full lifecycle moving from one control surface.
            </p>
            <div className="dash-hero-actions">
              <Link className="primary-button" href={ROUTES.training}>
                Launch Training ▲
              </Link>
              <Link className="secondary-button" href={ROUTES.dashboard_datasets}>
                Open Lab Loop
              </Link>
            </div>
            <div className="dash-command-strip">
              <span className="dash-command-pill">
                Ready {mission?.summary.ready_checks ?? 0}/{mission?.summary.total_checks ?? 0}
              </span>
              <span className="dash-command-pill">
                Agents {formatCount(mission?.summary.active_agents ?? 0)}
              </span>
              <span className="dash-command-pill">
                Sweeps {formatCount(mission?.summary.running_sweeps ?? 0)}
              </span>
              <span className="dash-command-pill">
                Telemetry {formatCount(mission?.summary.telemetry_backlog ?? 0)}
              </span>
              <span className="dash-command-pill">
                Circuits {formatCount(openCircuits)}
              </span>
              <span className="dash-command-pill">
                Criticality {criticality}
              </span>
            </div>
          </div>
          <div className="dash-hero-stats">
            <div className="dash-stat-card">
              <span className="dash-stat-value">{formatCount(totalInstances)}</span>
              <span className="dash-stat-label">Total instances</span>
            </div>
            <div className="dash-stat-card accent">
              <span className="dash-stat-value">{formatCount(running)}</span>
              <span className="dash-stat-label">Running now</span>
            </div>
            <div className="dash-stat-card secondary">
              <span className="dash-stat-value">{formatCount(completed)}</span>
              <span className="dash-stat-label">Completed</span>
            </div>
            <div className="dash-stat-card danger">
              <span className="dash-stat-value">{formatCount(failed)}</span>
              <span className="dash-stat-label">Failed</span>
            </div>
          </div>
        </div>
      </div>

      {state.error && (
        <div className="dash-error-banner panel">
          <span>⚠</span> {state.error}
        </div>
      )}

      <div className="dash-section-heading">
        <h2 className="dash-section-title">Priority Queue</h2>
        <p className="dash-section-desc">
          The next best actions based on readiness, failures, telemetry backlog, and idle capacity.
        </p>
      </div>

      {priorityQueue.length > 0 ? (
        <div className="dash-priority-grid">
          {priorityQueue.map((item) => {
            const tone = severityTone(item.severity);
            return (
              <article
                key={item.id}
                className="recommendation-card dash-priority-card"
                style={{ background: tone.bg, borderColor: tone.border }}
              >
                <div className="dash-priority-header">
                  <span className="status-pill" style={{ color: tone.color, borderColor: tone.border }}>
                    {tone.label}
                  </span>
                  <span className="dash-priority-surface">{item.surface}</span>
                </div>
                <h3 className="dash-priority-title" style={{ color: tone.color }}>
                  {item.title}
                </h3>
                <p>{item.detail}</p>
                <div className="dash-priority-footer">
                  <div className="dash-priority-meta">
                    {item.metric_label && item.metric_value ? (
                      <span className="dash-priority-metric">
                        {item.metric_label}: {item.metric_value}
                      </span>
                    ) : null}
                    {item.command ? (
                      <code className="dash-priority-command">{item.command}</code>
                    ) : null}
                  </div>
                  <Link className="ghost-button small" href={item.href as Route}>
                    Open →
                  </Link>
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="panel state-panel">
          <div className="eyebrow">Priority Queue</div>
          <h2 className="state-title">No urgent blockers detected.</h2>
          <p className="hero-copy">
            Mission control did not detect any immediate actions. The workspace and lab surfaces are in a steady state.
          </p>
        </div>
      )}

      {/* Lifecycle Stage Grid */}
      <div className="dash-section-heading">
        <h2 className="dash-section-title">AI Lifecycle</h2>
        <p className="dash-section-desc">
          Move cleanly from training to deployment without losing operational context.
        </p>
      </div>

      <div className="lifecycle-grid">
        {LIFECYCLE_STAGES.map((stage) => {
          const stageInstances = stage.instanceTypes.length
            ? instances.filter((i) => (stage.instanceTypes as string[]).includes(i.type))
            : instances;
          const stageRunning = stageInstances.filter((i) => i.status === "running").length;

          return (
            <Link key={stage.stage} href={stage.href} className="lifecycle-stage-card">
              <div
                className="lifecycle-stage-icon"
                style={{ background: stage.bg, border: `1px solid ${stage.border}`, color: stage.color }}
              >
                {stage.icon}
              </div>
              <div className="lifecycle-stage-body">
                <h3 className="lifecycle-stage-title" style={{ color: stage.color }}>
                  {stage.label}
                </h3>
                <p className="lifecycle-stage-desc">{stage.description}</p>
                {stageRunning > 0 && (
                  <span className="lifecycle-stage-badge" style={{ background: stage.bg, color: stage.color, border: `1px solid ${stage.border}` }}>
                    {stageRunning} running
                  </span>
                )}
              </div>
              <span className="lifecycle-stage-arrow">→</span>
            </Link>
          );
        })}
      </div>

      <div className="dash-section-heading">
        <h2 className="dash-section-title">Autonomous Lab</h2>
        <p className="dash-section-desc">
          Dataset curation, swarm control, sweep orchestration, and cluster visibility.
        </p>
      </div>

      <div className="lifecycle-grid">
        {LAB_SURFACES.map((stage) => (
          <Link key={stage.stage} href={stage.href} className="lifecycle-stage-card">
            <div
              className="lifecycle-stage-icon"
              style={{ background: stage.bg, border: `1px solid ${stage.border}`, color: stage.color }}
            >
              {stage.icon}
            </div>
            <div className="lifecycle-stage-body">
              <h3 className="lifecycle-stage-title" style={{ color: stage.color }}>
                {stage.label}
              </h3>
              <p className="lifecycle-stage-desc">{stage.description}</p>
              <span
                className="lifecycle-stage-badge"
                style={{ background: stage.bg, color: stage.color, border: `1px solid ${stage.border}` }}
              >
                {formatCount(labSurfaceCounts[stage.stage as keyof typeof labSurfaceCounts])} live
              </span>
            </div>
            <span className="lifecycle-stage-arrow">→</span>
          </Link>
        ))}
      </div>

      {/* Recent Instances */}
      {recent.length > 0 && (
        <>
          <div className="dash-section-heading">
            <h2 className="dash-section-title">Recent Activity</h2>
            <Link href={ROUTES.monitoring} className="ghost-button small">
              View all →
            </Link>
          </div>
          <div className="recent-instances-grid">
            {recent.map((instance) => (
              <Link
                key={instance.id}
                href={`/runs/${instance.id}`}
                className="recent-instance-card panel"
              >
                <div className="recent-instance-header">
                  <InstanceStatusDot status={instance.status} />
                  <span className="recent-instance-type">{instance.type}</span>
                  <span className="recent-instance-status">{instance.status}</span>
                </div>
                <div className="recent-instance-name">{instance.name}</div>
                {typeof instance.progress?.percent === "number" && (
                  <div className="recent-progress-track">
                    <div
                      className="recent-progress-fill"
                      style={{
                        width: `${Math.max(0, Math.min(100, instance.progress.percent * 100))}%`,
                        background: statusColor(instance.status),
                      }}
                    />
                  </div>
                )}
                <div className="recent-instance-meta">
                  {instance.lifecycle.learning_mode && (
                    <span className="recent-meta-chip">{instance.lifecycle.learning_mode}</span>
                  )}
                  <span className="recent-meta-chip">{instance.environment.kind}</span>
                </div>
              </Link>
            ))}
          </div>
        </>
      )}

      {/* Workspace Summary */}
      {mission?.workspace && (
        <>
          <div className="dash-section-heading">
            <h2 className="dash-section-title">Workspace</h2>
          </div>
          <div className="workspace-summary-grid">
            {[
              { label: "Models", value: mission.workspace.summary.models },
              { label: "Datasets", value: mission.workspace.summary.datasets },
              { label: "Training Profiles", value: mission.workspace.summary.training_profiles },
              { label: "Eval Configs", value: mission.workspace.summary.evaluation_configs },
              { label: "Benchmarks", value: mission.workspace.summary.benchmarks },
              { label: "Runs", value: mission.workspace.summary.runs },
            ].map((item) => (
              <div key={item.label} className="workspace-summary-card panel">
                <span className="workspace-summary-value">{formatCount(item.value) ?? "0"}</span>
                <span className="workspace-summary-label">{item.label}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {state.loading && !mission && (
        <div className="dash-loading panel">
          <span className="dash-loading-icon">⟳</span>
          <span>Loading AI-Factory dashboard…</span>
        </div>
      )}
    </div>
  );
}
