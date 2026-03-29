"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  getInstances,
  getOrchestrationSummary,
  getWorkspaceOverview,
  type InstanceSummary,
  type OrchestrationSummary,
  type WorkspaceOverview,
} from "@/lib/api";
import { ROUTES } from "@/lib/routes";
import { formatCount } from "@/lib/formatting";

type DashboardState = {
  instances: InstanceSummary[];
  summary: OrchestrationSummary | null;
  workspace: WorkspaceOverview | null;
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

export default function DashboardPage() {
  const [state, setState] = useState<DashboardState>({
    instances: [],
    summary: null,
    workspace: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let active = true;
    async function load() {
      const [instancesRes, summaryRes, workspaceRes] = await Promise.allSettled([
        getInstances(),
        getOrchestrationSummary(),
        getWorkspaceOverview(),
      ]);
      if (!active) return;
      setState({
        instances: instancesRes.status === "fulfilled" ? instancesRes.value : [],
        summary: summaryRes.status === "fulfilled" ? summaryRes.value : null,
        workspace: workspaceRes.status === "fulfilled" ? workspaceRes.value : null,
        loading: false,
        error: null,
      });
    }
    void load();
    const interval = setInterval(() => void load(), 8000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const running = state.instances.filter((i) => i.status === "running").length;
  const completed = state.instances.filter((i) => i.status === "completed").length;
  const failed = state.instances.filter((i) => i.status === "failed").length;
  const recent = [...state.instances]
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    .slice(0, 6);

  return (
    <div className="dashboard-content">
      {/* Hero */}
      <div className="dash-hero panel">
        <div className="dash-hero-inner">
          <div className="dash-hero-copy">
            <span className="eyebrow">AI-Factory v1</span>
            <h1 className="dash-hero-title">
              Your Personal<br />AI Laboratory
            </h1>
            <p className="dash-hero-desc">
              Full-cycle LLM platform — train, evaluate, finetune, and deploy models
              from one unified system. End-to-end AI lifecycle management.
            </p>
            <div className="dash-hero-actions">
              <Link className="primary-button" href={ROUTES.training}>
                Launch Training ▲
              </Link>
              <Link className="secondary-button" href={ROUTES.monitoring}>
                Monitor Instances
              </Link>
            </div>
          </div>
          <div className="dash-hero-stats">
            <div className="dash-stat-card">
              <span className="dash-stat-value">{formatCount(state.instances.length)}</span>
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

      {/* Lifecycle Stage Grid */}
      <div className="dash-section-heading">
        <h2 className="dash-section-title">AI Lifecycle</h2>
        <p className="dash-section-desc">
          Navigate the complete model lifecycle from training through production.
        </p>
      </div>

      <div className="lifecycle-grid">
        {LIFECYCLE_STAGES.map((stage) => {
          const stageInstances = stage.instanceTypes.length
            ? state.instances.filter((i) => (stage.instanceTypes as string[]).includes(i.type))
            : state.instances;
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
      {state.workspace && (
        <>
          <div className="dash-section-heading">
            <h2 className="dash-section-title">Workspace</h2>
          </div>
          <div className="workspace-summary-grid">
            {[
              { label: "Models", value: state.workspace.summary.models },
              { label: "Datasets", value: state.workspace.summary.datasets },
              { label: "Training Profiles", value: state.workspace.summary.training_profiles },
              { label: "Eval Configs", value: state.workspace.summary.evaluation_configs },
              { label: "Benchmarks", value: state.workspace.summary.benchmarks },
              { label: "Runs", value: state.workspace.summary.runs },
            ].map((item) => (
              <div key={item.label} className="workspace-summary-card panel">
                <span className="workspace-summary-value">{formatCount(item.value) ?? "0"}</span>
                <span className="workspace-summary-label">{item.label}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {state.loading && !state.instances.length && (
        <div className="dash-loading panel">
          <span className="dash-loading-icon">⟳</span>
          <span>Loading AI-Factory dashboard…</span>
        </div>
      )}
    </div>
  );
}
