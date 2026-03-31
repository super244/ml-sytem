"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState, useTransition } from "react";

import {
  createManagedInstance,
  getMissionControl,
  runManagedInstanceAction,
  type FeedbackRecommendation,
  type InstanceSummary,
  type MissionControlSnapshot,
} from "@/lib/api";
import { formatCount } from "@/lib/formatting";
import { ROUTES } from "@/lib/routes";

type DashboardState = {
  mission: MissionControlSnapshot | null;
  loading: boolean;
  error: string | null;
};

const DEFAULT_SOURCE_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct";

function topRecommendation(recommendations?: FeedbackRecommendation[]): FeedbackRecommendation | null {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }
  return recommendations.reduce((best, current) => (current.priority > best.priority ? current : best));
}

function formatTimestamp(value?: string | null): string {
  if (!value) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function progressLabel(instance: InstanceSummary): string {
  if (typeof instance.progress?.percent === "number") {
    return `${(instance.progress.percent * 100).toFixed(0)}%`;
  }
  return instance.progress?.stage ?? instance.status;
}

function latestLifecycleSource(instances: InstanceSummary[]): InstanceSummary | null {
  const completed = instances
    .filter(
      (instance) =>
        instance.status === "completed" &&
        ["train", "finetune", "evaluate", "deploy"].includes(instance.type),
    )
    .sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
  return completed[0] ?? null;
}

export default function Dashboard() {
  const router = useRouter();
  const [isNavigating, startTransition] = useTransition();
  const [state, setState] = useState<DashboardState>({
    mission: null,
    loading: true,
    error: null,
  });
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadMission() {
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
          error: error instanceof Error ? error.message : "Mission control is unavailable.",
        });
      }
    }

    void loadMission();
    const interval = setInterval(() => void loadMission(), 7_500);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const mission = state.mission;
  const instances = mission?.control_plane.instances ?? [];
  const runningInstances = mission?.watchlist.running_instances ?? [];
  const latestSource = useMemo(() => latestLifecycleSource(instances), [instances]);

  async function launchTrainingBranch(): Promise<void> {
    setBusyAction("launch:train");
    setNotice(null);
    try {
      const detail = await createManagedInstance({
        config_path: "configs/train.yaml",
        start: true,
        user_level: "hobbyist",
        lifecycle: {
          stage: "train",
          origin: "existing_model",
          learning_mode: "qlora",
          source_model: latestSource?.lifecycle.source_model ?? DEFAULT_SOURCE_MODEL,
          deployment_targets: ["ollama"],
        },
        metadata: {
          source: "dashboard_overview",
          action: "launch_train",
        },
      });
      setNotice(`Training branch ${detail.name} launched.`);
      startTransition(() => router.push(`/runs/${detail.id}`));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Training launch failed.");
    } finally {
      setBusyAction(null);
    }
  }

  async function launchFinetuneBranch(source: InstanceSummary): Promise<void> {
    setBusyAction(`launch:finetune:${source.id}`);
    setNotice(null);
    try {
      const detail = await createManagedInstance({
        config_path: "configs/finetune.yaml",
        start: true,
        parent_instance_id: source.id,
        user_level: "hobbyist",
        lifecycle: {
          stage: "finetune",
          origin: "existing_model",
          learning_mode: "qlora",
          source_model: source.lifecycle.source_model ?? DEFAULT_SOURCE_MODEL,
          deployment_targets: ["ollama"],
        },
        metadata: {
          source: "dashboard_overview",
          action: "launch_finetune",
        },
      });
      setNotice(`Finetune branch ${detail.name} launched from ${source.name}.`);
      startTransition(() => router.push(`/runs/${detail.id}`));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Finetune launch failed.");
    } finally {
      setBusyAction(null);
    }
  }

  async function triggerAction(
    instance: InstanceSummary,
    action: "evaluate" | "open_inference" | "deploy",
    options: {
      configPath: string;
      deploymentTarget?: "ollama";
    },
  ): Promise<void> {
    const actionKey = `${instance.id}:${action}`;
    setBusyAction(actionKey);
    setNotice(null);
    try {
      const detail = await runManagedInstanceAction(instance.id, {
        action,
        config_path: options.configPath,
        deployment_target: options.deploymentTarget,
        start: true,
      });
      setNotice(`${action} launched from ${instance.name}.`);
      startTransition(() => router.push(`/runs/${detail.id}`));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : `${action} failed.`);
    } finally {
      setBusyAction(null);
    }
  }

  async function dispatchRecommendation(instance: InstanceSummary, recommendation: FeedbackRecommendation): Promise<void> {
    const actionKey = `${instance.id}:${recommendation.action}`;
    setBusyAction(actionKey);
    setNotice(null);
    try {
      const detail = await runManagedInstanceAction(instance.id, {
        action: recommendation.action,
        config_path: recommendation.config_path ?? undefined,
        deployment_target: recommendation.deployment_target ?? undefined,
        start: true,
      });
      setNotice(`${recommendation.action} launched from ${instance.name}.`);
      startTransition(() => router.push(`/runs/${detail.id}`));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Recommended action failed.");
    } finally {
      setBusyAction(null);
    }
  }

  if (state.loading && !mission) {
    return (
      <div className="dashboard-content">
        <div className="dash-loading panel">
          <span>⟳</span>
          <span>Loading mission control…</span>
        </div>
      </div>
    );
  }

  if (state.error && !mission) {
    return (
      <div className="dashboard-content">
        <div className="dash-error-banner panel">
          <span>⚠</span>
          <span>{state.error}</span>
        </div>
      </div>
    );
  }

  if (!mission) {
    return null;
  }

  const latestRecommendationSource = runningInstances[0] ?? latestSource;

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div className="dash-page-header-inner">
          <div>
            <span className="eyebrow">Lifecycle + Lab</span>
            <h1 className="dash-page-title">Mission Control</h1>
            <p className="dash-page-desc">
              Launch managed training branches, promote completed runs into finetune or deploy flows,
              and monitor the autonomous lab from a single responsive control surface.
            </p>
          </div>
          <div className="dash-header-actions">
            <Link href={ROUTES.runs} className="secondary-button small">
              Open Runs
            </Link>
            <Link href={ROUTES.monitoring} className="primary-button small">
              Live Monitor
            </Link>
          </div>
        </div>
      </div>

      {state.error ? <div className="dash-error-banner panel">⚠ {state.error}</div> : null}
      {notice ? <div className="dash-note-banner panel">◎ {notice}</div> : null}

      <div className="workspace-summary-grid">
        {[
          { label: "Ready Checks", value: `${mission.summary.ready_checks}/${mission.summary.total_checks}` },
          { label: "Managed Instances", value: formatCount(mission.summary.instances) },
          { label: "Running", value: formatCount(mission.summary.running_instances) },
          { label: "AutoML Sweeps", value: formatCount(mission.summary.running_sweeps) },
          { label: "Telemetry Backlog", value: formatCount(mission.summary.telemetry_backlog) },
          { label: "Titan Mode", value: mission.titan.mode },
        ].map((item) => (
          <div key={item.label} className="workspace-summary-card panel">
            <span className="workspace-summary-value">{item.value}</span>
            <span className="workspace-summary-label">{item.label}</span>
          </div>
        ))}
      </div>

      <div className="workspace-section-grid">
        <section className="panel aside-section">
          <div className="model-chip-header">
            <div>
              <h2 className="section-title">Lifecycle Controls</h2>
              <p className="control-label">
                Each action posts into the managed instance API and routes execution through the same
                orchestration layer used by the CLI and TUI.
              </p>
            </div>
            <span className="status-pill">
              {mission.titan.backend} · {mission.titan.remote_execution ? "cloud-aware" : "local-first"}
            </span>
          </div>

          <div className="resource-list">
            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Start training</strong>
                <span>{mission.titan.mode}</span>
              </div>
              <p>Launch `configs/train.yaml` with a managed QLoRA-first training branch.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="primary-button small"
                  disabled={busyAction === "launch:train" || isNavigating}
                  onClick={() => void launchTrainingBranch()}
                >
                  {busyAction === "launch:train" ? "Working..." : "Launch training"}
                </button>
              </div>
            </article>

            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Start finetune</strong>
                <span>{latestSource ? latestSource.name : "needs source"}</span>
              </div>
              <p>Spawn `configs/finetune.yaml` from the latest completed managed branch.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="primary-button small"
                  disabled={!latestSource || busyAction === `launch:finetune:${latestSource?.id}` || isNavigating}
                  onClick={() => latestSource && void launchFinetuneBranch(latestSource)}
                >
                  {busyAction === `launch:finetune:${latestSource?.id}` ? "Working..." : "Launch finetune"}
                </button>
              </div>
            </article>

            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Evaluate latest</strong>
                <span>{latestSource ? latestSource.type : "needs source"}</span>
              </div>
              <p>Queue `configs/eval.yaml` against the current best completed lifecycle source.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="secondary-button small"
                  disabled={!latestSource || busyAction === `${latestSource?.id}:evaluate` || isNavigating}
                  onClick={() =>
                    latestSource &&
                    void triggerAction(latestSource, "evaluate", { configPath: "configs/eval.yaml" })
                  }
                >
                  {busyAction === `${latestSource?.id}:evaluate` ? "Working..." : "Evaluate"}
                </button>
              </div>
            </article>

            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Inference sandbox</strong>
                <span>{latestSource ? latestSource.name : "needs source"}</span>
              </div>
              <p>Open a managed inference branch from the latest completed train, finetune, or deploy run.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="secondary-button small"
                  disabled={!latestSource || busyAction === `${latestSource?.id}:open_inference` || isNavigating}
                  onClick={() =>
                    latestSource &&
                    void triggerAction(latestSource, "open_inference", { configPath: "configs/inference.yaml" })
                  }
                >
                  {busyAction === `${latestSource?.id}:open_inference` ? "Working..." : "Open inference"}
                </button>
              </div>
            </article>

            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Deploy latest</strong>
                <span>{latestSource ? latestSource.name : "needs source"}</span>
              </div>
              <p>Queue `configs/deploy.yaml` to the default Ollama deployment path.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="secondary-button small"
                  disabled={!latestSource || busyAction === `${latestSource?.id}:deploy` || isNavigating}
                  onClick={() =>
                    latestSource &&
                    void triggerAction(latestSource, "deploy", {
                      configPath: "configs/deploy.yaml",
                      deploymentTarget: "ollama",
                    })
                  }
                >
                  {busyAction === `${latestSource?.id}:deploy` ? "Working..." : "Deploy"}
                </button>
              </div>
            </article>
          </div>
        </section>

        <section className="panel aside-section">
          <div className="model-chip-header">
            <div>
              <h2 className="section-title">Titan Runtime</h2>
              <p className="control-label">
                Backend selection follows the same Titan probe used by the CLI, API, and generated hardware contract.
              </p>
            </div>
            <span className="status-pill">{mission.titan.preferred_training_backend}</span>
          </div>
          <div className="badge-row">
            <span className="status-pill">{mission.titan.silicon}</span>
            <span className="status-pill">{mission.titan.bandwidth_gbps ?? "n/a"} GB/s</span>
            <span className="status-pill">
              {mission.titan.supports_cuda
                ? mission.titan.cuda_compute_capability ?? "cuda-ready"
                : "metal-first"}
            </span>
            <span className="status-pill">
              {mission.titan.remote_execution
                ? mission.titan.cloud_provider ?? "remote-execution"
                : "on-device"}
            </span>
          </div>
        </section>
      </div>

      <div className="workspace-section-grid">
        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Running Branches</h2>
            <p className="control-label">
              Live managed instances with direct jump-off actions for the next lifecycle step.
            </p>
          </div>

          {runningInstances.length === 0 ? (
            <div className="dash-empty" style={{ padding: "2rem 1rem" }}>
              <p>No running branches right now.</p>
            </div>
          ) : (
            <div className="resource-list">
              {runningInstances.slice(0, 6).map((instance) => {
                const recommendation = topRecommendation(instance.recommendations);
                const actionKey = recommendation ? `${instance.id}:${recommendation.action}` : null;
                return (
                  <article key={instance.id} className="resource-card">
                    <div className="model-chip-header">
                      <strong>{instance.name}</strong>
                      <span>{instance.type}</span>
                    </div>
                    <p>
                      {instance.lifecycle.learning_mode ?? "managed"} · {instance.environment.kind} · {progressLabel(instance)}
                    </p>
                    <div className="badge-row">
                      <span className="status-pill">{instance.status}</span>
                      <span className="status-pill">{instance.lifecycle.stage ?? "queued"}</span>
                    </div>
                    <div className="workspace-actions">
                      <Link href={`/runs/${instance.id}`} className="secondary-button small">
                        Inspect
                      </Link>
                      {recommendation && latestRecommendationSource ? (
                        <button
                          type="button"
                          className="primary-button small"
                          disabled={busyAction === actionKey || isNavigating}
                          onClick={() => void dispatchRecommendation(instance, recommendation)}
                        >
                          {busyAction === actionKey ? "Working..." : recommendation.action}
                        </button>
                      ) : null}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Completed Sources</h2>
            <p className="control-label">
              Promote completed training and finetune outputs into evaluation, inference, deployment, or another finetune pass.
            </p>
          </div>

          {instances.filter((instance) => instance.status === "completed").length === 0 ? (
            <div className="dash-empty" style={{ padding: "2rem 1rem" }}>
              <p>No completed sources available yet.</p>
            </div>
          ) : (
            <div className="resource-list">
              {instances
                .filter((instance) => instance.status === "completed")
                .sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)))
                .slice(0, 6)
                .map((instance) => (
                  <article key={instance.id} className="resource-card">
                    <div className="model-chip-header">
                      <strong>{instance.name}</strong>
                      <span>{instance.type}</span>
                    </div>
                    <p>{formatTimestamp(instance.updated_at)}</p>
                    <div className="badge-row">
                      <span className="status-pill">{instance.lifecycle.learning_mode ?? "n/a"}</span>
                      <span className="status-pill">{instance.environment.kind}</span>
                    </div>
                    <div className="workspace-actions" style={{ flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className="secondary-button small"
                        disabled={busyAction === `${instance.id}:evaluate` || isNavigating}
                        onClick={() => void triggerAction(instance, "evaluate", { configPath: "configs/eval.yaml" })}
                      >
                        {busyAction === `${instance.id}:evaluate` ? "Working..." : "Evaluate"}
                      </button>
                      <button
                        type="button"
                        className="secondary-button small"
                        disabled={busyAction === `launch:finetune:${instance.id}` || isNavigating}
                        onClick={() => void launchFinetuneBranch(instance)}
                      >
                        {busyAction === `launch:finetune:${instance.id}` ? "Working..." : "Finetune"}
                      </button>
                      <button
                        type="button"
                        className="secondary-button small"
                        disabled={busyAction === `${instance.id}:open_inference` || isNavigating}
                        onClick={() => void triggerAction(instance, "open_inference", { configPath: "configs/inference.yaml" })}
                      >
                        {busyAction === `${instance.id}:open_inference` ? "Working..." : "Inference"}
                      </button>
                      <button
                        type="button"
                        className="secondary-button small"
                        disabled={busyAction === `${instance.id}:deploy` || isNavigating}
                        onClick={() =>
                          void triggerAction(instance, "deploy", {
                            configPath: "configs/deploy.yaml",
                            deploymentTarget: "ollama",
                          })
                        }
                      >
                        {busyAction === `${instance.id}:deploy` ? "Working..." : "Deploy"}
                      </button>
                    </div>
                  </article>
                ))}
            </div>
          )}
        </section>
      </div>

      <section className="panel aside-section">
        <div>
          <h2 className="section-title">Lab Recommendations</h2>
          <p className="control-label">
            Prioritized operational guidance from telemetry backlog, orchestration health, agents, and cluster state.
          </p>
        </div>
        <div className="resource-list">
          {mission.recommendations.map((recommendation) => (
            <article key={recommendation.id} className="resource-card">
              <div className="model-chip-header">
                <strong>{recommendation.title}</strong>
                <span>{recommendation.severity}</span>
              </div>
              <p>{recommendation.detail}</p>
              <div className="badge-row">
                {recommendation.metric_label && recommendation.metric_value ? (
                  <span className="status-pill">
                    {recommendation.metric_label}: {recommendation.metric_value}
                  </span>
                ) : null}
                <span className="status-pill">{recommendation.surface}</span>
              </div>
              <div className="workspace-actions">
                <a href={recommendation.href} className="secondary-button small">
                  Open surface
                </a>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
