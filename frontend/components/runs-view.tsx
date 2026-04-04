'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

import {
  getInstances,
  getOrchestrationRuns,
  getOrchestrationSummary,
  getWorkspaceOverview,
  runManagedInstanceAction,
  type FeedbackRecommendation,
  type InstanceSummary,
  type OrchestrationRun,
  type OrchestrationSummary,
  type WorkspaceOverview,
  type WorkspaceOrchestrationTemplate,
} from '@/lib/api';
import { formatCount, formatFixed, formatPercent } from '@/lib/formatting';
import { useLabMetadata } from '@/hooks/use-lab-metadata';
import { ROUTES } from '@/lib/routes';

import { AppShell } from '@/components/layout/app-shell';
import { NewInstancePanel } from '@/components/new-instance-panel';
import { MetricBadge } from '@/components/panels/metric-badge';
import { PageHeader } from '@/components/ui/page-header';
import { StatePanel } from '@/components/ui/state-panel';

type ControlPlaneState = {
  instances: InstanceSummary[];
  runs: OrchestrationRun[];
  summary: OrchestrationSummary | null;
  loading: boolean;
  error: string | null;
};

const INITIAL_CONTROL_PLANE_STATE: ControlPlaneState = {
  instances: [],
  runs: [],
  summary: null,
  loading: true,
  error: null,
};

function formatTimestamp(value?: string | null): string {
  if (!value) {
    return 'n/a';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function numericMetric(summary: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = summary[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function runningInstances(instances: InstanceSummary[]): number {
  return instances.filter((instance) => instance.status === 'running').length;
}

const DECISION_TONES: Record<string, string> = {
  deploy: 'decision-deploy',
  finetune: 'decision-finetune',
  retrain: 'decision-retrain',
  evaluate: 'decision-evaluate',
  inference: 'decision-inference',
};

function DecisionBadge({
  decision,
}: {
  decision?: { action: string; explanation: string } | null;
}) {
  if (!decision) {
    return null;
  }
  const tone = DECISION_TONES[decision.action] ?? 'decision-default';
  return (
    <span className={`decision-badge ${tone}`} title={decision.explanation}>
      {decision.action}
    </span>
  );
}

function ProgressBar({ percent }: { percent?: number | null }) {
  if (typeof percent !== 'number' || !Number.isFinite(percent)) {
    return null;
  }
  const clamped = Math.max(0, Math.min(100, percent * 100));
  return (
    <div className="progress-track">
      <div className="progress-fill" style={{ width: `${clamped}%` }} />
      <span className="progress-label">{clamped.toFixed(0)}%</span>
    </div>
  );
}

function topRecommendation(recs?: FeedbackRecommendation[]): FeedbackRecommendation | null {
  if (!recs || !recs.length) {
    return null;
  }
  return recs.reduce((best, item) => (item.priority > best.priority ? item : best), recs[0]);
}

export function RunsView() {
  const router = useRouter();
  const metadata = useLabMetadata();
  const [controlPlane, setControlPlane] = useState<ControlPlaneState>(INITIAL_CONTROL_PLANE_STATE);
  const [templates, setTemplates] = useState<WorkspaceOrchestrationTemplate[]>([]);
  const [createNotice, setCreateNotice] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadControlPlane() {
      setControlPlane((current) => ({ ...current, loading: true, error: null }));
      const [instancesResult, runsResult, summaryResult, workspaceResult] =
        await Promise.allSettled([
          getInstances(),
          getOrchestrationRuns(),
          getOrchestrationSummary(),
          getWorkspaceOverview(),
        ]);
      if (!active) {
        return;
      }
      const errors: string[] = [];
      const resolveResult = <T,>(
        result: PromiseSettledResult<T>,
        fallback: T,
        label: string,
      ): T => {
        if (result.status === 'fulfilled') {
          return result.value;
        }
        errors.push(
          `${label}: ${result.reason instanceof Error ? result.reason.message : 'request failed'}`,
        );
        return fallback;
      };
      setControlPlane({
        instances: resolveResult(instancesResult, [], 'instances'),
        runs: resolveResult(runsResult, [], 'orchestration runs'),
        summary: resolveResult(summaryResult, null, 'orchestration summary'),
        loading: false,
        error: errors.length ? errors.join(' | ') : null,
      });
      const workspaceOverview = resolveResult<WorkspaceOverview | null>(
        workspaceResult,
        null,
        'workspace overview',
      );
      setTemplates(workspaceOverview?.orchestration_templates ?? []);
    }

    void loadControlPlane();
    return () => {
      active = false;
    };
  }, []);

  const artifactRuns = [...metadata.runs].sort((left, right) =>
    String(right.output_dir).localeCompare(String(left.output_dir)),
  );
  const latestArtifactRun = artifactRuns[0];
  const instances = [...controlPlane.instances].sort((left, right) =>
    String(right.updated_at).localeCompare(String(left.updated_at)),
  );
  const orchestrationRuns = [...controlPlane.runs].sort((left, right) =>
    String(right.updated_at).localeCompare(String(left.updated_at)),
  );
  const openCircuits = Array.isArray(controlPlane.summary?.open_circuits)
    ? controlPlane.summary?.open_circuits
    : [];

  function lifecycleLabel(instance: InstanceSummary) {
    return instance.lifecycle.stage ?? instance.progress?.stage ?? 'queued';
  }

  function handleCreated(instance: InstanceSummary) {
    setCreateNotice(`Launched ${instance.type} instance ${instance.name}.`);
    setControlPlane((current) => ({
      ...current,
      instances: [instance, ...current.instances],
    }));
  }

  async function triggerQuickAction(instance: InstanceSummary, rec: FeedbackRecommendation) {
    const key = `${instance.id}:${rec.action}`;
    setBusyAction(key);
    setActionNotice(null);
    try {
      const nextDetail = await runManagedInstanceAction(instance.id, {
        action: rec.action,
        config_path: rec.config_path ?? undefined,
        deployment_target: rec.deployment_target ?? undefined,
        start: true,
      });
      setActionNotice(`Launched ${rec.action} → ${nextDetail.name}`);
      router.push(`/runs/${nextDetail.id}`);
    } catch (nextError) {
      setActionNotice(
        `Action failed: ${nextError instanceof Error ? nextError.message : 'unknown error'}`,
      );
    } finally {
      setBusyAction(null);
    }
  }

  return (
    <AppShell>
      <section className="route-stack">
        <PageHeader
          eyebrow="Managed Runs"
          title="AI Factory Control Center"
          description="Launch managed lifecycle branches, monitor training and evaluation progress, and move cleanly from train to evaluate to inference and publish."
          metrics={[
            { label: 'Instances', value: formatCount(instances.length) },
            {
              label: 'Running',
              value: formatCount(runningInstances(instances)),
              tone: 'secondary',
            },
            {
              label: 'Control-plane runs',
              value: formatCount(controlPlane.summary?.runs ?? orchestrationRuns.length),
              tone: 'accent',
            },
            {
              label: 'Latest eval loss',
              value: formatFixed(latestArtifactRun?.metrics.eval_loss),
            },
          ]}
          actions={
            <>
              <Link className="ghost-button small" href={ROUTES.workspace}>
                Workspace guide
              </Link>
              <Link className="primary-button small" href={ROUTES.solve}>
                Open solve workspace
              </Link>
            </>
          }
        />

        {actionNotice ? (
          <StatePanel
            eyebrow="Quick Action"
            title="Action dispatched."
            description={actionNotice}
          />
        ) : null}

        {createNotice ? (
          <StatePanel
            eyebrow="Launch Flow"
            title="Managed branch created."
            description={createNotice}
          />
        ) : null}

        {controlPlane.loading && !instances.length && !orchestrationRuns.length ? (
          <StatePanel
            eyebrow="Loading"
            title="Managed instances are loading."
            description="The control plane is scanning recent instances, orchestration runs, and task state."
            tone="loading"
          />
        ) : null}

        {controlPlane.error && !instances.length && !orchestrationRuns.length ? (
          <StatePanel
            eyebrow="Unavailable"
            title="Control-plane data could not be loaded."
            description={controlPlane.error}
            tone="error"
          />
        ) : null}

        {!controlPlane.loading && !instances.length && !orchestrationRuns.length ? (
          <StatePanel
            eyebrow="No Instances"
            title="No managed instances have been launched yet."
            description="Start with `ai-factory new --config configs/finetune.yaml` or post to `/v1/instances` to create the first tracked workflow."
          />
        ) : null}

        {controlPlane.summary ? (
          <article className="panel catalog-panel">
            <div className="message-meta">
              <span>Control plane</span>
              <span className="status-pill">
                {openCircuits.length
                  ? `${openCircuits.length} open circuits`
                  : 'all circuits healthy'}
              </span>
            </div>
            <h2>Orchestration summary</h2>
            <div className="badge-row">
              <MetricBadge label="Tasks" value={formatCount(controlPlane.summary.tasks)} />
              <MetricBadge
                label="Ready"
                value={formatCount(controlPlane.summary.task_status_counts?.ready)}
                tone="secondary"
              />
              <MetricBadge
                label="Running"
                value={formatCount(controlPlane.summary.task_status_counts?.running)}
                tone="accent"
              />
              <MetricBadge
                label="Retry waiting"
                value={formatCount(controlPlane.summary.task_status_counts?.retry_waiting)}
              />
            </div>
            {openCircuits.length ? (
              <div className="preview-block subtle">
                <strong>Open circuits</strong>
                <p>{openCircuits.join(', ')}</p>
              </div>
            ) : null}
          </article>
        ) : null}

        <NewInstancePanel templates={templates} onCreated={handleCreated} />

        {instances.length ? (
          <>
            <div className="section-heading">
              <h2>Recent instances</h2>
              <p>Every managed operation projects into the same instance model.</p>
            </div>
            <div className="card-grid compact">
              {instances.slice(0, 8).map((instance) => {
                const rec = topRecommendation(instance.recommendations);
                const actionKey = rec ? `${instance.id}:${rec.action}` : null;
                return (
                  <article key={instance.id} className="panel catalog-panel">
                    <div className="message-meta">
                      <span>{instance.type}</span>
                      <span className="status-pill">{instance.status}</span>
                      <DecisionBadge decision={instance.decision} />
                    </div>
                    <h2>{instance.name}</h2>
                    <ProgressBar percent={instance.progress?.percent} />
                    <div className="badge-row">
                      <MetricBadge label="Env" value={instance.environment.kind} />
                      <MetricBadge
                        label="Lifecycle"
                        value={lifecycleLabel(instance)}
                        tone="secondary"
                      />
                      <MetricBadge
                        label="Mode"
                        value={instance.lifecycle.learning_mode ?? 'n/a'}
                        tone="accent"
                      />
                    </div>
                    <div className="preview-block subtle">
                      <strong>Lifecycle</strong>
                      <p>
                        {instance.lifecycle.origin ?? 'origin pending'} •{' '}
                        {instance.lifecycle.source_model ?? 'profile-managed source'}
                      </p>
                      <p>
                        Updated {formatTimestamp(instance.updated_at)}
                        {instance.parent_instance_id
                          ? ` • child of ${instance.parent_instance_id}`
                          : ''}
                      </p>
                    </div>
                    <div className="badge-row">
                      <MetricBadge
                        label="Progress"
                        value={
                          typeof instance.progress?.percent === 'number'
                            ? formatPercent(instance.progress.percent, 0)
                            : 'n/a'
                        }
                      />
                      <MetricBadge
                        label="Accuracy"
                        value={formatPercent(
                          numericMetric(instance.metrics_summary, ['accuracy']),
                          1,
                        )}
                        tone="secondary"
                      />
                      <MetricBadge
                        label="Latency"
                        value={formatFixed(
                          numericMetric(instance.metrics_summary, ['avg_latency_s']),
                          2,
                        )}
                      />
                    </div>
                    <div className="workspace-actions">
                      <Link className="secondary-button small" href={`/runs/${instance.id}`}>
                        Inspect branch
                      </Link>
                      {rec ? (
                        <button
                          className="primary-button small"
                          type="button"
                          disabled={busyAction === actionKey}
                          onClick={() => void triggerQuickAction(instance, rec)}
                        >
                          {busyAction === actionKey ? 'Working...' : rec.action}
                        </button>
                      ) : null}
                    </div>
                  </article>
                );
              })}
            </div>
          </>
        ) : null}

        {orchestrationRuns.length ? (
          <>
            <div className="section-heading">
              <h2>Orchestration runs</h2>
              <p>Durable run records backed by the SQLite control plane.</p>
            </div>
            <div className="card-grid compact">
              {orchestrationRuns.slice(0, 8).map((run) => (
                <article key={run.id} className="panel catalog-panel">
                  <div className="message-meta">
                    <span>{run.legacy_instance_id ?? 'managed run'}</span>
                    <span className="status-pill">{run.status}</span>
                  </div>
                  <h2>{run.name}</h2>
                  <div className="preview-block subtle">
                    <strong>Run lineage</strong>
                    <p>Run id {run.id}</p>
                    <p>
                      Root {run.root_run_id ?? run.id}
                      {run.parent_run_id ? ` • parent ${run.parent_run_id}` : ''}
                    </p>
                  </div>
                  <div className="badge-row">
                    <MetricBadge label="Created" value={formatTimestamp(run.created_at)} />
                    <MetricBadge
                      label="Updated"
                      value={formatTimestamp(run.updated_at)}
                      tone="secondary"
                    />
                  </div>
                </article>
              ))}
            </div>
          </>
        ) : null}

        {metadata.loading && !artifactRuns.length ? (
          <StatePanel
            eyebrow="Loading"
            title="Artifact runs are loading."
            description="AI-Factory is scanning local training manifests and metric summaries."
            tone="loading"
          />
        ) : null}

        {metadata.error && !artifactRuns.length ? (
          <StatePanel
            eyebrow="Unavailable"
            title="Artifact metadata could not be loaded."
            description={metadata.error}
            tone="error"
          />
        ) : null}

        {!metadata.loading && !artifactRuns.length ? (
          <StatePanel
            eyebrow="No Artifacts"
            title="No local training artifacts have been discovered yet."
            description="Run a training profile directly or create a managed training instance to populate the artifact registry."
          />
        ) : null}

        {artifactRuns.length ? (
          <>
            <div className="section-heading">
              <h2>Training artifacts</h2>
              <p>The existing artifact registry remains available for low-level run inspection.</p>
            </div>
            <div className="card-grid compact">
              {artifactRuns.map((run) => (
                <article key={run.run_id ?? run.run_name} className="panel catalog-panel">
                  <div className="message-meta">
                    <span>{run.profile_name ?? 'profile'}</span>
                    <span className="status-pill">{run.base_model}</span>
                  </div>
                  <h2>{run.run_name}</h2>
                  <div className="badge-row">
                    <MetricBadge
                      label="Trainable"
                      value={formatPercent(run.model_report.trainable_ratio, 2)}
                    />
                    <MetricBadge
                      label="Eval loss"
                      value={formatFixed(run.metrics.eval_loss)}
                      tone="accent"
                    />
                  </div>
                  <div className="preview-block subtle">
                    <strong>Artifacts</strong>
                    <p>{run.output_dir}</p>
                  </div>
                </article>
              ))}
            </div>
          </>
        ) : null}
      </section>
    </AppShell>
  );
}
