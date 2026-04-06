'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useMemo, useState, useTransition } from 'react';

import {
  createManagedInstance,
  executeAutonomousLoop,
  planAutonomousLoop,
  runManagedInstanceAction,
  type FeedbackRecommendation,
  type InstanceSummary,
} from '@/lib/api';
import { AutonomyLoopPanel } from '@/components/autonomy-loop-panel';
import { formatCount } from '@/lib/formatting';
import { performanceMonitor } from '@/lib/performance';
import { ROUTES } from '@/lib/routes';
import type { TitanStatus } from '@/lib/titan-schema';
import { useMissionControl } from '@/hooks/use-mission-control';

const DEFAULT_SOURCE_MODEL = 'Qwen/Qwen2.5-Math-1.5B-Instruct';

function topRecommendation(
  recommendations?: FeedbackRecommendation[],
): FeedbackRecommendation | null {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }
  return recommendations.reduce((best, current) =>
    current.priority > best.priority ? current : best,
  );
}

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

function progressLabel(instance: InstanceSummary): string {
  if (typeof instance.progress?.percent === 'number') {
    return `${(instance.progress.percent * 100).toFixed(0)}%`;
  }
  return instance.progress?.stage ?? instance.status;
}

function latestLifecycleSource(instances: InstanceSummary[]): InstanceSummary | null {
  const completed = instances
    .filter(
      (instance) =>
        instance.status === 'completed' &&
        ['train', 'finetune', 'evaluate', 'deploy'].includes(instance.type),
    )
    .sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
  return completed[0] ?? null;
}

function autonomousTone(status: string): string {
  if (status === 'running' || status === 'ready') {
    return 'var(--accent)';
  }
  if (status === 'completed') {
    return '#6b8';
  }
  if (status === 'attention') {
    return '#d9a441';
  }
  return 'var(--danger)';
}

export default function Dashboard() {
  const router = useRouter();
  const [isNavigating, startTransition] = useTransition();
  const { mission, loading, error, refresh, replaceMission } = useMissionControl(7_500);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    performanceMonitor.start();
  }, []);

  const instances = useMemo(() => mission?.control_plane.instances ?? [], [mission?.control_plane.instances]);
  const runningInstances = mission?.watchlist.running_instances ?? [];
  const autonomous = mission?.autonomous;
  const autonomy = mission?.autonomy;
  const latestSource = useMemo(() => latestLifecycleSource(instances), [instances]);

  async function launchTrainingBranch(): Promise<void> {
    setBusyAction('launch:train');
    setNotice(null);
    try {
      const detail = await createManagedInstance({
        config_path: 'configs/train.yaml',
        start: true,
        user_level: 'hobbyist',
        lifecycle: {
          stage: 'train',
          origin: 'existing_model',
          learning_mode: 'qlora',
          source_model: latestSource?.lifecycle.source_model ?? DEFAULT_SOURCE_MODEL,
          deployment_targets: ['ollama'],
        },
        metadata: {
          source: 'dashboard_overview',
          action: 'launch_train',
        },
      });
      setNotice(`Training branch ${detail.name} launched.`);
      startTransition(() => router.push(`/runs/${detail.id}`));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Training launch failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function launchFinetuneBranch(source: InstanceSummary): Promise<void> {
    setBusyAction(`launch:finetune:${source.id}`);
    setNotice(null);
    try {
      const detail = await createManagedInstance({
        config_path: 'configs/finetune.yaml',
        start: true,
        parent_instance_id: source.id,
        user_level: 'hobbyist',
        lifecycle: {
          stage: 'finetune',
          origin: 'existing_model',
          learning_mode: 'qlora',
          source_model: source.lifecycle.source_model ?? DEFAULT_SOURCE_MODEL,
          deployment_targets: ['ollama'],
        },
        metadata: {
          source: 'dashboard_overview',
          action: 'launch_finetune',
        },
      });
      setNotice(`Finetune branch ${detail.name} launched from ${source.name}.`);
      startTransition(() => router.push(`/runs/${detail.id}`));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Finetune launch failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function triggerAction(
    instance: InstanceSummary,
    action: 'evaluate' | 'open_inference' | 'deploy',
    options: {
      configPath: string;
      deploymentTarget?: 'ollama';
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

  async function dispatchRecommendation(
    instance: InstanceSummary,
    recommendation: FeedbackRecommendation,
  ): Promise<void> {
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
      setNotice(error instanceof Error ? error.message : 'Recommended action failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function previewAutonomousPlan(): Promise<void> {
    setBusyAction('autonomous:plan');
    setNotice(null);
    try {
      const run = await planAutonomousLoop({ max_actions: 6 });
      setNotice(
        `Autonomous plan ${run.id} captured ${run.actions.length} action${run.actions.length === 1 ? '' : 's'}.`,
      );
      const refreshed = await refresh();
      if (refreshed) {
        replaceMission(refreshed);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Autonomous planning failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function queueAutonomousActions(): Promise<void> {
    setBusyAction('autonomous:execute');
    setNotice(null);
    try {
      const run = await executeAutonomousLoop({
        max_actions: 2,
        dry_run: false,
        start_instances: false,
      });
      const queued = Number(run.summary.created_instances ?? 0);
      setNotice(
        queued > 0
          ? `Autonomous loop queued ${queued} managed instance${queued === 1 ? '' : 's'}.`
          : `Autonomous loop ${run.status}.`,
      );
      const refreshed = await refresh();
      if (refreshed) {
        replaceMission(refreshed);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Autonomous execution failed.');
    } finally {
      setBusyAction(null);
    }
  }

  if (loading && !mission) {
    return (
      <div className="dashboard-content">
        <div className="dash-loading panel">
          <span>⟳</span>
          <span>Loading mission control…</span>
        </div>
      </div>
    );
  }

  if (error && !mission) {
    return (
      <div className="dashboard-content">
        <div className="dash-error-banner panel">
          <span>⚠</span>
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (!mission) {
    return null;
  }

  const titan = mission.titan as TitanStatus;
  const latestRecommendationSource = runningInstances[0] ?? latestSource;

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div className="dash-page-header-inner">
          <div>
            <span className="eyebrow">Lifecycle + Lab</span>
            <h1 className="dash-page-title">Mission Control</h1>
            <p className="dash-page-desc">
              Launch managed training branches, promote completed runs into finetune or deploy
              flows, and monitor the autonomous lab from a single responsive control surface.
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

      {error ? <div className="dash-error-banner panel">⚠ {error}</div> : null}
      {notice ? <div className="dash-note-banner panel">◎ {notice}</div> : null}

      {autonomy ? (
        <AutonomyLoopPanel
          autonomy={autonomy}
          title="Autonomous Loop"
          description="Shared loop state spanning datasets, training, evaluation, deployment, agents, and cluster capacity."
        />
      ) : null}

      <div className="workspace-summary-grid">
        {[
          {
            label: 'Ready Checks',
            value: `${mission.summary.ready_checks}/${mission.summary.total_checks}`,
          },
          { label: 'Managed Instances', value: formatCount(mission.summary.instances) },
          { label: 'Running', value: formatCount(mission.summary.running_instances) },
          { label: 'AutoML Sweeps', value: formatCount(mission.summary.running_sweeps) },
          { label: 'Loop Blockers', value: formatCount(mission.summary.autonomous_blockers) },
          { label: 'Telemetry Backlog', value: formatCount(mission.summary.telemetry_backlog) },
          { label: 'Titan Mode', value: mission.titan.mode },
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
                Each action posts into the managed instance API and routes execution through the
                same orchestration layer used by the CLI and TUI.
              </p>
            </div>
            <span className="status-pill">
              {titan.backend} · {titan.remote_execution ? 'cloud-aware' : 'local-first'}
            </span>
          </div>

          <div className="resource-list">
            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Start training</strong>
                <span>{titan.mode}</span>
              </div>
              <p>Launch `configs/train.yaml` with a managed QLoRA-first training branch.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="primary-button small"
                  disabled={busyAction === 'launch:train' || isNavigating}
                  onClick={() => void launchTrainingBranch()}
                >
                  {busyAction === 'launch:train' ? 'Working...' : 'Launch training'}
                </button>
              </div>
            </article>

            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Start finetune</strong>
                <span>{latestSource ? latestSource.name : 'needs source'}</span>
              </div>
              <p>Spawn `configs/finetune.yaml` from the latest completed managed branch.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="primary-button small"
                  disabled={
                    !latestSource ||
                    busyAction === `launch:finetune:${latestSource?.id}` ||
                    isNavigating
                  }
                  onClick={() => latestSource && void launchFinetuneBranch(latestSource)}
                >
                  {busyAction === `launch:finetune:${latestSource?.id}`
                    ? 'Working...'
                    : 'Launch finetune'}
                </button>
              </div>
            </article>

            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Evaluate latest</strong>
                <span>{latestSource ? latestSource.type : 'needs source'}</span>
              </div>
              <p>Queue `configs/eval.yaml` against the current best completed lifecycle source.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="secondary-button small"
                  disabled={
                    !latestSource || busyAction === `${latestSource?.id}:evaluate` || isNavigating
                  }
                  onClick={() =>
                    latestSource &&
                    void triggerAction(latestSource, 'evaluate', {
                      configPath: 'configs/eval.yaml',
                    })
                  }
                >
                  {busyAction === `${latestSource?.id}:evaluate` ? 'Working...' : 'Evaluate'}
                </button>
              </div>
            </article>

            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Inference sandbox</strong>
                <span>{latestSource ? latestSource.name : 'needs source'}</span>
              </div>
              <p>
                Open a managed inference branch from the latest completed train, finetune, or deploy
                run.
              </p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="secondary-button small"
                  disabled={
                    !latestSource ||
                    busyAction === `${latestSource?.id}:open_inference` ||
                    isNavigating
                  }
                  onClick={() =>
                    latestSource &&
                    void triggerAction(latestSource, 'open_inference', {
                      configPath: 'configs/inference.yaml',
                    })
                  }
                >
                  {busyAction === `${latestSource?.id}:open_inference`
                    ? 'Working...'
                    : 'Open inference'}
                </button>
              </div>
            </article>

            <article className="resource-card">
              <div className="model-chip-header">
                <strong>Deploy latest</strong>
                <span>{latestSource ? latestSource.name : 'needs source'}</span>
              </div>
              <p>Queue `configs/deploy.yaml` to the default Ollama deployment path.</p>
              <div className="workspace-actions">
                <button
                  type="button"
                  className="secondary-button small"
                  disabled={
                    !latestSource || busyAction === `${latestSource?.id}:deploy` || isNavigating
                  }
                  onClick={() =>
                    latestSource &&
                    void triggerAction(latestSource, 'deploy', {
                      configPath: 'configs/deploy.yaml',
                      deploymentTarget: 'ollama',
                    })
                  }
                >
                  {busyAction === `${latestSource?.id}:deploy` ? 'Working...' : 'Deploy'}
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
                Backend selection follows the same Titan probe used by the CLI, API, and generated
                hardware contract.
              </p>
            </div>
            <span className="status-pill">{mission.titan.preferred_training_backend}</span>
          </div>
          <div className="badge-row">
            <span className="status-pill">{titan.silicon}</span>
            <span className="status-pill">{titan.bandwidth_gbps ?? 'n/a'} GB/s</span>
            <span className="status-pill">
              {titan.runtime.selected ?? titan.engine.runtime_mode ?? 'python'}
            </span>
            <span className="status-pill">{titan.engine.cache_strategy}</span>
            <span className="status-pill">
              {titan.supports_cuda
                ? (titan.cuda_compute_capability ?? 'cuda-ready')
                : 'metal-first'}
            </span>
            <span className="status-pill">
              {titan.remote_execution ? (titan.cloud_provider ?? 'remote-execution') : 'on-device'}
            </span>
            {titan.runtime.gguf_support ? <span className="status-pill">gguf-ready</span> : null}
            {titan.engine.acceleration.cpp_kernels ? (
              <span className="status-pill">cpp-kernels</span>
            ) : null}
          </div>
          <p className="control-label" style={{ marginTop: '0.85rem' }}>
            {titan.engine.decode_model} · cache {titan.engine.cache_strategy} · sampler{' '}
            {(titan.runtime.sampler_stack ?? titan.engine.sampler_stack).join(' / ')}
          </p>
        </section>
      </div>

      {autonomous ? (
        <section className="panel aside-section">
          <div className="model-chip-header">
            <div>
              <h2 className="section-title">Loop Planner</h2>
              <p className="control-label">
                Planner-backed execution queue for the next lifecycle moves. This complements the
                richer autonomy state above.
              </p>
            </div>
            <span className="status-pill">
              {autonomous.ready ? 'ready' : autonomous.blockers.length ? 'blocked' : 'standby'}
            </span>
          </div>

          <div className="badge-row">
            <span className="status-pill">
              {autonomous.summary.executable_actions} executable actions
            </span>
            <span className="status-pill">{autonomous.summary.advisory_actions} advisories</span>
            <span className="status-pill">
              {autonomous.summary.telemetry_backlog} flagged prompts
            </span>
            <span className="status-pill">{autonomous.summary.idle_nodes} idle nodes</span>
          </div>

          {autonomous.blockers.length > 0 ? (
            <div className="dash-error-banner panel" style={{ marginTop: '1rem' }}>
              ⚠ {autonomous.blockers[0]}
            </div>
          ) : null}

          <div className="workspace-actions" style={{ marginTop: '1rem' }}>
            <button
              type="button"
              className="primary-button small"
              disabled={busyAction === 'autonomous:execute'}
              onClick={() => void queueAutonomousActions()}
            >
              {busyAction === 'autonomous:execute' ? 'Queuing...' : 'Queue Next Actions'}
            </button>
            <button
              type="button"
              className="secondary-button small"
              disabled={busyAction === 'autonomous:plan'}
              onClick={() => void previewAutonomousPlan()}
            >
              {busyAction === 'autonomous:plan' ? 'Planning...' : 'Capture Plan'}
            </button>
          </div>

          {autonomous.actions.length > 0 ? (
            <div style={{ marginTop: '1rem' }}>
              <h3 className="section-title" style={{ marginBottom: '0.75rem' }}>
                Next Actions
              </h3>
              <div className="resource-list">
                {autonomous.actions.map((action) => (
                  <article key={action.id} className="resource-card">
                    <div className="model-chip-header">
                      <strong>{action.title}</strong>
                      <span>p{action.priority}</span>
                    </div>
                    <p>{action.detail}</p>
                    <div className="badge-row">
                      <span
                        className="status-pill"
                        style={{ color: autonomousTone(action.executable ? 'ready' : 'attention') }}
                      >
                        {action.executable ? 'executable' : 'advisory'}
                      </span>
                      {action.source_instance_name ? (
                        <span className="status-pill">{action.source_instance_name}</span>
                      ) : null}
                    </div>
                    <div className="workspace-actions">
                      <Link href={action.href as never} className="secondary-button small">
                        Open surface
                      </Link>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          ) : null}
        </section>
      ) : null}

      <div className="workspace-section-grid">
        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Running Branches</h2>
            <p className="control-label">
              Live managed instances with direct jump-off actions for the next lifecycle step.
            </p>
          </div>

          {runningInstances.length === 0 ? (
            <div className="dash-empty" style={{ padding: '2rem 1rem' }}>
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
                      {instance.lifecycle.learning_mode ?? 'managed'} · {instance.environment.kind}{' '}
                      · {progressLabel(instance)}
                    </p>
                    <div className="badge-row">
                      <span className="status-pill">{instance.status}</span>
                      <span className="status-pill">{instance.lifecycle.stage ?? 'queued'}</span>
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
                          {busyAction === actionKey ? 'Working...' : recommendation.action}
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
              Promote completed training and finetune outputs into evaluation, inference,
              deployment, or another finetune pass.
            </p>
          </div>

          {instances.filter((instance) => instance.status === 'completed').length === 0 ? (
            <div className="dash-empty" style={{ padding: '2rem 1rem' }}>
              <p>No completed sources available yet.</p>
            </div>
          ) : (
            <div className="resource-list">
              {instances
                .filter((instance) => instance.status === 'completed')
                .sort((left, right) =>
                  String(right.updated_at).localeCompare(String(left.updated_at)),
                )
                .slice(0, 6)
                .map((instance) => (
                  <article key={instance.id} className="resource-card">
                    <div className="model-chip-header">
                      <strong>{instance.name}</strong>
                      <span>{instance.type}</span>
                    </div>
                    <p>{formatTimestamp(instance.updated_at)}</p>
                    <div className="badge-row">
                      <span className="status-pill">
                        {instance.lifecycle.learning_mode ?? 'n/a'}
                      </span>
                      <span className="status-pill">{instance.environment.kind}</span>
                    </div>
                    <div className="workspace-actions" style={{ flexWrap: 'wrap' }}>
                      <button
                        type="button"
                        className="secondary-button small"
                        disabled={busyAction === `${instance.id}:evaluate` || isNavigating}
                        onClick={() =>
                          void triggerAction(instance, 'evaluate', {
                            configPath: 'configs/eval.yaml',
                          })
                        }
                      >
                        {busyAction === `${instance.id}:evaluate` ? 'Working...' : 'Evaluate'}
                      </button>
                      <button
                        type="button"
                        className="secondary-button small"
                        disabled={busyAction === `launch:finetune:${instance.id}` || isNavigating}
                        onClick={() => void launchFinetuneBranch(instance)}
                      >
                        {busyAction === `launch:finetune:${instance.id}`
                          ? 'Working...'
                          : 'Finetune'}
                      </button>
                      <button
                        type="button"
                        className="secondary-button small"
                        disabled={busyAction === `${instance.id}:open_inference` || isNavigating}
                        onClick={() =>
                          void triggerAction(instance, 'open_inference', {
                            configPath: 'configs/inference.yaml',
                          })
                        }
                      >
                        {busyAction === `${instance.id}:open_inference`
                          ? 'Working...'
                          : 'Inference'}
                      </button>
                      <button
                        type="button"
                        className="secondary-button small"
                        disabled={busyAction === `${instance.id}:deploy` || isNavigating}
                        onClick={() =>
                          void triggerAction(instance, 'deploy', {
                            configPath: 'configs/deploy.yaml',
                            deploymentTarget: 'ollama',
                          })
                        }
                      >
                        {busyAction === `${instance.id}:deploy` ? 'Working...' : 'Deploy'}
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
            Prioritized operational guidance from telemetry backlog, orchestration health, agents,
            and cluster state.
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
                <Link href={recommendation.href as never} className="secondary-button small">
                  Open surface
                </Link>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
