'use client';

import Link from 'next/link';

import { AutonomyLoopPanel } from '@/components/autonomy-loop-panel';
import { formatCount } from '@/lib/formatting';
import { useMissionControl } from '@/hooks/use-mission-control';
import { ROUTES } from '@/lib/routes';
import type { TitanStatus } from '@/lib/titan-schema';

function nodeTone(status: string) {
  if (status === 'online') {
    return 'var(--accent)';
  }
  if (status === 'idle') {
    return 'var(--secondary)';
  }
  return 'var(--danger)';
}

export default function ClusterPage() {
  const { mission, loading, error, refresh } = useMissionControl(10_000);
  const titan = mission?.titan as TitanStatus | undefined;
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
              Track node availability, see where active work is running, and move quickly into
              training or monitoring when capacity opens up.
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

      {error ? <div className="dash-error-banner panel">⚠ {error}</div> : null}

      {titan ? (
        <section className="panel aside-section" style={{ marginBottom: '1.5rem' }}>
          <div className="model-chip-header">
            <div>
              <h2 className="section-title">Titan Status</h2>
              <p className="control-label">
                {titan.silicon} · {titan.mode} · {titan.bandwidth_gbps ?? 'n/a'} GB/s
              </p>
            </div>
            <span className="status-pill">
              {titan.silent_mode
                ? `Mac-Silent ${titan.gpu_cap_pct}%`
                : `GPU Cap ${titan.gpu_cap_pct}%`}
            </span>
          </div>
          <div className="badge-row" style={{ marginTop: '0.75rem' }}>
            <span className="status-pill">{titan.backend}</span>
            <span className="status-pill">{titan.scheduler.runtime}</span>
            <span className="status-pill">
              {titan.runtime.selected ?? titan.engine.runtime_mode ?? 'python'}
            </span>
            <span className="status-pill">{titan.engine.cache_strategy}</span>
            <span className="status-pill">{titan.quantization.formats.join(' / ')}</span>
            <span className="status-pill">{titan.telemetry.bridge}</span>
            {titan.runtime.gguf_support ? <span className="status-pill">gguf-ready</span> : null}
            {titan.engine.acceleration.cpp_kernels ? (
              <span className="status-pill">cpp-kernels</span>
            ) : null}
          </div>
          <p className="control-label" style={{ marginTop: '0.75rem' }}>
            {titan.engine.decode_model} · queue depth {titan.engine.scheduler_queue_depth} · sampler{' '}
            {(titan.runtime.sampler_stack ?? titan.engine.sampler_stack).join(' / ')}
          </p>
        </section>
      ) : null}

      <div className="workspace-summary-grid">
        {[
          { label: 'Nodes', value: mission?.summary.cluster_nodes },
          { label: 'Running Jobs', value: mission?.summary.running_instances },
          { label: 'Open Circuits', value: mission?.summary.open_circuits },
          { label: 'Telemetry Backlog', value: mission?.summary.telemetry_backlog },
          { label: 'Agents', value: mission?.summary.active_agents },
          { label: 'Sweeps', value: mission?.summary.running_sweeps },
        ].map((item) => (
          <div key={item.label} className="workspace-summary-card panel">
            <span className="workspace-summary-value">{formatCount(item.value)}</span>
            <span className="workspace-summary-label">{item.label}</span>
          </div>
        ))}
      </div>

      {mission?.autonomy ? (
        <AutonomyLoopPanel
          autonomy={mission.autonomy}
          title="Dispatch Posture"
          description="Cluster state is evaluated against queue demand, sweep pressure, and orchestration health."
          maxStages={2}
          maxActions={2}
          compact
        />
      ) : null}

      {mission?.autonomy ? (
        <section className="panel aside-section" style={{ marginTop: '1.5rem' }}>
          <div className="model-chip-header">
            <div>
              <h2 className="section-title">Capacity Envelope</h2>
              <p className="control-label">
                Resource classes and execution modes are derived from live orchestration tasks and
                active runs.
              </p>
            </div>
            <span className="status-pill">{mission.autonomy.capacity.bottleneck}</span>
          </div>
          <div className="badge-row" style={{ marginTop: '0.75rem' }}>
            <span className="status-pill">
              gpu tasks {mission.autonomy.capacity.active_gpu_tasks}
            </span>
            <span className="status-pill">
              cpu tasks {mission.autonomy.capacity.active_cpu_tasks}
            </span>
            <span className="status-pill">
              local {mission.autonomy.capacity.execution_modes.local ?? 0}
            </span>
            <span className="status-pill">
              cloud {mission.autonomy.capacity.execution_modes.cloud ?? 0}
            </span>
            <span className="status-pill">
              parallelism {mission.autonomy.capacity.suggested_parallelism}
            </span>
          </div>
        </section>
      ) : null}

      <div className="workspace-section-grid">
        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Node Health</h2>
            <p className="control-label">
              Local and remote capacity visible from the same surface.
            </p>
          </div>

          <div className="workspace-actions" style={{ marginTop: '0.75rem' }}>
            <button type="button" className="secondary-button small" onClick={() => void refresh()}>
              Refresh cluster
            </button>
          </div>

          {loading && !nodes.length ? (
            <div className="dash-loading">
              <span>⟳</span> Loading cluster nodes…
            </div>
          ) : nodes.length === 0 ? (
            <div className="dash-empty" style={{ padding: '2rem 1rem' }}>
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
                  <p>
                    {node.type} · {node.memory}
                  </p>
                  <div className="monitor-progress-track" style={{ marginTop: '0.65rem' }}>
                    <div
                      className="monitor-progress-fill"
                      data-status={
                        node.status === 'offline'
                          ? 'failed'
                          : node.status === 'idle'
                            ? 'completed'
                            : 'running'
                      }
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
            <p className="control-label">Active branches currently occupying cluster capacity.</p>
          </div>

          {runningInstances.length === 0 ? (
            <div className="dash-empty" style={{ padding: '2rem 1rem' }}>
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
                    {instance.environment.kind} · {instance.lifecycle.learning_mode ?? 'managed'} ·{' '}
                    {instance.progress?.stage ?? instance.status}
                  </p>
                  {typeof instance.progress?.percent === 'number' ? (
                    <div className="monitor-progress-track" style={{ marginTop: '0.65rem' }}>
                      <div
                        className="monitor-progress-fill"
                        data-status="running"
                        style={{
                          width: `${Math.max(0, Math.min(100, instance.progress.percent * 100))}%`,
                        }}
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
