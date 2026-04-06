import type { AutonomyOverview } from '@/lib/api';

type Props = {
  autonomy: AutonomyOverview;
  title?: string;
  description?: string;
  maxStages?: number;
  maxActions?: number;
  compact?: boolean;
};

function tone(status: string): string {
  if (status === 'blocked') return 'var(--danger)';
  if (status === 'degraded' || status === 'attention') return '#d9a441';
  if (status === 'active') return 'var(--accent)';
  if (status === 'ready') return '#6b8';
  return '#8b949e';
}

export function AutonomyLoopPanel({
  autonomy,
  title = 'Autonomous Loop',
  description,
  maxStages = 4,
  maxActions = 4,
  compact = false,
}: Props) {
  const stages = autonomy.stages.slice(0, maxStages);
  const actions = autonomy.next_actions.slice(0, maxActions);

  return (
    <section className="panel aside-section">
      <div className="model-chip-header">
        <div>
          <h2 className="section-title">{title}</h2>
          <p className="control-label">{description ?? autonomy.summary}</p>
        </div>
        <span
          className="status-pill"
          style={{ borderColor: tone(autonomy.status), color: tone(autonomy.status) }}
        >
          {autonomy.mode} · {autonomy.status}
        </span>
      </div>

      <div className="workspace-summary-grid" style={{ marginTop: '1rem' }}>
        {[
          { label: 'Active Runs', value: autonomy.active_runs },
          { label: 'Sweeps', value: autonomy.running_sweeps },
          { label: 'Telemetry', value: autonomy.telemetry_backlog },
          { label: 'Circuits', value: autonomy.open_circuits },
          { label: 'Stalled', value: autonomy.stalled_runs },
          { label: 'Schedulable', value: autonomy.capacity.schedulable_trials },
        ].map((item) => (
          <div key={item.label} className="workspace-summary-card panel">
            <span className="workspace-summary-value">{item.value}</span>
            <span className="workspace-summary-label">{item.label}</span>
          </div>
        ))}
      </div>

      {!compact ? (
        <div className="workspace-section-grid" style={{ marginTop: '1rem' }}>
          <div className="resource-list">
            {stages.map((stage) => (
              <a
                key={stage.id}
                href={stage.href}
                className="resource-card"
                style={{ borderTop: `3px solid ${tone(stage.status)}` }}
              >
                <div className="model-chip-header">
                  <strong>{stage.title}</strong>
                  <span style={{ color: tone(stage.status), textTransform: 'capitalize' }}>
                    {stage.status}
                  </span>
                </div>
                <p>{stage.headline}</p>
                <p className="control-label">{stage.detail}</p>
                {stage.metric_label ? (
                  <div className="badge-row">
                    <span className="status-pill">
                      {stage.metric_label}: {stage.metric_value ?? 'n/a'}
                    </span>
                  </div>
                ) : null}
              </a>
            ))}
          </div>

          <div className="resource-list">
            {actions.map((action) => (
              <a
                key={action.id}
                href={action.href}
                className="resource-card"
                style={{
                  borderTop: `3px solid ${action.blocking ? 'var(--danger)' : tone(autonomy.status)}`,
                }}
              >
                <div className="model-chip-header">
                  <strong>{action.title}</strong>
                  <span style={{ textTransform: 'capitalize' }}>{action.category}</span>
                </div>
                <p>{action.detail}</p>
                {action.command ? (
                  <code style={{ fontSize: '0.8rem', opacity: 0.8 }}>{action.command}</code>
                ) : null}
              </a>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
