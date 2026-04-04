'use client';

import { useState } from 'react';

import { runAutonomousCampaign, type AutonomousCampaign } from '@/lib/api';
import { useMissionControl } from '@/hooks/use-mission-control';

function tone(status: string): string {
  if (['running', 'active', 'ready'].includes(status)) {
    return 'var(--accent)';
  }
  if (['blocked', 'failed', 'degraded'].includes(status)) {
    return 'var(--danger)';
  }
  return 'var(--muted)';
}

function formatStamp(value?: string | null): string {
  if (!value) {
    return 'n/a';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

export default function AutonomousPage() {
  const { mission, loading, error, refresh, replaceMission } = useMissionControl(8_000);
  const [experimentName, setExperimentName] = useState('Expo autonomy wave');
  const [goal, setGoal] = useState(
    'Convert telemetry into the next finetune branch and reconcile lineage.',
  );
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  async function launchCampaign(autoStart: boolean) {
    if (!experimentName.trim() || !goal.trim() || busy) {
      return;
    }
    setBusy(true);
    setNotice(null);
    try {
      const response = await runAutonomousCampaign({
        experiment_name: experimentName,
        goal,
        auto_start: autoStart,
        max_actions: 3,
      });
      setNotice(
        `${response.campaign.experiment_name} created with status ${response.campaign.status}.`,
      );
      const nextMission = await refresh();
      if (nextMission) {
        replaceMission(nextMission);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Autonomous campaign launch failed.');
    } finally {
      setBusy(false);
    }
  }

  const autonomy = mission?.autonomy;
  const campaigns = mission?.autonomous?.campaigns ?? [];
  const readyActions = mission?.autonomous?.ready_actions ?? [];
  const lineageGaps = mission?.lineage?.gaps ?? [];
  const placements = mission?.cluster?.placements ?? [];

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">V2 Lab → Autonomous</span>
          <h1 className="dash-page-title">Autonomous Loop Director</h1>
          <p className="dash-page-desc">
            Turn mission control state into executable campaign waves, reconcile lineage, and keep
            cluster capacity pointed at the next best branch.
          </p>
        </div>
      </div>

      {error ? <div className="dash-error-banner panel">⚠ {error}</div> : null}
      {notice ? <div className="panel state-panel">{notice}</div> : null}

      <div className="workspace-summary-grid">
        {[
          { label: 'Loop', value: autonomy?.status ?? 'n/a' },
          { label: 'Mode', value: autonomy?.mode ?? 'n/a' },
          { label: 'Ready Actions', value: mission?.summary.ready_autonomous_actions ?? 0 },
          { label: 'Campaigns', value: mission?.summary.autonomous_campaigns ?? 0 },
          { label: 'Lineage Gaps', value: mission?.summary.lineage_gaps ?? 0 },
          { label: 'Open Circuits', value: mission?.summary.open_circuits ?? 0 },
        ].map((item) => (
          <div key={item.label} className="workspace-summary-card panel">
            <span
              className="workspace-summary-value"
              style={{ color: tone(String(item.value).toLowerCase()) }}
            >
              {String(item.value)}
            </span>
            <span className="workspace-summary-label">{item.label}</span>
          </div>
        ))}
      </div>

      <div className="workspace-section-grid">
        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Launch Wave</h2>
            <p className="control-label">
              Package the current repo state into an autonomous campaign. `Auto-start` immediately
              turns ready actions into managed instances.
            </p>
          </div>
          <div style={{ display: 'grid', gap: '0.9rem', marginTop: '1rem' }}>
            <div className="input-group">
              <label className="control-label" htmlFor="autonomy-name">
                Campaign name
              </label>
              <input
                id="autonomy-name"
                value={experimentName}
                onChange={(event) => setExperimentName(event.target.value)}
              />
            </div>
            <div className="input-group">
              <label className="control-label" htmlFor="autonomy-goal">
                Goal
              </label>
              <textarea
                id="autonomy-goal"
                value={goal}
                onChange={(event) => setGoal(event.target.value)}
                rows={4}
                style={{ resize: 'vertical' }}
              />
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <button
                className="secondary-button"
                disabled={busy}
                onClick={() => void launchCampaign(false)}
              >
                {busy ? 'Planning...' : 'Plan Campaign'}
              </button>
              <button
                className="primary-button"
                disabled={busy}
                onClick={() => void launchCampaign(true)}
              >
                {busy ? 'Dispatching...' : 'Plan + Dispatch'}
              </button>
            </div>
          </div>
        </section>

        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Ready Actions</h2>
            <p className="control-label">
              The next highest-leverage actions inferred from telemetry, lineage, orchestration, and
              cluster state.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '0.75rem' }}>
            <button type="button" className="secondary-button small" onClick={() => void refresh()}>
              Refresh actions
            </button>
          </div>
          {loading && !readyActions.length ? (
            <div className="dash-loading">
              <span>⟳</span> Loading action queue…
            </div>
          ) : readyActions.length === 0 ? (
            <div className="dash-empty" style={{ padding: '2rem 1rem' }}>
              <p>No autonomous actions are ready right now.</p>
            </div>
          ) : (
            <div className="resource-list">
              {readyActions.map((action) => (
                <article key={action.id} className="resource-card">
                  <div className="model-chip-header">
                    <strong>{action.title}</strong>
                    <span style={{ color: tone(action.status) }}>{action.kind}</span>
                  </div>
                  <p>{action.detail}</p>
                  <div className="badge-row">
                    <span className="status-pill">{action.status}</span>
                    {action.source_instance_id ? (
                      <span className="status-pill">source {action.source_instance_id}</span>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>

      <div className="workspace-section-grid">
        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Campaign History</h2>
            <p className="control-label">
              Persisted campaign waves and the actions each one planned or started.
            </p>
          </div>
          {campaigns.length === 0 ? (
            <div className="dash-empty" style={{ padding: '2rem 1rem' }}>
              <p>No autonomous campaigns have been recorded yet.</p>
            </div>
          ) : (
            <div className="resource-list">
              {campaigns.map((campaign: AutonomousCampaign) => (
                <article key={campaign.campaign_id} className="resource-card">
                  <div className="model-chip-header">
                    <strong>{campaign.experiment_name}</strong>
                    <span style={{ color: tone(campaign.status) }}>{campaign.status}</span>
                  </div>
                  <p>{campaign.goal}</p>
                  <div className="badge-row">
                    <span className="status-pill">{campaign.plan.length} planned steps</span>
                    <span className="status-pill">
                      {campaign.execution.length} execution records
                    </span>
                    <span className="status-pill">updated {formatStamp(campaign.updated_at)}</span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="panel aside-section">
          <div>
            <h2 className="section-title">Lineage + Capacity</h2>
            <p className="control-label">
              Track provenance gaps alongside cluster placement hints so each wave stays auditable
              and schedulable.
            </p>
          </div>
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div className="resource-card">
              <div className="model-chip-header">
                <strong>Lineage Gaps</strong>
                <span style={{ color: tone(String(mission?.summary.lineage_gaps ?? 0)) }}>
                  {mission?.summary.lineage_gaps ?? 0}
                </span>
              </div>
              {lineageGaps.length === 0 ? (
                <p>No unresolved lineage gaps are visible.</p>
              ) : (
                <div className="badge-row">
                  {lineageGaps.slice(0, 4).map((gap, index) => (
                    <span key={`${String(gap.instance_id ?? index)}`} className="status-pill">
                      {String(gap.name ?? gap.instance_id ?? 'gap')}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div className="resource-card">
              <div className="model-chip-header">
                <strong>Placement Hints</strong>
                <span>{placements.length} node(s)</span>
              </div>
              {placements.length === 0 ? (
                <p>No placement hints are available yet.</p>
              ) : (
                <div className="resource-list">
                  {placements.map((placement, index) => (
                    <div
                      key={`${String(placement.node_id ?? index)}`}
                      className="resource-card"
                      style={{ margin: 0 }}
                    >
                      <div className="model-chip-header">
                        <strong>
                          {String(placement.node_name ?? placement.node_id ?? 'node')}
                        </strong>
                        <span style={{ color: tone(String(placement.status ?? 'idle')) }}>
                          {String(placement.status ?? 'idle')}
                        </span>
                      </div>
                      <p>{String(placement.preferred_workload ?? 'managed workloads')}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
