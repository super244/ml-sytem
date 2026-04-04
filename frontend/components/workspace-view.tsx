'use client';

import clsx from 'clsx';
import Link from 'next/link';
import { useEffect, useState } from 'react';

import {
  type WorkspaceExtensionPoint,
  type WorkspaceExperienceTier,
  type WorkspaceCapability,
  type WorkspaceInterfaceSurface,
  getWorkspaceOverview,
  type WorkspaceCheck,
  type WorkspaceEvaluationConfig,
  type WorkspaceOrchestrationTemplate,
  type WorkspaceOverview,
  type WorkspaceRecipe,
  type WorkspaceTrainingProfile,
} from '@/lib/api';
import { formatCount } from '@/lib/formatting';
import { ROUTES } from '@/lib/routes';

import { AppShell } from '@/components/layout/app-shell';
import { MetricBadge } from '@/components/panels/metric-badge';
import { PageHeader } from '@/components/ui/page-header';
import { StatePanel } from '@/components/ui/state-panel';

type CopyMap = Record<string, boolean>;

function CommandBlock({
  label,
  command,
  copied,
  onCopy,
}: {
  label: string;
  command: string;
  copied: boolean;
  onCopy: (key: string, command: string) => void;
}) {
  return (
    <div className="workspace-command-stack">
      <strong>{label}</strong>
      <pre className="workspace-command">
        <code>{command}</code>
      </pre>
      <button
        className="secondary-button small"
        type="button"
        onClick={() => onCopy(command, command)}
      >
        {copied ? 'Copied' : `Copy ${label.toLowerCase()}`}
      </button>
    </div>
  );
}

function ReadinessCard({ check }: { check: WorkspaceCheck }) {
  return (
    <article
      className={clsx('workspace-card', 'workspace-check', { ok: check.ok, bad: !check.ok })}
    >
      <div className="workspace-check-header">
        <span className={clsx('workspace-dot', { ok: check.ok, bad: !check.ok })} />
        <strong>{check.label}</strong>
      </div>
      <p className="hero-copy">{check.detail}</p>
      <MetricBadge
        label="State"
        value={check.ok ? 'ready' : 'needs setup'}
        tone={check.ok ? 'accent' : 'secondary'}
      />
    </article>
  );
}

export function WorkspaceView() {
  const [overview, setOverview] = useState<WorkspaceOverview | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState<CopyMap>({});

  useEffect(() => {
    let active = true;
    async function load() {
      setLoading(true);
      setLoadError(null);
      try {
        const payload = await getWorkspaceOverview();
        if (!active) {
          return;
        }
        setOverview(payload);
      } catch (nextError) {
        if (!active) {
          return;
        }
        setLoadError(
          nextError instanceof Error ? nextError.message : 'Failed to load the command center.',
        );
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, []);

  async function copyCommand(key: string, command: string) {
    try {
      await navigator.clipboard.writeText(command);
      setNotice(null);
      setCopied((current) => ({ ...current, [key]: true }));
      window.setTimeout(() => {
        setCopied((current) => ({ ...current, [key]: false }));
      }, 1600);
    } catch {
      setNotice('Clipboard access failed. You can still copy the commands manually.');
    }
  }

  const summary = overview?.summary;
  const recipes = overview?.command_recipes ?? [];
  const orchestrationCapabilities = overview?.orchestration_capabilities ?? [];
  const orchestrationTemplates = overview?.orchestration_templates ?? [];
  const readinessChecks = overview?.readiness_checks ?? [];
  const trainingProfiles = overview?.training_profiles ?? [];
  const evaluationConfigs = overview?.evaluation_configs ?? [];
  const interfaces = overview?.interfaces ?? [];
  const experienceTiers = overview?.experience_tiers ?? [];
  const extensionPoints = overview?.extension_points ?? [];

  return (
    <AppShell>
      <section className="route-stack">
        <PageHeader
          eyebrow="Workspace Command Center"
          title="Operate the AI factory from one place"
          description="Inspect local readiness, discover training and evaluation entry points, and copy the exact commands for setup, serving, dry-runs, and benchmark loops."
          metrics={[
            {
              label: 'Ready checks',
              value: summary ? `${summary.ready_checks}/${summary.total_checks}` : 'n/a',
            },
            {
              label: 'Profiles',
              value: formatCount(summary?.training_profiles),
              tone: 'secondary',
            },
            {
              label: 'Eval configs',
              value: formatCount(summary?.evaluation_configs),
              tone: 'accent',
            },
            {
              label: 'Control templates',
              value: formatCount(summary?.orchestration_templates),
              tone: 'secondary',
            },
            {
              label: 'Runs',
              value: formatCount(summary?.runs),
            },
          ]}
          actions={
            <>
              <Link className="ghost-button small" href={ROUTES.runs}>
                Inspect runs
              </Link>
              <Link className="primary-button small" href={ROUTES.solve}>
                Open solve workspace
              </Link>
            </>
          }
        />

        {loading ? (
          <StatePanel
            eyebrow="Loading"
            title="Workspace overview is loading."
            description="AI-Factory is discovering configs, artifacts, and readiness checks."
            tone="loading"
          />
        ) : null}

        {loadError ? (
          <StatePanel
            eyebrow="Unavailable"
            title="The command center could not be loaded."
            description={loadError}
            tone="error"
          />
        ) : null}

        {overview ? (
          <>
            {notice ? (
              <StatePanel
                eyebrow="Clipboard"
                title="Copying needs manual fallback right now."
                description={notice}
                tone="error"
              />
            ) : null}

            <section className="workspace-section-grid">
              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">Readiness</div>
                    <h2 className="workspace-title">Local environment status</h2>
                  </div>
                  <div className="badge-row">
                    <MetricBadge label="Datasets" value={formatCount(summary?.datasets)} />
                    <MetricBadge
                      label="Packs"
                      value={formatCount(summary?.packs)}
                      tone="secondary"
                    />
                    <MetricBadge
                      label="Benchmarks"
                      value={formatCount(summary?.benchmarks)}
                      tone="accent"
                    />
                  </div>
                </div>
                <div className="workspace-card-grid compact">
                  {readinessChecks.map((check) => (
                    <ReadinessCard key={check.id} check={check} />
                  ))}
                </div>
              </section>

              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">Common flows</div>
                    <h2 className="workspace-title">Command recipes</h2>
                  </div>
                </div>
                <div className="workspace-card-grid">
                  {recipes.map((recipe: WorkspaceRecipe) => (
                    <article key={recipe.id} className="workspace-card">
                      <div className="message-meta">
                        <span>{recipe.category}</span>
                        <span className="status-pill">
                          {copied[recipe.id] ? 'Copied' : 'Ready'}
                        </span>
                      </div>
                      <h2>{recipe.title}</h2>
                      <p className="hero-copy">{recipe.description}</p>
                      <pre className="workspace-command">
                        <code>{recipe.command}</code>
                      </pre>
                      <div className="workspace-actions">
                        <button
                          className="secondary-button small"
                          type="button"
                          onClick={() => copyCommand(recipe.id, recipe.command)}
                        >
                          {copied[recipe.id] ? 'Copied' : 'Copy command'}
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            </section>

            <section className="workspace-section-grid">
              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">Interfaces</div>
                    <h2 className="workspace-title">Shared control surfaces</h2>
                  </div>
                </div>
                <div className="workspace-card-grid compact">
                  {interfaces.map((surface: WorkspaceInterfaceSurface) => (
                    <article key={surface.id} className="workspace-card">
                      <div className="message-meta">
                        <span>{surface.id}</span>
                        <span className="status-pill">{surface.status}</span>
                      </div>
                      <h2>{surface.label}</h2>
                      <p className="hero-copy">{surface.description}</p>
                      <div className="preview-block subtle">
                        <strong>Entrypoint</strong>
                        <p>{surface.entrypoint}</p>
                        <p>{surface.backend_contract}</p>
                      </div>
                    </article>
                  ))}
                </div>
              </section>

              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">User tiers</div>
                    <h2 className="workspace-title">Experience profiles</h2>
                  </div>
                </div>
                <div className="workspace-card-grid compact">
                  {experienceTiers.map((tier: WorkspaceExperienceTier) => (
                    <article key={tier.id} className="workspace-card">
                      <div className="message-meta">
                        <span>{tier.id}</span>
                        <span className="status-pill">Adaptive</span>
                      </div>
                      <h2>{tier.label}</h2>
                      <p className="hero-copy">{tier.description}</p>
                      <div className="preview-block subtle">
                        <strong>Visible controls</strong>
                        <p>{tier.visible_controls.join(' • ')}</p>
                        <p>Modes: {tier.recommended_modes.join(' • ')}</p>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            </section>

            <section className="workspace-section-grid">
              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">Control plane</div>
                    <h2 className="workspace-title">Orchestration capabilities</h2>
                  </div>
                </div>
                <div className="workspace-card-grid compact">
                  {orchestrationCapabilities.map((capability: WorkspaceCapability) => (
                    <article key={capability.id} className="workspace-card">
                      <div className="message-meta">
                        <span>Capability</span>
                        <span className="status-pill">Shared</span>
                      </div>
                      <h2>{capability.title}</h2>
                      <p className="hero-copy">{capability.detail}</p>
                    </article>
                  ))}
                </div>
              </section>

              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">Extension points</div>
                    <h2 className="workspace-title">Lifecycle plugins and targets</h2>
                  </div>
                </div>
                <div className="workspace-card-grid">
                  {extensionPoints.map((extension: WorkspaceExtensionPoint) => (
                    <article key={extension.id} className="workspace-card">
                      <div className="message-meta">
                        <span>{extension.kind}</span>
                        <span className="status-pill">{extension.maturity}</span>
                      </div>
                      <h2>{extension.label}</h2>
                      <p className="hero-copy">{extension.description}</p>
                      <div className="preview-block subtle">
                        <strong>Instance types</strong>
                        <p>{extension.supported_instance_types.join(' • ')}</p>
                        {extension.config_hint ? <p>Hint: {extension.config_hint}</p> : null}
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            </section>

            <section className="workspace-section-grid">
              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">Starter templates</div>
                    <h2 className="workspace-title">Managed instance entry points</h2>
                  </div>
                </div>
                <div className="workspace-card-grid">
                  {orchestrationTemplates.map((template: WorkspaceOrchestrationTemplate) => (
                    <article key={template.id} className="workspace-card">
                      <div className="message-meta">
                        <span>{template.instance_type}</span>
                        <span className="status-pill">{template.path}</span>
                      </div>
                      <h2>{template.title}</h2>
                      <div className="badge-row">
                        <MetricBadge label="User level" value={template.user_level} />
                        <MetricBadge
                          label="Mode"
                          value={template.orchestration_mode}
                          tone="secondary"
                        />
                      </div>
                      <CommandBlock
                        label="Create instance"
                        command={template.command}
                        copied={Boolean(copied[template.command])}
                        onCopy={copyCommand}
                      />
                    </article>
                  ))}
                </div>
              </section>
            </section>

            <section className="workspace-section-grid">
              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">Training</div>
                    <h2 className="workspace-title">Discovered training profiles</h2>
                  </div>
                </div>
                <div className="workspace-card-grid">
                  {trainingProfiles.map((profile: WorkspaceTrainingProfile) => (
                    <article key={profile.id} className="workspace-card">
                      <div className="message-meta">
                        <span>Config</span>
                        <span className="status-pill">{profile.path}</span>
                      </div>
                      <h2>{profile.title}</h2>
                      <CommandBlock
                        label="Dry-run"
                        command={profile.dry_run_command}
                        copied={Boolean(copied[profile.dry_run_command])}
                        onCopy={copyCommand}
                      />
                      <CommandBlock
                        label="Full train"
                        command={profile.train_command}
                        copied={Boolean(copied[profile.train_command])}
                        onCopy={copyCommand}
                      />
                    </article>
                  ))}
                </div>
              </section>

              <section className="panel workspace-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">Evaluation</div>
                    <h2 className="workspace-title">Benchmark config entry points</h2>
                  </div>
                </div>
                <div className="workspace-card-grid">
                  {evaluationConfigs.map((config: WorkspaceEvaluationConfig) => (
                    <article key={config.id} className="workspace-card">
                      <div className="message-meta">
                        <span>Config</span>
                        <span className="status-pill">{config.path}</span>
                      </div>
                      <h2>{config.title}</h2>
                      <CommandBlock
                        label="Run evaluation"
                        command={config.run_command}
                        copied={Boolean(copied[config.run_command])}
                        onCopy={copyCommand}
                      />
                    </article>
                  ))}
                </div>
              </section>
            </section>
          </>
        ) : null}
      </section>
    </AppShell>
  );
}
