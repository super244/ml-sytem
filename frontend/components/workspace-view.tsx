"use client";

import clsx from "clsx";
import Link from "next/link";
import { useEffect, useState } from "react";

import {
  getWorkspaceOverview,
  type WorkspaceCheck,
  type WorkspaceEvaluationConfig,
  type WorkspaceOverview,
  type WorkspaceRecipe,
  type WorkspaceTrainingProfile,
} from "@/lib/api";
import { formatCount } from "@/lib/formatting";

import { AppShell } from "@/components/layout/app-shell";
import { MetricBadge } from "@/components/panels/metric-badge";
import { PageHeader } from "@/components/ui/page-header";
import { StatePanel } from "@/components/ui/state-panel";

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
      <button className="secondary-button small" type="button" onClick={() => onCopy(command, command)}>
        {copied ? "Copied" : `Copy ${label.toLowerCase()}`}
      </button>
    </div>
  );
}

function ReadinessCard({ check }: { check: WorkspaceCheck }) {
  return (
    <article className={clsx("workspace-card", "workspace-check", { ok: check.ok, bad: !check.ok })}>
      <div className="workspace-check-header">
        <span className={clsx("workspace-dot", { ok: check.ok, bad: !check.ok })} />
        <strong>{check.label}</strong>
      </div>
      <p className="hero-copy">{check.detail}</p>
      <MetricBadge
        label="State"
        value={check.ok ? "ready" : "needs setup"}
        tone={check.ok ? "accent" : "secondary"}
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
        setLoadError(nextError instanceof Error ? nextError.message : "Failed to load the command center.");
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
      setNotice("Clipboard access failed. You can still copy the commands manually.");
    }
  }

  const summary = overview?.summary;
  const recipes = overview?.command_recipes ?? [];
  const readinessChecks = overview?.readiness_checks ?? [];
  const trainingProfiles = overview?.training_profiles ?? [];
  const evaluationConfigs = overview?.evaluation_configs ?? [];

  return (
    <AppShell>
      <section className="route-stack">
        <PageHeader
          eyebrow="Workspace Command Center"
          title="Operate the AI factory from one place"
          description="Inspect local readiness, discover training and evaluation entry points, and copy the exact commands for setup, serving, dry-runs, and benchmark loops."
          metrics={[
            {
              label: "Ready checks",
              value: summary ? `${summary.ready_checks}/${summary.total_checks}` : "n/a",
            },
            {
              label: "Profiles",
              value: formatCount(summary?.training_profiles),
              tone: "secondary",
            },
            {
              label: "Eval configs",
              value: formatCount(summary?.evaluation_configs),
              tone: "accent",
            },
            {
              label: "Runs",
              value: formatCount(summary?.runs),
            },
          ]}
          actions={
            <>
              <Link className="ghost-button small" href="/runs">
                Inspect runs
              </Link>
              <Link className="primary-button small" href="/">
                Open solve workspace
              </Link>
            </>
          }
        />

        {loading ? (
          <StatePanel
            eyebrow="Loading"
            title="Workspace overview is loading."
            description="Atlas is discovering configs, artifacts, and readiness checks."
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
                    <MetricBadge label="Packs" value={formatCount(summary?.packs)} tone="secondary" />
                    <MetricBadge label="Benchmarks" value={formatCount(summary?.benchmarks)} tone="accent" />
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
                        <span className="status-pill">{copied[recipe.id] ? "Copied" : "Ready"}</span>
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
                          {copied[recipe.id] ? "Copied" : "Copy command"}
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
