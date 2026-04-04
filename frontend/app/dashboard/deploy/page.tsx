'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import {
  deployManagedInstance,
  getInstances,
  type DeploymentTarget,
  type InstanceSummary,
} from '@/lib/api';
import { useRouter } from 'next/navigation';

const DEPLOY_TARGETS: {
  id: DeploymentTarget;
  label: string;
  icon: string;
  desc: string;
  color: string;
  bg: string;
}[] = [
  {
    id: 'ollama',
    label: 'Ollama',
    icon: '🦙',
    desc: 'Run locally via Ollama. Instant local inference with model management.',
    color: '#e07b39',
    bg: 'rgba(224, 123, 57, 0.08)',
  },
  {
    id: 'lmstudio',
    label: 'LM Studio',
    icon: '🎛',
    desc: 'Export for LM Studio. Desktop GUI with full model control.',
    color: '#8857c4',
    bg: 'rgba(136, 87, 196, 0.08)',
  },
  {
    id: 'huggingface',
    label: 'HuggingFace',
    icon: '🤗',
    desc: 'Push to HuggingFace Hub. Share publicly or keep private.',
    color: '#f5a623',
    bg: 'rgba(245, 166, 35, 0.08)',
  },
  {
    id: 'openai_compatible_api',
    label: 'OpenAI-compatible API',
    icon: '⚡',
    desc: 'Serve via OpenAI-compatible REST API endpoint.',
    color: 'var(--accent)',
    bg: 'rgba(15, 122, 97, 0.08)',
  },
  {
    id: 'api',
    label: 'Custom API',
    icon: '◎',
    desc: 'Deploy to a custom API backend with your own server.',
    color: 'var(--secondary)',
    bg: 'rgba(37, 95, 155, 0.08)',
  },
];

export default function DeployPage() {
  const router = useRouter();
  const [sources, setSources] = useState<InstanceSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [target, setTarget] = useState<DeploymentTarget>('ollama');
  const [configPath, setConfigPath] = useState('configs/deploy.yaml');
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [deployed, setDeployed] = useState<InstanceSummary[]>([]);

  useEffect(() => {
    getInstances()
      .then((list) => {
        const deployable = list.filter(
          (i) =>
            i.status === 'completed' &&
            (i.type === 'train' || i.type === 'finetune' || i.type === 'evaluate'),
        );
        const deploys = list.filter((i) => i.type === 'deploy');
        setSources(deployable);
        setDeployed(deploys);
        if (deployable.length > 0) setSelectedId(deployable[0].id);
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load instances'))
      .finally(() => setLoading(false));
  }, []);

  async function launch() {
    if (!selectedId) return;
    setLaunching(true);
    setError(null);
    setNotice(null);
    try {
      const instance = await deployManagedInstance(selectedId, {
        target,
        config_path: configPath,
        start: true,
      });
      setNotice(`Deployment ${instance.name} launched for ${target}.`);
      router.push(`/runs/${instance.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Deploy failed');
    } finally {
      setLaunching(false);
    }
  }

  const selectedInstance = sources.find((s) => s.id === selectedId);

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">Lifecycle → Deploy</span>
          <h1 className="dash-page-title">Deployment</h1>
          <p className="dash-page-desc">
            Deploy your trained models to HuggingFace, Ollama, LM Studio, or a custom API endpoint.
            Seamless transition from model to production system.
          </p>
        </div>
      </div>

      {error && <div className="dash-error-banner panel">⚠ {error}</div>}

      {notice && <div className="dash-note-banner panel">◎ {notice}</div>}

      <div className="deploy-grid">
        {/* Target Selection */}
        <div className="panel deploy-section">
          <h2 className="section-title">Deployment Target</h2>
          <div className="deploy-target-grid">
            {DEPLOY_TARGETS.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`deploy-target-card ${target === t.id ? 'active' : ''}`}
                style={target === t.id ? { background: t.bg, borderColor: t.color } : undefined}
                onClick={() => setTarget(t.id)}
              >
                <span className="deploy-target-icon">{t.icon}</span>
                <span
                  className="deploy-target-label"
                  style={target === t.id ? { color: t.color } : undefined}
                >
                  {t.label}
                </span>
                <span className="deploy-target-desc">{t.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Source Selection */}
        <div className="panel deploy-section">
          <h2 className="section-title">Source Instance</h2>
          {loading && <p className="control-label">Loading…</p>}
          {!loading && sources.length === 0 && (
            <p className="control-label">No completed training or finetuning runs found.</p>
          )}
          <div className="source-list">
            {sources.slice(0, 6).map((inst) => (
              <button
                key={inst.id}
                type="button"
                className={`source-item ${selectedId === inst.id ? 'active' : ''}`}
                onClick={() => setSelectedId(inst.id)}
              >
                <span className="source-name">{inst.name}</span>
                <span className="source-type">
                  {inst.type} · {inst.lifecycle?.learning_mode ?? '—'}
                </span>
              </button>
            ))}
          </div>

          <div className="input-group">
            <label className="control-label" htmlFor="deploy-config">
              Config path
            </label>
            <input
              id="deploy-config"
              type="text"
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
            />
          </div>
        </div>

        {/* Launch */}
        <div className="launch-panel panel">
          <h2 className="launch-title">Deploy Model</h2>
          <div className="launch-summary">
            <div className="launch-summary-row">
              <span>Target</span>
              <strong>{target}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Source</span>
              <strong>{selectedInstance?.name ?? 'not selected'}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Config</span>
              <strong>{configPath}</strong>
            </div>
          </div>
          <button
            type="button"
            className="primary-button launch-btn"
            disabled={launching || !selectedId}
            onClick={() => void launch()}
          >
            {launching ? '⟳ Deploying…' : `⬆ Deploy to ${target}`}
          </button>
        </div>

        {/* Past Deployments */}
        {deployed.length > 0 && (
          <div className="panel deploy-section">
            <h2 className="section-title">Past Deployments</h2>
            <div className="deploy-history-list">
              {deployed.slice(0, 6).map((inst) => (
                <div
                  key={inst.id}
                  className="deploy-history-item-container"
                  style={{ display: 'flex', alignItems: 'center', gap: '12px' }}
                >
                  <Link
                    href={`/runs/${inst.id}`}
                    className="deploy-history-item"
                    style={{ flex: 1 }}
                  >
                    <div className="deploy-history-header">
                      <span className="deploy-history-name">{inst.name}</span>
                      <span className={`deploy-history-status status-${inst.status}`}>
                        {inst.status}
                      </span>
                    </div>
                    <span className="deploy-history-meta">
                      {inst.lifecycle?.deployment_targets?.join(', ') ?? '—'} ·{' '}
                      {inst.updated_at ? new Date(inst.updated_at).toLocaleDateString() : '—'}
                    </span>
                  </Link>
                  {inst.status === 'completed' && (
                    <Link href="/dashboard/inference" className="secondary-button small">
                      ◎ Sandbox
                    </Link>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
