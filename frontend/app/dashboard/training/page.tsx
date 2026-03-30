"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import {
  createManagedInstance,
  getWorkspaceOverview,
  type WorkspaceTrainingProfile,
  type WorkspaceOrchestrationTemplate,
} from "@/lib/api";
import { ROUTES } from "@/lib/routes";

type UserLevel = "beginner" | "hobbyist" | "dev";

type LaunchState = {
  step: number;
  userLevel: UserLevel;
  origin: "existing_model" | "from_scratch";
  learningMode: string;
  sourceModel: string;
  configPath: string;
  environment: "local" | "remote" | "cloud";
  instanceName: string;
  remoteHost: string;
  remoteUser: string;
  remotePort: string;
  remoteKeyPath: string;
  remoteRepoRoot: string;
  cloudProfile: string;
};

const USER_LEVELS: { id: UserLevel; label: string; desc: string; icon: string }[] = [
  {
    id: "beginner",
    label: "Beginner",
    icon: "🌱",
    desc: "Guided workflow with smart defaults. Click and run.",
  },
  {
    id: "hobbyist",
    label: "Hobbyist",
    icon: "⚙️",
    desc: "Adjustable parameters. Moderate control over training.",
  },
  {
    id: "dev",
    label: "Developer",
    icon: "🔬",
    desc: "Full control. Architecture-level customization.",
  },
];

const ENVIRONMENTS = [
  { id: "local", label: "Local Mac", desc: "Run natively on local Apple Silicon.", icon: "💻" },
  { id: "remote", label: "Remote Linux Rig", desc: "Dispatch via SSH to a dedicated GPU box.", icon: "🖥️" },
  { id: "cloud", label: "Cloud Fleet", desc: "Auto-provision EC2/Lambda orchestrator.", icon: "☁️" },
];

const LEARNING_MODES = [
  { id: "qlora", label: "QLoRA", desc: "Quantized LoRA — best for limited VRAM" },
  { id: "lora", label: "LoRA", desc: "Low-rank adaptation — fast fine-tuning" },
  { id: "full_finetune", label: "Full Finetune", desc: "All parameters updated" },
  { id: "supervised", label: "Supervised", desc: "Standard supervised learning" },
  { id: "unsupervised", label: "Unsupervised", desc: "Self-supervised pretraining" },
  { id: "rlhf", label: "RLHF", desc: "Reinforcement learning from human feedback" },
];

const DEFAULT_STATE: LaunchState = {
  step: 1,
  userLevel: "hobbyist",
  origin: "existing_model",
  learningMode: "qlora",
  sourceModel: "",
  configPath: "configs/finetune.yaml",
  environment: "local",
  instanceName: "",
  remoteHost: "",
  remoteUser: "",
  remotePort: "22",
  remoteKeyPath: "",
  remoteRepoRoot: "/tmp/ai-factory",
  cloudProfile: "",
};

export default function TrainingPage() {
  const router = useRouter();
  const [form, setForm] = useState<LaunchState>(DEFAULT_STATE);
  const [profiles, setProfiles] = useState<WorkspaceTrainingProfile[]>([]);
  const [templates, setTemplates] = useState<WorkspaceOrchestrationTemplate[]>([]);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [launched, setLaunched] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoadError(null);
    getWorkspaceOverview()
      .then((ws) => {
        if (!active) return;
        setProfiles(ws.training_profiles ?? []);
        setTemplates(
          (ws.orchestration_templates ?? []).filter(
            (t) => t.instance_type === "train" || t.instance_type === "finetune",
          ),
        );
      })
      .catch((nextError) => {
        if (!active) return;
        setLoadError(
          nextError instanceof Error
            ? nextError.message
            : "Workspace metadata could not be loaded.",
        );
      });
    return () => {
      active = false;
    };
  }, []);

  const update = (patch: Partial<LaunchState>) => setForm((f) => ({ ...f, ...patch }));

  async function launch() {
    setLaunching(true);
    setError(null);
    try {
      const isRemote = form.environment !== "local";
      const remoteHost = form.remoteHost.trim();
      const remoteProfile = form.cloudProfile.trim();
      const instance = await createManagedInstance({
        config_path: form.configPath,
        start: true,
        user_level: form.userLevel,
        lifecycle: {
          origin: form.origin,
          learning_mode: form.learningMode as never,
          source_model: form.sourceModel || null,
        },
        environment: !isRemote
          ? { kind: "local" }
          : {
              kind: "cloud",
              host: remoteHost || undefined,
              profile_name: form.environment === "cloud" ? remoteProfile || undefined : undefined,
              user: form.remoteUser.trim() || undefined,
              port: Number(form.remotePort) || 22,
              key_path: form.remoteKeyPath.trim() || undefined,
              remote_repo_root: form.remoteRepoRoot.trim() || undefined,
            },
        name: form.instanceName || undefined,
      });
      setLaunched(instance.id);
      router.push(`/runs/${instance.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Launch failed");
    } finally {
      setLaunching(false);
    }
  }

  const needsRemoteDetails = form.environment !== "local";
  const canLaunch =
    !launching &&
    form.configPath.trim().length > 0 &&
    (!needsRemoteDetails ||
      form.remoteHost.trim().length > 0 ||
      (form.environment === "cloud" && form.cloudProfile.trim().length > 0));

  return (
    <div className="dashboard-content">
      {/* Header */}
      <div className="dash-page-header panel">
        <div className="dash-page-header-inner">
          <div>
            <span className="eyebrow">Lifecycle → Train</span>
            <h1 className="dash-page-title">Training Launcher</h1>
            <p className="dash-page-desc">
              Start a new training run from scratch or continue from an existing model.
              Choose your experience level to get the right amount of control.
            </p>
          </div>
          <Link href={ROUTES.monitoring} className="secondary-button">
            View Running Instances
          </Link>
        </div>
      </div>

      {error && (
        <div className="dash-error-banner panel">
          <span className="dash-error-icon">⚠</span>
          <span>{error}</span>
        </div>
      )}

      {loadError && (
        <div className="dash-error-banner panel">
          <span className="dash-error-icon">⚠</span>
          <span>{loadError}</span>
        </div>
      )}

      <div className="training-launch-grid">
        {/* Step 1: User Level */}
        <div className="training-step panel">
          <div className="step-header">
            <span className="step-number">01</span>
            <h2>Experience Level</h2>
          </div>
          <div className="user-level-grid">
            {USER_LEVELS.map((level) => (
              <button
                key={level.id}
                type="button"
                className={`level-card ${form.userLevel === level.id ? "active" : ""}`}
                onClick={() => update({ userLevel: level.id })}
              >
                <span className="level-icon">{level.icon}</span>
                <span className="level-label">{level.label}</span>
                <span className="level-desc">{level.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Step 2: Training Origin */}
        <div className="training-step panel">
          <div className="step-header">
            <span className="step-number">02</span>
            <h2>Starting Point</h2>
          </div>
          <div className="origin-grid">
            <button
              type="button"
              className={`origin-card ${form.origin === "existing_model" ? "active" : ""}`}
              onClick={() => update({ origin: "existing_model", configPath: "configs/finetune.yaml" })}
            >
              <span className="origin-icon">⬡</span>
              <span className="origin-title">Existing Model</span>
              <span className="origin-desc">Continue from a pretrained checkpoint or HuggingFace model</span>
            </button>
            <button
              type="button"
              className={`origin-card ${form.origin === "from_scratch" ? "active" : ""}`}
              onClick={() => update({ origin: "from_scratch", configPath: "configs/train.yaml" })}
            >
              <span className="origin-icon">◈</span>
              <span className="origin-title">From Scratch</span>
              <span className="origin-desc">Train a custom architecture with fully configurable parameters</span>
            </button>
          </div>

          {form.origin === "existing_model" && (
            <div className="input-group">
              <label className="control-label" htmlFor="source-model">
                Source Model (HuggingFace ID or local path)
              </label>
              <input
                id="source-model"
                type="text"
                placeholder="e.g. Qwen/Qwen2.5-Math-1.5B-Instruct"
                value={form.sourceModel}
                onChange={(e) => update({ sourceModel: e.target.value })}
              />
            </div>
          )}
        </div>

        {/* Step 3: Learning Mode */}
        {(form.userLevel === "hobbyist" || form.userLevel === "dev") && (
          <div className="training-step panel">
            <div className="step-header">
              <span className="step-number">03</span>
              <h2>Training Method</h2>
            </div>
            <div className="learning-mode-grid">
              {LEARNING_MODES.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  className={`mode-card ${form.learningMode === mode.id ? "active" : ""}`}
                  onClick={() => update({ learningMode: mode.id })}
                >
                  <span className="mode-label">{mode.label}</span>
                  <span className="mode-desc">{mode.desc}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Step 4: Environment */}
        <div className="training-step panel">
          <div className="step-header">
            <span className="step-number">{form.userLevel === "beginner" ? "03" : "04"}</span>
            <h2>Compute Environment</h2>
          </div>
          <div className="learning-mode-grid">
            {ENVIRONMENTS.map((env) => (
              <button
                key={env.id}
                type="button"
                className={`mode-card ${form.environment === env.id ? "active" : ""}`}
                onClick={() => update({ environment: env.id as LaunchState["environment"] })}
                style={{ padding: "1rem" }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.25rem" }}>
                  <span style={{ fontSize: "1.2rem" }}>{env.icon}</span>
                  <span className="mode-label" style={{ margin: 0 }}>{env.label}</span>
                </div>
                <span className="mode-desc">{env.desc}</span>
              </button>
            ))}
          </div>
          {needsRemoteDetails && (
            <div className="training-remote-grid">
              <div className="input-group">
                <label className="control-label" htmlFor="remote-host">
                  {form.environment === "cloud" ? "Cloud host or profile" : "SSH host"}
                </label>
                <input
                  id="remote-host"
                  type="text"
                  placeholder="gpu-node.local or 10.0.0.24"
                  value={form.remoteHost}
                  onChange={(e) => update({ remoteHost: e.target.value })}
                />
              </div>
              <div className="input-group">
                <label className="control-label" htmlFor="remote-user">
                  SSH user
                </label>
                <input
                  id="remote-user"
                  type="text"
                  placeholder="ubuntu"
                  value={form.remoteUser}
                  onChange={(e) => update({ remoteUser: e.target.value })}
                />
              </div>
              <div className="input-group">
                <label className="control-label" htmlFor="remote-port">
                  SSH port
                </label>
                <input
                  id="remote-port"
                  type="number"
                  min={1}
                  value={form.remotePort}
                  onChange={(e) => update({ remotePort: e.target.value })}
                />
              </div>
              <div className="input-group">
                <label className="control-label" htmlFor="remote-key-path">
                  SSH key path
                </label>
                <input
                  id="remote-key-path"
                  type="text"
                  placeholder="~/.ssh/id_ed25519"
                  value={form.remoteKeyPath}
                  onChange={(e) => update({ remoteKeyPath: e.target.value })}
                />
              </div>
              <div className="input-group control-form-span-2">
                <label className="control-label" htmlFor="remote-repo-root">
                  Remote repo root
                </label>
                <input
                  id="remote-repo-root"
                  type="text"
                  placeholder="/tmp/ai-factory"
                  value={form.remoteRepoRoot}
                  onChange={(e) => update({ remoteRepoRoot: e.target.value })}
                />
              </div>
              {form.environment === "cloud" && (
                <div className="input-group control-form-span-2">
                  <label className="control-label" htmlFor="cloud-profile">
                    Cloud profile name
                  </label>
                  <input
                    id="cloud-profile"
                    type="text"
                    placeholder="default"
                    value={form.cloudProfile}
                    onChange={(e) => update({ cloudProfile: e.target.value })}
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Step 5: Configuration */}
        <div className="training-step panel">
          <div className="step-header">
            <span className="step-number">{form.userLevel === "beginner" ? "04" : "05"}</span>
            <h2>Configuration Details</h2>
          </div>

          {profiles.length > 0 && (
            <div className="profile-list">
              <p className="control-label">Quick profiles</p>
              {profiles.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className={`profile-item ${form.configPath === p.path ? "active" : ""}`}
                  onClick={() => update({ configPath: p.path })}
                >
                  <span className="profile-title">{p.title}</span>
                  <code className="profile-cmd">{p.train_command}</code>
                </button>
              ))}
            </div>
          )}

          <div className="input-group">
            <label className="control-label" htmlFor="config-path">Config file path</label>
            <input
              id="config-path"
              type="text"
              value={form.configPath}
              onChange={(e) => update({ configPath: e.target.value })}
            />
          </div>

          <div className="control-row">
            <div className="input-group">
              <label className="control-label" htmlFor="instance-name">Instance name (optional)</label>
              <input
                id="instance-name"
                type="text"
                placeholder="Auto-generated if empty"
                value={form.instanceName}
                onChange={(e) => update({ instanceName: e.target.value })}
              />
            </div>
            <div className="input-group">
            </div>
          </div>
        </div>

        {/* Templates */}
        {templates.length > 0 && (
          <div className="training-step panel">
            <div className="step-header">
              <span className="step-number">⚡</span>
              <h2>Quick Templates</h2>
            </div>
            <div className="template-list">
              {templates.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  className={`template-item ${form.configPath === t.path ? "active" : ""}`}
                  onClick={() => update({ configPath: t.path })}
                >
                  <div className="template-header">
                    <span className="template-title">{t.title}</span>
                    <span className="template-type">{t.instance_type}</span>
                  </div>
                  <code className="profile-cmd">{t.command}</code>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Launch */}
        <div className="launch-panel panel">
          <h2 className="launch-title">Ready to Launch</h2>
          <div className="launch-summary">
            <div className="launch-summary-row">
              <span>Level</span><strong>{form.userLevel}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Origin</span><strong>{form.origin.replace("_", " ")}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Method</span><strong>{form.learningMode}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Config</span><strong>{form.configPath}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Environment</span><strong>{form.environment}</strong>
            </div>
            {form.sourceModel && (
              <div className="launch-summary-row">
                <span>Model</span><strong>{form.sourceModel}</strong>
              </div>
            )}
          </div>
          <button
            type="button"
            className="primary-button launch-btn"
            disabled={!canLaunch}
            onClick={() => void launch()}
          >
            {launching ? "⟳ Launching…" : "▲ Launch Training Instance"}
          </button>
          {needsRemoteDetails && !form.remoteHost.trim() && (
            <p className="launch-hint" style={{ color: "var(--danger)" }}>
              Remote and cloud launches need at least a host or profile name.
            </p>
          )}
          <p className="launch-hint">
            The instance will be tracked in real-time on the monitoring page.
          </p>
        </div>
      </div>
    </div>
  );
}
