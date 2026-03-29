"use client";

import { useState } from "react";

import {
  createManagedInstance,
  getWorkspaceOverview,
  type WorkspaceTrainingProfile,
  type WorkspaceOrchestrationTemplate,
} from "@/lib/api";
import { ROUTES } from "@/lib/routes";
import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

type UserLevel = "beginner" | "hobbyist" | "dev";

type LaunchState = {
  step: number;
  userLevel: UserLevel;
  origin: "existing_model" | "from_scratch";
  learningMode: string;
  sourceModel: string;
  configPath: string;
  environment: "local" | "cloud";
  instanceName: string;
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
};

export default function TrainingPage() {
  const router = useRouter();
  const [form, setForm] = useState<LaunchState>(DEFAULT_STATE);
  const [profiles, setProfiles] = useState<WorkspaceTrainingProfile[]>([]);
  const [templates, setTemplates] = useState<WorkspaceOrchestrationTemplate[]>([]);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [launched, setLaunched] = useState<string | null>(null);

  useEffect(() => {
    getWorkspaceOverview().then((ws) => {
      setProfiles(ws.training_profiles ?? []);
      setTemplates(
        (ws.orchestration_templates ?? []).filter(
          (t) => t.instance_type === "train" || t.instance_type === "finetune",
        ),
      );
    }).catch(() => null);
  }, []);

  const update = (patch: Partial<LaunchState>) => setForm((f) => ({ ...f, ...patch }));

  async function launch() {
    setLaunching(true);
    setError(null);
    try {
      const instance = await createManagedInstance({
        config_path: form.configPath,
        start: true,
        user_level: form.userLevel,
        lifecycle: {
          origin: form.origin,
          learning_mode: form.learningMode as never,
          source_model: form.sourceModel || null,
        },
        environment: form.environment === "cloud" ? { kind: "cloud" } : { kind: "local" },
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

        {/* Step 4: Config / Profile */}
        <div className="training-step panel">
          <div className="step-header">
            <span className="step-number">{form.userLevel === "beginner" ? "03" : "04"}</span>
            <h2>Configuration</h2>
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
              <label className="control-label" htmlFor="environment">Environment</label>
              <select
                id="environment"
                value={form.environment}
                onChange={(e) => update({ environment: e.target.value as "local" | "cloud" })}
              >
                <option value="local">Local</option>
                <option value="cloud">Cloud / SSH</option>
              </select>
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
            disabled={launching}
            onClick={() => void launch()}
          >
            {launching ? "⟳ Launching…" : "▲ Launch Training Instance"}
          </button>
          <p className="launch-hint">
            The instance will be tracked in real-time on the monitoring page.
          </p>
        </div>
      </div>
    </div>
  );
}
