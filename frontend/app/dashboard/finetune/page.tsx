"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { createManagedInstance, getInstances, type InstanceSummary } from "@/lib/api";
import { useRouter } from "next/navigation";

type FinetuneMode = "qlora" | "lora" | "full_finetune";

const FINETUNE_MODES: { id: FinetuneMode; label: string; desc: string; vram: string; speed: string }[] = [
  {
    id: "qlora",
    label: "QLoRA",
    desc: "Quantized 4-bit adaptation. Best for limited VRAM. Memory-efficient and fast.",
    vram: "6–16 GB",
    speed: "Fast",
  },
  {
    id: "lora",
    label: "LoRA",
    desc: "Low-rank adaptation. Trains a fraction of parameters. Great balance of speed and quality.",
    vram: "16–40 GB",
    speed: "Moderate",
  },
  {
    id: "full_finetune",
    label: "Full Finetune",
    desc: "All parameters updated. Maximum flexibility and quality, requires the most resources.",
    vram: "40–80+ GB",
    speed: "Slow",
  },
];

export default function FinetunePage() {
  const router = useRouter();
  const [sources, setSources] = useState<InstanceSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [mode, setMode] = useState<FinetuneMode>("qlora");
  const [sourceModel, setSourceModel] = useState("");
  const [configPath, setConfigPath] = useState("configs/finetune.yaml");
  const [instanceName, setInstanceName] = useState("");
  const [iterations, setIterations] = useState(1);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoadError(null);
    getInstances()
      .then((list) => {
        const completed = list.filter(
          (i) =>
            i.status === "completed" &&
            (i.type === "train" || i.type === "evaluate" || i.type === "finetune"),
        );
        setSources(completed);
        if (completed.length > 0) {
          setSelectedSource(completed[0].id);
          setSourceModel(completed[0].lifecycle?.source_model ?? "");
        }
      })
      .catch((nextError) =>
        setLoadError(
          nextError instanceof Error
            ? nextError.message
            : "Completed instances could not be loaded.",
        ),
      )
      .finally(() => setLoading(false));
  }, []);

  async function launchFinetune() {
    setLaunching(true);
    setError(null);
    try {
      const instance = await createManagedInstance({
        config_path: configPath,
        start: true,
        user_level: "hobbyist",
        parent_instance_id: selectedSource ?? undefined,
        lifecycle: {
          origin: "existing_model",
          learning_mode: mode,
          source_model: sourceModel || undefined,
        },
        name: instanceName || undefined,
      });
      router.push(`/runs/${instance.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Launch failed");
    } finally {
      setLaunching(false);
    }
  }

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">Lifecycle → Finetune</span>
          <h1 className="dash-page-title">Finetuning</h1>
          <p className="dash-page-desc">
            Refine an existing model with LoRA, QLoRA, or full finetuning.
            Supports multiple iterations and version tracking.
          </p>
        </div>
      </div>

      {error && (
        <div className="dash-error-banner panel">
          <span>⚠</span> {error}
        </div>
      )}

      {loadError && (
        <div className="dash-error-banner panel">
          <span>⚠</span> {loadError}
        </div>
      )}

      <div className="finetune-grid">
        {/* Mode Selection */}
        <div className="panel finetune-section">
          <h2 className="section-title">Finetuning Method</h2>
          <div className="finetune-mode-grid">
            {FINETUNE_MODES.map((m) => (
              <button
                key={m.id}
                type="button"
                className={`finetune-mode-card ${mode === m.id ? "active" : ""}`}
                onClick={() => setMode(m.id)}
              >
                <div className="finetune-mode-header">
                  <span className="finetune-mode-label">{m.label}</span>
                  <div className="finetune-mode-specs">
                    <span className="finetune-spec-chip">{m.vram}</span>
                    <span className="finetune-spec-chip">{m.speed}</span>
                  </div>
                </div>
                <span className="finetune-mode-desc">{m.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Source Model */}
        <div className="panel finetune-section">
          <h2 className="section-title">Source Model</h2>

          {sources.length > 0 && (
            <>
              <p className="control-label">From completed instances</p>
              <div className="source-list">
                {sources.slice(0, 5).map((inst) => (
                  <button
                    key={inst.id}
                    type="button"
                    className={`source-item ${selectedSource === inst.id ? "active" : ""}`}
                    onClick={() => {
                      setSelectedSource(inst.id);
                      setSourceModel(inst.lifecycle?.source_model ?? "");
                    }}
                  >
                    <span className="source-name">{inst.name}</span>
                    <span className="source-type">{inst.type}</span>
                  </button>
                ))}
              </div>
            </>
          )}

          <div className="input-group">
            <label className="control-label" htmlFor="ft-source-model">
              Or enter model ID / path directly
            </label>
            <input
              id="ft-source-model"
              type="text"
              placeholder="e.g. Qwen/Qwen2.5-Math-1.5B-Instruct"
              value={sourceModel}
              onChange={(e) => setSourceModel(e.target.value)}
            />
          </div>
        </div>

        {/* Config */}
        <div className="panel finetune-section">
          <h2 className="section-title">Configuration</h2>
          <div className="control-row">
            <div className="input-group">
              <label className="control-label" htmlFor="ft-config">Config path</label>
              <input
                id="ft-config"
                type="text"
                value={configPath}
                onChange={(e) => setConfigPath(e.target.value)}
              />
            </div>
            <div className="input-group">
              <label className="control-label" htmlFor="ft-name">Instance name (optional)</label>
              <input
                id="ft-name"
                type="text"
                placeholder="Auto-generated"
                value={instanceName}
                onChange={(e) => setInstanceName(e.target.value)}
              />
            </div>
          </div>
          <div className="input-group">
            <label className="control-label" htmlFor="ft-iterations">
              Planned iterations: {iterations}
            </label>
            <input
              id="ft-iterations"
              type="range"
              min={1}
              max={5}
              value={iterations}
              onChange={(e) => setIterations(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Launch */}
        <div className="launch-panel panel">
          <h2 className="launch-title">Launch Finetuning</h2>
          <div className="launch-summary">
            <div className="launch-summary-row">
              <span>Method</span><strong>{mode}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Source</span>
              <strong>{sourceModel || selectedSource || "not set"}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Config</span><strong>{configPath}</strong>
            </div>
            <div className="launch-summary-row">
              <span>Iterations</span><strong>{iterations}</strong>
            </div>
          </div>
          <button
            type="button"
            className="primary-button launch-btn"
            disabled={launching || (!sourceModel && !selectedSource)}
            onClick={() => void launchFinetune()}
          >
            {launching ? "⟳ Launching…" : "⟳ Launch Finetuning"}
          </button>
          {!sourceModel && !selectedSource && (
            <p className="launch-hint" style={{ color: "var(--danger)" }}>
              Select a source model or completed instance to continue.
            </p>
          )}
          {selectedSource && (
            <p className="launch-hint">
              Chaining from a prior finetune keeps lineage intact and preserves the parent branch.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
