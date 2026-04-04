"use client";

import { useEffect, useState } from "react";
import {
  getModels,
  getMissionControl,
  getSweepsSnapshot,
  launchSweep,
  type AutoMLSweepSnapshot,
  type AutoMLSweep,
  type AutoMLTrial,
  type ModelInfo,
  type MissionControlSnapshot,
} from "@/lib/api";
import { AutonomyLoopPanel } from "@/components/autonomy-loop-panel";
import { isDemoMode } from "@/lib/runtime-mode";

const MODELS = ["qwen-2-7b", "llama-3-8b", "mistral-7b", "phi-3-mini"];
const STRATEGIES = [
  { value: "bayesian", label: "Bayesian Optimization" },
  { value: "random", label: "Random Search" },
  { value: "grid", label: "Grid Search" },
];

export default function AutoMLPage() {
  const demoMode = isDemoMode();
  const [sweeps, setSweeps] = useState<AutoMLSweep[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [mission, setMission] = useState<MissionControlSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [writeEnabled, setWriteEnabled] = useState(false);
  const [activeSweep, setActiveSweep] = useState<AutoMLSweep | null>(null);

  // Launch form state
  const [sweepName, setSweepName] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [strategy, setStrategy] = useState("bayesian");
  const [numTrials, setNumTrials] = useState(10);
  const [launching, setLaunching] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function refresh(initialLoad: boolean = false) {
      if (initialLoad) {
        setLoading(true);
      }
      try {
        const [sweepsResponse, models, missionResponse] = await Promise.allSettled([
          getSweepsSnapshot(),
          getModels(),
          getMissionControl(),
        ]);
        if (cancelled) {
          return;
        }

        if (sweepsResponse.status === "fulfilled") {
          const payload: AutoMLSweepSnapshot = sweepsResponse.value;
          setSweeps(payload.sweeps);
          setWriteEnabled(Boolean(payload.write_enabled));
          setActiveSweep((current) => {
            if (current) {
              return payload.sweeps.find((sweep) => sweep.id === current.id) ?? current;
            }
            return payload.sweeps[0] ?? null;
          });
        } else {
          setError(sweepsResponse.reason instanceof Error ? sweepsResponse.reason.message : "Failed to load sweeps.");
        }

        if (models.status === "fulfilled") {
          setAvailableModels(models.value.filter((model) => model.available));
        } else if (!demoMode) {
          setError((current) => current ?? "Model inventory is unavailable.");
        }

        if (missionResponse.status === "fulfilled") {
          setMission(missionResponse.value);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void refresh(true);
    const interval = setInterval(() => {
      void refresh();
    }, 5000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const modelOptions = availableModels.length > 0 ? availableModels.map((model) => model.name) : demoMode ? MODELS : [];

  useEffect(() => {
    if (!baseModel || !modelOptions.includes(baseModel)) {
      setBaseModel(modelOptions[0] ?? "");
    }
  }, [baseModel, modelOptions]);

  async function handleLaunch() {
    if (!sweepName.trim() || launching || !writeEnabled || !baseModel) return;
    setLaunching(true);
    setError(null);
    try {
      const newSweep = await launchSweep({
        name: sweepName,
        base_model: baseModel,
        strategy,
        num_trials: numTrials,
      });
      setSweeps((prev) => [newSweep, ...prev]);
      setActiveSweep(newSweep);
      setSweepName("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to launch sweep.");
    } finally {
      setLaunching(false);
    }
  }

  function getBestTrial(sweep: AutoMLSweep): AutoMLTrial | null {
    return sweep.best_trial ?? sweep.trials[0] ?? null;
  }

  function statusColor(status: string) {
    if (status === "running") return "var(--accent)";
    if (status === "completed") return "#6b8";
    return "#888";
  }

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">V2 Lab → AutoML</span>
          <h1 className="dash-page-title">AutoML Studio</h1>
          <p className="dash-page-desc">
            Launch Bayesian, grid, or random hyperparameter sweeps across your models. The
            studio surfaces optimal configurations automatically so you never hand-tune again.
          </p>
        </div>
      </div>

      {error && <div className="dash-error-banner panel">⚠ {error}</div>}
      {!writeEnabled && (
        <div className="panel state-panel">
          AutoML inventory is live in production, but launching simulated sweeps is disabled outside demo mode.
        </div>
      )}
      {!demoMode && availableModels.length === 0 && (
        <div className="panel state-panel">
          No live model registry entries are available yet, so new sweep launch inputs stay disabled until inventory appears.
        </div>
      )}

      {mission?.autonomy ? (
        <AutonomyLoopPanel
          autonomy={mission.autonomy}
          title="Search Readiness"
          description="AutoML now reads the same cluster, agent, and lineage posture as the rest of the lab."
          maxStages={2}
          maxActions={2}
          compact
        />
      ) : null}

      {mission?.autonomy ? (
        <section className="panel aside-section">
          <div className="model-chip-header">
            <div>
              <h2 className="section-title">Dispatch Envelope</h2>
              <p className="control-label">
                Sweep fan-out is constrained by current cluster capacity, queue pressure, and lineage backlog.
              </p>
            </div>
            <span className="status-pill">{mission.autonomy.capacity.status}</span>
          </div>
          <div className="badge-row" style={{ marginTop: "0.75rem" }}>
            <span className="status-pill">idle nodes {mission.autonomy.capacity.idle_nodes}</span>
            <span className="status-pill">schedulable trials {mission.autonomy.capacity.schedulable_trials}</span>
            <span className="status-pill">suggested parallelism {mission.autonomy.capacity.suggested_parallelism}</span>
            <span className="status-pill">bottleneck {mission.autonomy.capacity.bottleneck}</span>
          </div>
        </section>
      ) : null}

      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: "1.5rem", alignItems: "start" }}>

        {/* Left: Launch Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

          <div className="panel aside-section">
            <h2 className="section-title">Launch Sweep</h2>

            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", marginTop: "1rem" }}>
              <div className="input-group">
                <label className="control-label" htmlFor="sweep-name">Sweep Name</label>
                <input
                  id="sweep-name"
                  type="text"
                  value={sweepName}
                  onChange={(e) => setSweepName(e.target.value)}
                  placeholder="e.g. qwen-lr-sweep-v2"
                />
              </div>

              <div className="input-group">
                <label className="control-label" htmlFor="base-model">Base Model</label>
                <select
                  id="base-model"
                  value={baseModel}
                  onChange={(e) => setBaseModel(e.target.value)}
                  disabled={modelOptions.length === 0}
                >
                  {modelOptions.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label className="control-label" htmlFor="strategy">Search Strategy</label>
                <select
                  id="strategy"
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                >
                  {STRATEGIES.map((s) => (
                    <option key={s.value} value={s.value}>{s.label}</option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label className="control-label" htmlFor="num-trials">Trials: {numTrials}</label>
                <input
                  id="num-trials"
                  type="range"
                  min={3}
                  max={50}
                  value={numTrials}
                  onChange={(e) => setNumTrials(Number(e.target.value))}
                />
              </div>

              <button
                id="launch-sweep-btn"
                className="primary-button"
                disabled={launching || !sweepName.trim() || !writeEnabled || !baseModel}
                onClick={() => void handleLaunch()}
                style={{ justifyContent: "center", padding: "0.85rem" }}
              >
                {launching ? "⟳ Launching..." : writeEnabled ? "▶ Launch Sweep" : "Read-Only Inventory"}
              </button>
            </div>
          </div>

          {/* Sweep history list */}
          <div className="panel aside-section">
            <h2 className="section-title">All Sweeps</h2>
            {loading ? (
              <div className="dash-loading" style={{ marginTop: "1rem" }}>⟳ Loading...</div>
            ) : sweeps.length === 0 ? (
              <div className="dash-empty" style={{ padding: "2rem 0" }}>No sweeps yet.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "1rem" }}>
                {sweeps.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setActiveSweep(s)}
                    style={{
                      background: activeSweep?.id === s.id ? "rgba(15,122,97,0.12)" : "rgba(255,255,255,0.03)",
                      border: `1px solid ${activeSweep?.id === s.id ? "var(--accent)" : "var(--line)"}`,
                      borderRadius: "6px",
                      padding: "0.75rem 1rem",
                      cursor: "pointer",
                      textAlign: "left",
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.25rem",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <strong style={{ fontSize: "0.9rem" }}>{s.name}</strong>
                      <span style={{ fontSize: "0.75rem", color: statusColor(s.status), textTransform: "capitalize" }}>
                        ● {s.status}
                      </span>
                    </div>
                    <span style={{ fontSize: "0.8rem", opacity: 0.6 }}>
                      {s.base_model} · {s.completed_trials}/{s.num_trials} trials
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: Results Panel */}
        {activeSweep ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

            {/* Best trial highlight */}
            <div className="resource-card" style={{ borderTop: "3px solid var(--accent)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                <div>
                  <h2 style={{ margin: 0 }}>
                    {getBestTrial(activeSweep) ? `🏆 Best Trial: ${getBestTrial(activeSweep)?.trial_id}` : "No completed trials yet"}
                  </h2>
                  <span style={{ fontSize: "0.85rem", opacity: 0.7 }}>{activeSweep.name} · {activeSweep.strategy}</span>
                </div>
                <span style={{ fontSize: "0.8rem", color: statusColor(activeSweep.status), textTransform: "capitalize" }}>
                  ● {activeSweep.status}
                </span>
              </div>

              {getBestTrial(activeSweep) ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" }}>
                  {Object.entries(getBestTrial(activeSweep)?.metrics ?? {}).map(([key, val]) => (
                    <div key={key} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--line)", borderRadius: "6px", padding: "1rem", textAlign: "center" }}>
                      <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--accent)" }}>{String(val)}</div>
                      <div style={{ fontSize: "0.8rem", opacity: 0.7, textTransform: "capitalize", marginTop: "0.25rem" }}>{key.replace("_", " ")}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="dash-empty" style={{ padding: "1.5rem 0" }}>
                  This sweep has not produced any completed trials yet.
                </div>
              )}

              {getBestTrial(activeSweep) ? (
                <div style={{ marginTop: "1rem", display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "0.75rem" }}>
                  {Object.entries(getBestTrial(activeSweep)?.params ?? {}).map(([key, val]) => (
                    <div key={key} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--line)", borderRadius: "4px", padding: "0.6rem 0.75rem" }}>
                      <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: "0.2rem" }}>{key.replace("_", " ")}</div>
                      <div style={{ fontSize: "0.9rem", fontWeight: 600 }}>{String(val)}</div>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>

            {/* Trial leaderboard */}
            <div className="panel aside-section">
              <h2 className="section-title">Trial Leaderboard</h2>
              <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "1rem", fontSize: "0.85rem" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--line)" }}>
                    {["Rank", "Trial", "Loss ↑", "Accuracy", "PPL", "LR", "BS", "LoRA Rank", "Duration"].map(col => (
                      <th key={col} style={{ padding: "0.5rem 0.75rem", textAlign: "left", opacity: 0.7, fontWeight: 600 }}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {activeSweep.trials.map((trial, idx) => (
                    <tr
                      key={trial.trial_id}
                      style={{
                        borderBottom: "1px solid rgba(255,255,255,0.05)",
                        background: idx === 0 ? "rgba(15,122,97,0.07)" : "transparent",
                      }}
                    >
                      <td style={{ padding: "0.6rem 0.75rem", fontWeight: idx === 0 ? 700 : 400, color: idx === 0 ? "var(--accent)" : "inherit" }}>
                        {idx === 0 ? "🏆 1" : idx + 1}
                      </td>
                      <td style={{ padding: "0.6rem 0.75rem", fontFamily: "monospace" }}>{trial.trial_id}</td>
                      <td style={{ padding: "0.6rem 0.75rem" }}>{trial.metrics.final_loss}</td>
                      <td style={{ padding: "0.6rem 0.75rem" }}>{(trial.metrics.accuracy * 100).toFixed(1)}%</td>
                      <td style={{ padding: "0.6rem 0.75rem" }}>{trial.metrics.perplexity}</td>
                      <td style={{ padding: "0.6rem 0.75rem", fontFamily: "monospace" }}>{trial.params.learning_rate}</td>
                      <td style={{ padding: "0.6rem 0.75rem" }}>{trial.params.batch_size}</td>
                      <td style={{ padding: "0.6rem 0.75rem" }}>r={trial.params.lora_rank}</td>
                      <td style={{ padding: "0.6rem 0.75rem", opacity: 0.7 }}>{trial.duration_s}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="panel dash-empty" style={{ minHeight: "400px", display: "flex", flexDirection: "column", justifyContent: "center", gap: "1rem" }}>
            <span style={{ fontSize: "3rem", opacity: 0.3 }}>⚗️</span>
            <h2 style={{ opacity: 0.5, marginBottom: 0 }}>No sweep selected</h2>
            <p style={{ opacity: 0.45 }}>Launch a sweep using the form, or select one from the history panel.</p>
          </div>
        )}
      </div>
    </div>
  );
}
