"use client";

import { useEffect, useState } from "react";
import {
  getSweeps,
  launchSweep,
  type AutoMLSweep,
  type AutoMLTrial,
} from "@/lib/api";

const MODELS = ["qwen-2-7b", "llama-3-8b", "mistral-7b", "phi-3-mini"];
const STRATEGIES = [
  { value: "bayesian", label: "Bayesian Optimization" },
  { value: "random", label: "Random Search" },
  { value: "grid", label: "Grid Search" },
];

export default function AutoMLPage() {
  const [sweeps, setSweeps] = useState<AutoMLSweep[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeSweep, setActiveSweep] = useState<AutoMLSweep | null>(null);

  // Launch form state
  const [sweepName, setSweepName] = useState("");
  const [baseModel, setBaseModel] = useState(MODELS[0]);
  const [strategy, setStrategy] = useState("bayesian");
  const [numTrials, setNumTrials] = useState(10);
  const [launching, setLaunching] = useState(false);

  useEffect(() => {
    getSweeps()
      .then(setSweeps)
      .catch(() => null)
      .finally(() => setLoading(false));
  }, []);

  async function handleLaunch() {
    if (!sweepName.trim() || launching) return;
    setLaunching(true);
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
    } catch (e) {
      console.error(e);
    } finally {
      setLaunching(false);
    }
  }

  function getBestTrial(sweep: AutoMLSweep): AutoMLTrial {
    return sweep.best_trial ?? sweep.trials[0];
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

      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: "1.5rem", alignItems: "start" }}>

        {/* Left: Launch Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

          <div className="panel">
            <h2 className="eval-section-title">Launch Sweep</h2>

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
                >
                  {MODELS.map((m) => (
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
                disabled={launching || !sweepName.trim()}
                onClick={() => void handleLaunch()}
                style={{ justifyContent: "center", padding: "0.85rem" }}
              >
                {launching ? "⟳ Launching..." : "▶ Launch Sweep"}
              </button>
            </div>
          </div>

          {/* Sweep history list */}
          <div className="panel">
            <h2 className="eval-section-title">All Sweeps</h2>
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
                      border: `1px solid ${activeSweep?.id === s.id ? "var(--accent)" : "var(--border)"}`,
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
            <div className="panel" style={{ borderTop: "3px solid var(--accent)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                <div>
                  <h2 style={{ margin: 0 }}>🏆 Best Trial: {getBestTrial(activeSweep).trial_id}</h2>
                  <span style={{ fontSize: "0.85rem", opacity: 0.7 }}>{activeSweep.name} · {activeSweep.strategy}</span>
                </div>
                <span style={{ fontSize: "0.8rem", color: statusColor(activeSweep.status), textTransform: "capitalize" }}>
                  ● {activeSweep.status}
                </span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" }}>
                {Object.entries(getBestTrial(activeSweep).metrics).map(([key, val]) => (
                  <div key={key} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)", borderRadius: "6px", padding: "1rem", textAlign: "center" }}>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--accent)" }}>{String(val)}</div>
                    <div style={{ fontSize: "0.8rem", opacity: 0.7, textTransform: "capitalize", marginTop: "0.25rem" }}>{key.replace("_", " ")}</div>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: "1rem", display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "0.75rem" }}>
                {Object.entries(getBestTrial(activeSweep).params).map(([key, val]) => (
                  <div key={key} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: "4px", padding: "0.6rem 0.75rem" }}>
                    <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: "0.2rem" }}>{key.replace("_", " ")}</div>
                    <div style={{ fontSize: "0.9rem", fontWeight: 600 }}>{String(val)}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Trial leaderboard */}
            <div className="panel">
              <h2 className="eval-section-title">Trial Leaderboard</h2>
              <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "1rem", fontSize: "0.85rem" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)" }}>
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
