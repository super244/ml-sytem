"use client";

import { useEffect, useState } from "react";
import {
  getTelemetryBacklog,
  synthesizeDataset,
  type TelemetryRecord,
} from "@/lib/api";

export default function DatasetsPage() {
  const [telemetry, setTelemetry] = useState<TelemetryRecord[]>([]);
  const [loading, setLoading] = useState(true);

  // Synthesizer State
  const [seedPrompt, setSeedPrompt] = useState("");
  const [numVariants, setNumVariants] = useState(10);
  const [modelVariant, setModelVariant] = useState("gpt-4o");
  const [synthesizing, setSynthesizing] = useState(false);
  const [synthJob, setSynthJob] = useState<{ id: string; time: number } | null>(null);

  useEffect(() => {
    getTelemetryBacklog()
      .then(setTelemetry)
      .catch(() => null)
      .finally(() => setLoading(false));
  }, []);

  async function handleSynthesize() {
    if (!seedPrompt.trim() || synthesizing) return;
    setSynthesizing(true);
    setSynthJob(null);
    try {
      const res = await synthesizeDataset({
        seed_prompt: seedPrompt,
        num_variants: numVariants,
        model_variant: modelVariant,
      });
      setSynthJob({ id: res.job_id, time: res.estimated_time_s });
      setSeedPrompt("");
    } catch (e) {
      console.error(e);
    } finally {
      setSynthesizing(false);
    }
  }

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">V2 Lab → Datasets</span>
          <h1 className="dash-page-title">Dataset Curator</h1>
          <p className="dash-page-desc">
            Harvest failed model inferences from Telemetry or synthesize entirely new JSONL variants using autonomous pipelines.
          </p>
        </div>
      </div>

      <div className="workspace-section-grid">
        
        {/* Left Pane: Telemetry Queue */}
        <div className="panel aside-section">
          <div>
            <h2 className="section-title">Telemetry Queue</h2>
            <p className="control-label">Flagged failure cases ready for dataset inclusion.</p>
          </div>

          {loading ? (
            <div className="dash-loading"><span>⟳</span> Loading records...</div>
          ) : telemetry.length === 0 ? (
            <div className="dash-empty" style={{ padding: "3rem 1rem" }}>
              <span className="dash-empty-icon" style={{ fontSize: "2rem", opacity: 0.5 }}>✓</span>
              <p style={{ marginTop: "1rem" }}>No flagged records found.</p>
            </div>
          ) : (
            <div className="control-rail" style={{ padding: 0, paddingRight: "0.5rem" }}>
              {telemetry.map((t, idx) => (
                <div key={idx} className="resource-card">
                  <div className="model-chip-header" style={{ marginBottom: "0.5rem" }}>
                    <span style={{ fontSize: "0.75rem", opacity: 0.7 }}>Model: {t.model_variant}</span>
                    <span style={{ fontSize: "0.75rem", opacity: 0.7 }}>{new Date(t.timestamp * 1000).toLocaleString()}</span>
                  </div>
                  
                  <div style={{ marginBottom: "1rem" }}>
                    <strong style={{ fontSize: "0.85rem", color: "var(--accent)" }}>Prompt:</strong>
                    <div style={{ fontSize: "0.85rem", opacity: 0.9, marginTop: "0.25rem", whiteSpace: "pre-wrap" }}>{t.prompt}</div>
                  </div>
                  
                  <div style={{ marginBottom: "1rem", opacity: 0.7 }}>
                    <strong style={{ fontSize: "0.85rem" }}>Failed Output:</strong>
                    <div style={{ fontSize: "0.85rem", marginTop: "0.25rem", whiteSpace: "pre-wrap" }}>{t.assistant_output}</div>
                  </div>
                  
                  <div className="state-panel" style={{ padding: "0.75rem", border: "1px solid var(--accent)", background: "rgba(15, 122, 97, 0.1)" }}>
                    <strong style={{ fontSize: "0.85rem", color: "var(--accent)" }}>Expected Correction:</strong>
                    <div style={{ fontSize: "0.85rem", marginTop: "0.25rem", whiteSpace: "pre-wrap" }}>{t.expected_output}</div>
                  </div>
                  
                  <div className="control-chip-group" style={{ marginTop: "1rem" }}>
                    <button className="ghost-button small">Discard</button>
                    <button className="primary-button small">Promote to Dataset</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Pane: Synthesizer */}
        <div className="panel aside-section">
          <div>
            <h2 className="section-title">Data Synthesizer (Auto-Prompt)</h2>
            <p className="control-label">Use an LLM to automatically generate diverse JSONL variations.</p>
          </div>

          <div className="input-group">
            <label className="control-label" htmlFor="model-select">Synthesizer Model</label>
            <select
              id="model-select"
              value={modelVariant}
              onChange={(e) => setModelVariant(e.target.value)}
              style={{ width: "100%" }}
            >
              <option value="gpt-4o">GPT-4o (High Quality)</option>
              <option value="claude-3-sonnet">Claude 3.5 Sonnet</option>
              <option value="qwen-2">Qwen-2 72B Instruct</option>
            </select>
          </div>

          <div className="input-group">
            <label className="control-label" htmlFor="variants-range">Dataset Size (Rows): {numVariants}</label>
            <input
              id="variants-range"
              type="range"
              min={10}
              max={10000}
              step={10}
              value={numVariants}
              onChange={(e) => setNumVariants(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>

          <div className="input-group" style={{ flex: 1 }}>
            <label className="control-label" htmlFor="seed-prompt">Seed Prompt Details</label>
            <textarea
              id="seed-prompt"
              value={seedPrompt}
              onChange={(e) => setSeedPrompt(e.target.value)}
              placeholder="e.g. 'Generate 50 math reasoning puzzles focusing on topology...'"
              style={{ width: "100%", height: "150px", resize: "vertical" }}
            />
          </div>

          {synthJob ? (
            <div className="state-panel" style={{ border: "1px solid var(--accent)", background: "rgba(15, 122, 97, 0.1)", borderRadius: "var(--radius-md)" }}>
              <div className="model-chip-header" style={{ marginBottom: "0.5rem" }}>
                <div style={{display: "flex", alignItems: "center", gap: "0.75rem"}}>
                  <span className="workspace-dot ok" />
                  <strong style={{ color: "var(--accent)" }}>Synthesis Job Accepted</strong>
                </div>
              </div>
              <p style={{ fontSize: "0.85rem", opacity: 0.8, margin: 0 }}>Job ID: <code>{synthJob.id}</code></p>
              <p style={{ fontSize: "0.85rem", opacity: 0.8, margin: 0 }}>Estimated completion: {synthJob.time.toFixed(1)}s</p>
            </div>
          ) : (
            <button
              className="primary-button"
              disabled={synthesizing || !seedPrompt.trim()}
              onClick={() => void handleSynthesize()}
              style={{ padding: "1rem", fontSize: "1rem", justifyContent: "center" }}
            >
              {synthesizing ? "⟳ Dispatching to Cluster..." : "▶ Start Synthesis Loop"}
            </button>
          )}

        </div>

      </div>
    </div>
  );
}
