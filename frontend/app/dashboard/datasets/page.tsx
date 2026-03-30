"use client";

import { useEffect, useState } from "react";
import {
  discardTelemetryRecord,
  getSynthesisJob,
  getTelemetryBacklog,
  promoteTelemetryRecord,
  synthesizeDataset,
  type SynthesisJob,
  type TelemetryRecord,
} from "@/lib/api";

export default function DatasetsPage() {
  const [telemetry, setTelemetry] = useState<TelemetryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [actioningId, setActioningId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Synthesizer State
  const [seedPrompt, setSeedPrompt] = useState("");
  const [numVariants, setNumVariants] = useState(10);
  const [modelVariant, setModelVariant] = useState("gpt-4o");
  const [synthesizing, setSynthesizing] = useState(false);
  const [synthJob, setSynthJob] = useState<SynthesisJob | null>(null);

  useEffect(() => {
    getTelemetryBacklog()
      .then(setTelemetry)
      .catch((nextError) => {
        setError(nextError instanceof Error ? nextError.message : "Failed to load telemetry backlog.");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!synthJob || synthJob.status === "completed") {
      return;
    }

    let active = true;
    const interval = setInterval(() => {
      getSynthesisJob(synthJob.job_id)
        .then((job) => {
          if (active) {
            setSynthJob(job);
          }
        })
        .catch(() => null);
    }, 2500);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [synthJob]);

  async function handleTelemetryAction(recordId: string, action: "promote" | "discard") {
    setActioningId(recordId);
    setError(null);
    setNotice(null);

    try {
      const result =
        action === "promote"
          ? await promoteTelemetryRecord(recordId)
          : await discardTelemetryRecord(recordId);
      setTelemetry((current) => current.filter((record) => record.id !== recordId));
      setNotice(result.message ?? `${action}d telemetry record ${recordId}.`);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : `Failed to ${action} telemetry record.`);
    } finally {
      setActioningId(null);
    }
  }

  async function handleSynthesize() {
    if (!seedPrompt.trim() || synthesizing) return;
    setSynthesizing(true);
    setError(null);
    setNotice(null);
    setSynthJob(null);
    try {
      const res = await synthesizeDataset({
        seed_prompt: seedPrompt,
        num_variants: numVariants,
        model_variant: modelVariant,
      });
      setNotice(res.message);
      setSynthJob({
        job_id: res.job_id,
        status: "running",
        seed_prompt: seedPrompt,
        num_variants: numVariants,
        model_variant: modelVariant,
        created_at: Date.now() / 1000,
        estimated_time_s: res.estimated_time_s,
        completed_rows: 0,
        output_path: "",
      });
      setSeedPrompt("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to dispatch synthesis job.");
    } finally {
      setSynthesizing(false);
    }
  }

  const synthesisProgress = synthJob
    ? Math.min(100, Math.round((synthJob.completed_rows / Math.max(synthJob.num_variants, 1)) * 100))
    : 0;

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">V2 Lab → Datasets</span>
          <h1 className="dash-page-title">Dataset Curator</h1>
          <p className="dash-page-desc">
            Turn production failures into curated training assets and launch new synthetic data jobs without leaving the lab.
          </p>
        </div>
      </div>

      {error && <div className="dash-error-banner panel">⚠ {error}</div>}
      {notice && <div className="panel state-panel">{notice}</div>}

      <div className="workspace-section-grid">
        {/* Left Pane: Telemetry Queue */}
        <div className="panel aside-section">
          <div>
            <h2 className="section-title">Telemetry Queue</h2>
            <p className="control-label">
              Review flagged failure cases, then promote them into the curation queue or discard them cleanly.
            </p>
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
              {telemetry.map((t) => (
                <div key={t.id} className="resource-card">
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
                    <button
                      className="ghost-button small"
                      disabled={actioningId === t.id}
                      onClick={() => void handleTelemetryAction(t.id, "discard")}
                    >
                      {actioningId === t.id ? "Working..." : "Discard"}
                    </button>
                    <button
                      className="primary-button small"
                      disabled={actioningId === t.id}
                      onClick={() => void handleTelemetryAction(t.id, "promote")}
                    >
                      {actioningId === t.id ? "Working..." : "Promote to Dataset"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Pane: Synthesizer */}
        <div className="panel aside-section">
          <div>
            <h2 className="section-title">Synthetic Expansion</h2>
            <p className="control-label">Launch a synthesis job and track row-level progress while it runs.</p>
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
                  <strong style={{ color: "var(--accent)" }}>
                    {synthJob.status === "completed" ? "Synthesis Job Completed" : "Synthesis Job Running"}
                  </strong>
                </div>
              </div>
              <p style={{ fontSize: "0.85rem", opacity: 0.8, margin: 0 }}>Job ID: <code>{synthJob.job_id}</code></p>
              <p style={{ fontSize: "0.85rem", opacity: 0.8, margin: 0 }}>
                Progress: {synthJob.completed_rows}/{synthJob.num_variants} rows
              </p>
              <p style={{ fontSize: "0.85rem", opacity: 0.8, margin: 0 }}>
                Estimated completion: {synthJob.estimated_time_s.toFixed(1)}s
              </p>
              <div style={{ marginTop: "0.85rem", height: "10px", background: "rgba(19, 33, 28, 0.08)", borderRadius: "999px", overflow: "hidden" }}>
                <div
                  style={{
                    width: `${synthesisProgress}%`,
                    height: "100%",
                    background: "linear-gradient(135deg, var(--accent), var(--accent-strong))",
                    transition: "width 0.25s ease",
                  }}
                />
              </div>
              {synthJob.output_path ? (
                <p style={{ fontSize: "0.8rem", opacity: 0.7, margin: "0.75rem 0 0" }}>
                  Output: <code>{synthJob.output_path}</code>
                </p>
              ) : null}
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
