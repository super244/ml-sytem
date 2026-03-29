"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  getInstances,
  evaluateManagedInstance,
  type InstanceSummary,
} from "@/lib/api";

function formatMetric(val: unknown): string {
  if (typeof val === "number") {
    if (Math.abs(val) < 10) return val.toFixed(4);
    return val.toFixed(2);
  }
  if (val === null || val === undefined) return "—";
  return String(val);
}

export default function EvaluatePage() {
  const [instances, setInstances] = useState<InstanceSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [compareId, setCompareId] = useState<string | null>(null);
  const [launching, setLaunching] = useState<string | null>(null);

  useEffect(() => {
    getInstances()
      .then((list) => {
        const sorted = [...list].sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
        setInstances(sorted);
        // Auto-select most recent completed
        const firstCompleted = sorted.find((i) => i.status === "completed");
        if (firstCompleted) setSelected(firstCompleted.id);
      })
      .catch(() => null)
      .finally(() => setLoading(false));
  }, []);

  async function launchEval(instanceId: string) {
    setLaunching(instanceId);
    try {
      await evaluateManagedInstance(instanceId, { start: true });
    } finally {
      setLaunching(null);
    }
  }

  const evaluatable = instances.filter(
    (i) => (i.type === "train" || i.type === "finetune") && i.status === "completed",
  );
  const evaluated = instances.filter((i) => i.type === "evaluate");
  const selectedInstance = instances.find((i) => i.id === selected);
  const compareInstance = instances.find((i) => i.id === compareId);

  const allMetricKeys = Array.from(
    new Set([
      ...Object.keys(selectedInstance?.metrics_summary ?? {}),
      ...Object.keys(compareInstance?.metrics_summary ?? {}),
    ]),
  ).filter((k) => typeof (selectedInstance?.metrics_summary[k] ?? compareInstance?.metrics_summary[k]) === "number");

  return (
    <div className="dashboard-content">
      {/* Header */}
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">Lifecycle → Evaluate</span>
          <h1 className="dash-page-title">Evaluation</h1>
          <p className="dash-page-desc">
            Run standardized benchmarks, compare model performance, and review AI-assisted
            decisions for the next lifecycle step.
          </p>
        </div>
      </div>

      <div className="eval-grid">
        {/* Launch Evaluation */}
        {evaluatable.length > 0 && (
          <div className="eval-launch-panel panel">
            <h2 className="section-title">Launch Evaluation</h2>
            <p className="control-label">Select a completed training or finetuning run to evaluate.</p>
            <div className="eval-launch-list">
              {evaluatable.slice(0, 6).map((inst) => (
                <div key={inst.id} className="eval-launch-item">
                  <div className="eval-launch-info">
                    <span className="eval-launch-name">{inst.name}</span>
                    <span className="eval-launch-type">{inst.type} · {inst.lifecycle?.learning_mode ?? "—"}</span>
                  </div>
                  <button
                    type="button"
                    className="primary-button small"
                    disabled={launching === inst.id}
                    onClick={() => void launchEval(inst.id)}
                  >
                    {launching === inst.id ? "⟳" : "◆ Evaluate"}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Evaluation Results */}
        {evaluated.length > 0 && (
          <div className="eval-results-panel panel">
            <h2 className="section-title">Evaluation Results</h2>
            <div className="eval-results-list">
              {evaluated.slice(0, 8).map((inst) => {
                const isSelected = inst.id === selected;
                const isCompare = inst.id === compareId;
                return (
                  <div
                    key={inst.id}
                    className={`eval-result-item ${isSelected ? "selected" : ""} ${isCompare ? "compare" : ""}`}
                    onClick={() => {
                      if (isSelected) return;
                      if (isCompare) {
                        setCompareId(null);
                      } else if (!selected) {
                        setSelected(inst.id);
                      } else {
                        setCompareId(inst.id);
                      }
                    }}
                  >
                    <div className="eval-result-header">
                      <span className="eval-result-name">{inst.name}</span>
                      <span className={`eval-result-badge ${isSelected ? "selected" : isCompare ? "compare" : ""}`}>
                        {isSelected ? "Primary" : isCompare ? "Compare" : inst.status}
                      </span>
                    </div>
                    {inst.decision && (
                      <div className="eval-decision">
                        <span className="eval-decision-icon">◆</span>
                        <span>{inst.decision.action}</span>
                        <span className="eval-decision-rule">({inst.decision.rule})</span>
                        {inst.decision.explanation && (
                          <span className="eval-decision-exp">{inst.decision.explanation}</span>
                        )}
                      </div>
                    )}
                    <div className="eval-metrics-row">
                      {Object.entries(inst.metrics_summary)
                        .filter(([, v]) => typeof v === "number")
                        .slice(0, 3)
                        .map(([key, val]) => (
                          <div key={key} className="eval-metric-chip">
                            <span className="eval-metric-key">{key}</span>
                            <span className="eval-metric-val">{formatMetric(val)}</span>
                          </div>
                        ))}
                    </div>
                    {inst.recommendations && inst.recommendations.length > 0 && (
                      <div className="eval-recommendations">
                        {inst.recommendations.slice(0, 2).map((rec, idx) => (
                          <div key={idx} className="eval-rec">
                            <span className="eval-rec-priority">{"★".repeat(Math.min(rec.priority, 3))}</span>
                            <span className="eval-rec-action">{rec.action}</span>
                            <span className="eval-rec-reason">{rec.reason}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {selected && (
              <p className="eval-compare-hint">
                Click another result to compare side-by-side.
              </p>
            )}
          </div>
        )}

        {/* Comparison View */}
        {selectedInstance && compareInstance && (
          <div className="eval-comparison-panel panel">
            <h2 className="section-title">Side-by-Side Comparison</h2>
            <div className="comparison-header-row">
              <div className="comparison-col-header">
                <span className="eval-col-label selected">Primary</span>
                <span className="eval-col-name">{selectedInstance.name}</span>
              </div>
              <div className="comparison-col-header">
                <span className="eval-col-label compare">Compare</span>
                <span className="eval-col-name">{compareInstance.name}</span>
              </div>
            </div>
            <div className="comparison-metrics-table">
              <div className="comparison-table-header">
                <span>Metric</span>
                <span>Primary</span>
                <span>Compare</span>
                <span>Delta</span>
              </div>
              {allMetricKeys.map((key) => {
                const lv = selectedInstance.metrics_summary[key] as number | undefined;
                const rv = compareInstance.metrics_summary[key] as number | undefined;
                const delta = typeof lv === "number" && typeof rv === "number" ? rv - lv : null;
                return (
                  <div key={key} className="comparison-table-row">
                    <span className="comparison-metric-key">{key}</span>
                    <span>{formatMetric(lv)}</span>
                    <span>{formatMetric(rv)}</span>
                    <span className={`comparison-delta ${delta !== null ? (delta > 0 ? "positive" : "negative") : ""}`}>
                      {delta !== null ? `${delta > 0 ? "+" : ""}${delta.toFixed(4)}` : "—"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {loading && (
          <div className="dash-loading panel">
            <span className="dash-loading-icon">⟳</span>
            <span>Loading evaluation data…</span>
          </div>
        )}

        {!loading && evaluated.length === 0 && evaluatable.length === 0 && (
          <div className="dash-empty panel">
            <p>No evaluation data yet. Complete a training run first.</p>
            <Link href="/dashboard/training" className="primary-button small">
              ▲ Launch Training
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
