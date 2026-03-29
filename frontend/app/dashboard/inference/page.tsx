"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  generateAnswer,
  getInstances,
  startManagedInference,
  flagTelemetry,
  type GenerateResult,
  type InstanceSummary,
} from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  latency?: number;
  flagged?: boolean;
};

export default function InferencePage() {
  const [instances, setInstances] = useState<InstanceSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>("default");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [launchingId, setLaunchingId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  
  // V2 Telemetry Flag State
  const [flagModalIdx, setFlagModalIdx] = useState<number | null>(null);
  const [flagReason, setFlagReason] = useState("");

  useEffect(() => {
    getInstances()
      .then((list) => {
        const inferenceReady = list.filter(
          (i) =>
            (i.type === "inference" && i.status === "running") ||
            (i.type === "deploy" && i.status === "completed"),
        );
        setInstances(inferenceReady);
      })
      .catch(() => null)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    if (!input.trim() || generating) return;
    const userMsg: Message = { role: "user", content: input };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setGenerating(true);
    const t0 = Date.now();
    try {
      const result = await generateAnswer({
        question: input,
        model_variant: selectedModel,
        compare_to_base: false,
        prompt_preset: "default",
        temperature,
        top_p: 0.95,
        max_new_tokens: maxTokens,
        show_reasoning: false,
        difficulty_target: "medium",
        num_samples: 1,
        use_calculator: false,
        solver_mode: "concise",
        output_format: "text",
        use_cache: false,
      });
      const latency = (Date.now() - t0) / 1000;
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: result.answer || result.raw_text,
          latency,
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Error: ${e instanceof Error ? e.message : "Inference failed"}`,
        },
      ]);
    } finally {
      setGenerating(false);
    }
  }

  async function launchInference(instanceId: string) {
    setLaunchingId(instanceId);
    try {
      await startManagedInference(instanceId, { start: true });
      const list = await getInstances();
      setInstances(list.filter((i) => i.type === "inference" && i.status === "running"));
    } finally {
      setLaunchingId(null);
    }
  }

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">Lifecycle → Inference</span>
          <h1 className="dash-page-title">Inference</h1>
          <p className="dash-page-desc">
            Chat with your trained and deployed models. Built-in inference UI with
            configurable temperature and token limits.
          </p>
        </div>
      </div>

      <div className="inference-layout">
        {/* Sidebar: Model & Settings */}
        <div className="inference-sidebar">
          {/* Model Selection */}
          <div className="panel inference-settings-panel">
            <h2 className="section-title">Model</h2>
            <div className="input-group">
              <label className="control-label" htmlFor="model-select">Active model</label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="default">Default (finetuned)</option>
                <option value="base">Base model</option>
              </select>
            </div>

            {instances.length > 0 && (
              <>
                <p className="control-label">Live inference instances</p>
                {instances.map((inst) => (
                  <div key={inst.id} className="inference-instance-chip">
                    <span className={`monitor-status-dot status-${inst.status}`} />
                    <span className="inference-chip-name">{inst.name}</span>
                  </div>
                ))}
              </>
            )}
          </div>

          {/* Settings */}
          <div className="panel inference-settings-panel">
            <h2 className="section-title">Settings</h2>
            <div className="input-group">
              <label className="control-label" htmlFor="temperature">
                Temperature: {temperature.toFixed(1)}
              </label>
              <input
                id="temperature"
                type="range"
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label className="control-label" htmlFor="max-tokens">
                Max tokens: {maxTokens}
              </label>
              <input
                id="max-tokens"
                type="range"
                min={64}
                max={2048}
                step={64}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
            </div>
          </div>

          {/* Quick Launch */}
          {!loading && instances.length === 0 && (
            <div className="panel inference-settings-panel">
              <h2 className="section-title">No Active Instances</h2>
              <p className="control-label">
                Start an inference instance from a completed training run.
              </p>
              <Link href="/dashboard/deploy" className="secondary-button small">
                ⬆ Deploy a Model
              </Link>
            </div>
          )}

          {messages.length > 0 && (
            <button
              type="button"
              className="ghost-button"
              onClick={() => setMessages([])}
            >
              Clear chat
            </button>
          )}
        </div>

        {/* Chat Window */}
        <div className="panel inference-chat-panel">
          <div className="inference-messages">
            {messages.length === 0 && (
              <div className="inference-empty-state">
                <span className="inference-empty-icon">◎</span>
                <p>Start a conversation with your model.</p>
                <p className="inference-empty-hint">
                  Model: <strong>{selectedModel}</strong>
                </p>
              </div>
            )}
            {messages.map((msg, idx) => (
              <div key={idx} className={`inference-message ${msg.role}`}>
                <div className="inference-message-role" style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>{msg.role === "user" ? "You" : "Model"}</span>
                  {msg.role === "assistant" && (
                    <button
                      type="button"
                      className="ghost-button small"
                      style={{ padding: "0 0.5rem", height: "auto", fontSize: "0.75rem", opacity: msg.flagged ? 1 : 0.6 }}
                      onClick={() => setFlagModalIdx(flagModalIdx === idx ? null : idx)}
                      disabled={msg.flagged}
                    >
                      {msg.flagged ? "✓ Flagged for dataset" : "👎 Flag as failure"}
                    </button>
                  )}
                </div>
                <div className="inference-message-content" style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>
                
                {flagModalIdx === idx && (
                  <div className="panel" style={{ marginTop: "1rem", padding: "1rem", background: "rgba(255,255,255,0.02)", border: "1px solid var(--line)" }}>
                    <p style={{ fontSize: "0.8rem", marginBottom: "0.5rem", opacity: 0.8 }}>
                      What went wrong? This pairs the prompt with your correction for the V2 synthetics pipeline.
                    </p>
                    <textarea 
                      value={flagReason} 
                      onChange={(e) => setFlagReason(e.target.value)}
                      placeholder="Expected output or reason for failure..."
                      style={{ width: "100%", background: "transparent", color: "inherit", border: "1px solid var(--line)", borderRadius: "4px", padding: "0.5rem", minHeight: "60px", fontSize: "0.85rem", marginBottom: "0.5rem" }}
                    />
                    <div style={{ display: "flex", gap: "0.5rem" }}>
                      <button 
                        className="primary-button small"
                        onClick={() => {
                          const newMessages = [...messages];
                          newMessages[idx].flagged = true;
                          setMessages(newMessages);
                          
                          let promptText = "";
                          for (let i = idx - 1; i >= 0; i--) {
                            if (messages[i].role === "user") {
                              promptText = messages[i].content;
                              break;
                            }
                          }

                          void flagTelemetry({
                            prompt: promptText,
                            assistant_output: msg.content,
                            expected_output: flagReason,
                            model_variant: selectedModel,
                            latency_s: msg.latency
                          });

                          setFlagModalIdx(null);
                          setFlagReason("");
                        }}
                      >
                        Submit to Telemetry
                      </button>
                      <button className="ghost-button small" onClick={() => setFlagModalIdx(null)}>Cancel</button>
                    </div>
                  </div>
                )}

                {msg.latency != null && (
                  <div className="inference-message-meta">
                    {msg.latency.toFixed(2)}s
                  </div>
                )}
              </div>
            ))}
            {generating && (
              <div className="inference-message assistant">
                <div className="inference-message-role">Model</div>
                <div className="inference-generating">
                  <span className="inference-thinking-dot" />
                  <span className="inference-thinking-dot" />
                  <span className="inference-thinking-dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="inference-composer">
            <textarea
              className="inference-input"
              placeholder="Ask your model anything…"
              value={input}
              rows={3}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            <button
              type="button"
              className="primary-button inference-send-btn"
              disabled={generating || !input.trim()}
              onClick={() => void send()}
            >
              {generating ? "⟳" : "Send →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
