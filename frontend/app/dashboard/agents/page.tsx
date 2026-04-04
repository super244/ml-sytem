"use client";

import { useEffect, useState, useRef } from "react";
import {
  getAgentSwarmSnapshot,
  getAgentLogsSnapshot,
  getMissionControl,
  deployAgent,
  updateAgent,
  type AgentLogsSnapshot,
  type AgentSwarmSnapshot,
  type AgentSwarmStatus,
  type AgentLogEvent,
  type AgentDeployRequest,
  type AgentUpdateRequest,
  type MissionControlSnapshot,
} from "@/lib/api";
import { AutonomyLoopPanel } from "@/components/autonomy-loop-panel";

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentSwarmStatus[]>([]);
  const [logs, setLogs] = useState<AgentLogEvent[]>([]);
  const [mission, setMission] = useState<MissionControlSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [deployEnabled, setDeployEnabled] = useState(false);
  const [simulationEnabled, setSimulationEnabled] = useState(false);
  
  // Deploy Agent State
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [newAgent, setNewAgent] = useState<AgentDeployRequest>({
    name: "",
    role: "",
    model: "gpt-4o",
  });
  const [deploying, setDeploying] = useState(false);
  const [editingAgentId, setEditingAgentId] = useState<string | null>(null);
  const [editingAgent, setEditingAgent] = useState<AgentUpdateRequest>({
    name: "",
    role: "",
    model: "gpt-4o",
    status: "active",
  });
  const [savingAgent, setSavingAgent] = useState(false);

  const endOfTerminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;

    async function refresh(initialLoad: boolean = false) {
      if (initialLoad) {
        setLoading(true);
      }

      try {
        const [swarmResponse, logsResponse, missionResponse] = await Promise.all([
          getAgentSwarmSnapshot(),
          getAgentLogsSnapshot(initialLoad ? 5 : 50),
          getMissionControl(),
        ]);
        if (cancelled) {
          return;
        }
        const swarmPayload: AgentSwarmSnapshot = swarmResponse;
        const logsPayload: AgentLogsSnapshot = logsResponse;
        setAgents(swarmPayload.swarm);
        setDeployEnabled(Boolean(swarmPayload.deploy_enabled));
        setSimulationEnabled(Boolean(logsPayload.simulation_enabled));
        setLogs(logsPayload.logs);
        setMission(missionResponse);
        setError(null);
      } catch (nextError) {
        if (!cancelled) {
          setError(nextError instanceof Error ? nextError.message : "Failed to load swarm state.");
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
    }, 3000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    endOfTerminalRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  function formatUptime(seconds: number) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  }

  async function handleDeploy() {
    if (!newAgent.name || !newAgent.role || deploying) return;
    setDeploying(true);
    setError(null);
    setNotice(null);
    try {
      const res = await deployAgent(newAgent);
      setAgents((prev) => [...prev, res.agent]);
      setShowDeployModal(false);
      setNewAgent({ name: "", role: "", model: "gpt-4o" });
      setNotice(`Agent '${res.agent.name}' deployed.`);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to deploy agent.");
    } finally {
      setDeploying(false);
    }
  }

  function openAgentEditor(agent: AgentSwarmStatus) {
    setEditingAgentId(agent.id);
    setEditingAgent({
      name: agent.name,
      role: agent.role,
      model: agent.model,
      status: agent.status,
    });
  }

  async function handleAgentSave() {
    if (!editingAgentId || savingAgent) return;
    setSavingAgent(true);
    setError(null);
    setNotice(null);

    try {
      const res = await updateAgent(editingAgentId, editingAgent);
      setAgents((prev) => prev.map((agent) => (agent.id === editingAgentId ? res.agent : agent)));
      setNotice(`Updated ${res.agent.name}.`);
      setEditingAgentId(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to update agent.");
    } finally {
      setSavingAgent(false);
    }
  }

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">V2 Lab → Agents</span>
          <h1 className="dash-page-title">Swarm Controller</h1>
          <p className="dash-page-desc">
            Inspect the active swarm, adjust agent roles, and watch orchestration activity as it happens.
          </p>
        </div>
      </div>

      {error && <div className="dash-error-banner panel">⚠ {error}</div>}
      {notice && <div className="panel state-panel">{notice}</div>}
      {!deployEnabled && (
        <div className="panel state-panel">
          Agent registry and logs are live, but deploying simulated swarm agents is disabled outside demo mode.
        </div>
      )}

      {mission?.autonomy ? (
        <AutonomyLoopPanel
          autonomy={mission.autonomy}
          title="Swarm Coverage"
          description="Agent coverage is measured against the shared orchestration queue, not just the persisted registry."
          maxStages={2}
          maxActions={2}
          compact
        />
      ) : null}

      {mission?.autonomy ? (
        <section className="panel aside-section">
          <div className="model-chip-header">
            <div>
              <h2 className="section-title">Agent Coverage Matrix</h2>
              <p className="control-label">
                Queue pressure, running tasks, and swarm coverage are synchronized against the same autonomy snapshot.
              </p>
            </div>
            <span className="status-pill">{mission.autonomy.mode}</span>
          </div>
          <div className="resource-list" style={{ marginTop: "1rem" }}>
            {mission.autonomy.agent_coverage.map((coverage) => (
              <article
                key={coverage.agent_type}
                className="resource-card"
                style={{
                  borderTop: `3px solid ${
                    coverage.open_circuit ? "var(--danger)" : coverage.status === "attention" ? "#d9a441" : coverage.status === "active" ? "var(--accent)" : "var(--line)"
                  }`,
                }}
              >
                <div className="model-chip-header">
                  <strong>{coverage.label}</strong>
                  <span style={{ textTransform: "capitalize" }}>{coverage.status}</span>
                </div>
                <p>{coverage.recommended_action}</p>
                <div className="badge-row">
                  <span className="status-pill">queued {coverage.queued_tasks}</span>
                  <span className="status-pill">running {coverage.running_tasks}</span>
                  <span className="status-pill">swarm {coverage.active_swarm_agents}</span>
                  <span className="status-pill">{coverage.resource_classes.join(" / ") || "control"}</span>
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <div className="workspace-section-grid">
        
        {/* Top: Active Swarm Cards */}
        <div className="panel aside-section">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <h2 className="section-title" style={{ margin: 0 }}>Active Swarm Agents</h2>
            <button 
              className="primary-button small" 
              onClick={() => setShowDeployModal(true)}
              disabled={!deployEnabled}
            >
              {deployEnabled ? "+ Add Agent" : "Demo Only"}
            </button>
          </div>
          
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1rem" }}>
            {loading ? (
              <div className="dash-loading"><span>⟳</span> Connecting to Swarm...</div>
            ) : agents.length === 0 ? (
              <div className="dash-empty" style={{ padding: "2rem 0" }}>
                No persisted agents are registered yet.
              </div>
            ) : agents.map((agent) => (
              <div key={agent.id} className="resource-card" style={{ borderTop: agent.status === "active" ? "3px solid var(--accent)" : "3px solid var(--line)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
                  <h3 style={{ margin: 0, fontSize: "1.1rem" }}>{agent.name}</h3>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span className={`monitor-status-dot status-${agent.status === "active" ? "running" : "idle"}`} />
                    <span style={{ fontSize: "0.8rem", textTransform: "capitalize", opacity: 0.8 }}>{agent.status}</span>
                  </div>
                </div>
                
                <p style={{ fontSize: "0.85rem", opacity: 0.8, minHeight: "2.5rem", marginBottom: "1rem" }}>
                  {agent.role}
                </p>
                
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.85rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ opacity: 0.6 }}>Backend</span>
                    <strong style={{ opacity: 0.9 }}>{agent.model}</strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ opacity: 0.6 }}>Uptime</span>
                    <strong style={{ opacity: 0.9 }}>{formatUptime(agent.uptime_s)}</strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ opacity: 0.6 }}>Tokens Consumed</span>
                    <strong style={{ opacity: 0.9 }}>{agent.tokens_used.toLocaleString()}</strong>
                  </div>
                </div>
                
                <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid var(--line)", display: "flex", justifyContent: "flex-end" }}>
                  <button className="ghost-button small" onClick={() => openAgentEditor(agent)}>
                    Configure Instructions
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom: Thought Stream Terminal */}
        <div className="panel aside-section" style={{ flex: 1 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2 className="section-title" style={{ margin: 0 }}>Live Thought Stream</h2>
            <div style={{ fontSize: "0.8rem", color: "var(--accent)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span className="monitor-status-dot status-running" /> {simulationEnabled ? "Simulated stream" : "Persisted log stream"}
            </div>
          </div>
          
          <div 
            style={{ 
              backgroundColor: "#0d1117", 
              border: "1px solid var(--line)", 
              borderRadius: "8px", 
              padding: "1rem", 
              height: "400px", 
              overflowY: "auto",
              fontFamily: "monospace",
              fontSize: "0.85rem",
              color: "#c9d1d9",
              display: "flex",
              flexDirection: "column",
              gap: "0.5rem"
            }}
          >
            {logs.length === 0 ? (
              <div style={{ color: "#8b949e" }}>
                No agent log events have been recorded yet.
              </div>
            ) : logs.map((log) => {
              const timeStr = new Date(log.timestamp * 1000).toLocaleTimeString([], { hour12: false });
              const isAccent = log.message.includes("Optimization") || log.message.includes("Red-Team");
              return (
                <div key={log.timestamp} style={{ display: "flex", gap: "1rem" }}>
                  <span style={{ color: "#8b949e", minWidth: "80px" }}>[{timeStr}]</span>
                  <span style={{ color: isAccent ? "var(--accent)" : "inherit" }}>{log.message}</span>
                </div>
              );
            })}
            <div ref={endOfTerminalRef} />
          </div>
        </div>

      </div>

      {/* Deploy Agent Modal */}
      {showDeployModal && (
        <div className="modal-overlay" style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, backdropFilter: "blur(4px)" }}>
          <div className="panel" style={{ width: "400px", padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
            <h2 className="section-title">Deploy New Swarm Agent</h2>
            
            <div className="input-group">
              <label className="control-label">Agent Name</label>
              <input 
                type="text" 
                value={newAgent.name} 
                onChange={e => setNewAgent({...newAgent, name: e.target.value})}
                placeholder="e.g. Code Reviewer"
              />
            </div>

            <div className="input-group">
              <label className="control-label">Primary Role</label>
              <textarea 
                value={newAgent.role} 
                onChange={e => setNewAgent({...newAgent, role: e.target.value})}
                placeholder="e.g. Analyzes diffs for security vulnerabilities..."
                style={{ height: "80px" }}
              />
            </div>

            <div className="input-group">
              <label className="control-label">Inference Model</label>
              <select 
                value={newAgent.model} 
                onChange={e => setNewAgent({...newAgent, model: e.target.value})}
              >
                <option value="gpt-4o">GPT-4o</option>
                <option value="claude-3-sonnet">Claude 3.5 Sonnet</option>
                <option value="qwen-2-72b">Qwen-2 72B</option>
              </select>
            </div>

            <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.5rem" }}>
              <button 
                className="primary-button" 
                style={{ flex: 1, justifyContent: "center" }}
                disabled={deploying || !newAgent.name || !newAgent.role || !deployEnabled}
                onClick={() => void handleDeploy()}
              >
                {deploying ? "⟳ Deploying..." : "Deploy Agent"}
              </button>
              <button 
                className="ghost-button" 
                onClick={() => setShowDeployModal(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {editingAgentId && (
        <div className="modal-overlay" style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, backdropFilter: "blur(4px)" }}>
          <div className="panel" style={{ width: "420px", padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
            <h2 className="section-title">Configure Agent</h2>

            <div className="input-group">
              <label className="control-label">Agent Name</label>
              <input
                type="text"
                value={editingAgent.name ?? ""}
                onChange={(e) => setEditingAgent((current) => ({ ...current, name: e.target.value }))}
              />
            </div>

            <div className="input-group">
              <label className="control-label">Role</label>
              <textarea
                value={editingAgent.role ?? ""}
                onChange={(e) => setEditingAgent((current) => ({ ...current, role: e.target.value }))}
                style={{ height: "88px" }}
              />
            </div>

            <div className="input-group">
              <label className="control-label">Model</label>
              <input
                type="text"
                value={editingAgent.model ?? ""}
                onChange={(e) => setEditingAgent((current) => ({ ...current, model: e.target.value }))}
              />
            </div>

            <div className="input-group">
              <label className="control-label">Status</label>
              <select
                value={editingAgent.status ?? "active"}
                onChange={(e) => setEditingAgent((current) => ({ ...current, status: e.target.value as AgentSwarmStatus["status"] }))}
              >
                <option value="active">Active</option>
                <option value="sleeping">Sleeping</option>
                <option value="offline">Offline</option>
              </select>
            </div>

            <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.5rem" }}>
              <button
                className="primary-button"
                style={{ flex: 1, justifyContent: "center" }}
                disabled={savingAgent || !editingAgent.name || !editingAgent.role}
                onClick={() => void handleAgentSave()}
              >
                {savingAgent ? "Saving..." : "Save Changes"}
              </button>
              <button className="ghost-button" onClick={() => setEditingAgentId(null)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
