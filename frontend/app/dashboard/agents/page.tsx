"use client";

import { useEffect, useState, useRef } from "react";
import {
  getAgentSwarmStatus,
  getAgentLogs,
  deployAgent,
  type AgentSwarmStatus,
  type AgentLogEvent,
  type AgentDeployRequest,
} from "@/lib/api";

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentSwarmStatus[]>([]);
  const [logs, setLogs] = useState<AgentLogEvent[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Deploy Agent State
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [newAgent, setNewAgent] = useState<AgentDeployRequest>({
    name: "",
    role: "",
    model: "gpt-4o",
  });
  const [deploying, setDeploying] = useState(false);

  const endOfTerminalRef = useRef<HTMLDivElement>(null);

  // Initial load
  useEffect(() => {
    Promise.all([getAgentSwarmStatus(), getAgentLogs(5)])
      .then(([swarm, initialLogs]) => {
        setAgents(swarm);
        setLogs(initialLogs);
      })
      .catch(() => null)
      .finally(() => setLoading(false));
  }, []);

  // Polling loop for thought stream
  useEffect(() => {
    if (loading) return;
    const interval = setInterval(() => {
      getAgentLogs(1).then((newLogs) => {
        if (newLogs.length > 0) {
          setLogs((prev) => {
            const next = [...prev, ...newLogs];
            // Deduplicate roughly and keep last 50
            const unique = Array.from(new Map(next.map(item => [item.timestamp, item])).values());
            return unique.slice(-50);
          });
        }
      }).catch(() => null);
    }, 3000);
    return () => clearInterval(interval);
  }, [loading]);

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
    try {
      const res = await deployAgent(newAgent);
      setAgents((prev) => [...prev, res.agent]);
      setShowDeployModal(false);
      setNewAgent({ name: "", role: "", model: "gpt-4o" });
    } catch (e) {
      console.error(e);
    } finally {
      setDeploying(false);
    }
  }

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">V2 Lab → Agents</span>
          <h1 className="dash-page-title">Swarm Controller</h1>
          <p className="dash-page-desc">
            Monitor the multi-agent AI brain orchestrating the lab. 
            View autonomous data curation, red-team evaluations, and training optimizations in real-time.
          </p>
        </div>
      </div>

      <div className="workspace-section-grid">
        
        {/* Top: Active Swarm Cards */}
        <div className="panel aside-section">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <h2 className="section-title" style={{ margin: 0 }}>Active Swarm Agents</h2>
            <button 
              className="primary-button small" 
              onClick={() => setShowDeployModal(true)}
            >
              + Add Agent
            </button>
          </div>
          
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1rem" }}>
            {loading ? (
              <div className="dash-loading"><span>⟳</span> Connecting to Swarm...</div>
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
                  <button className="ghost-button small">Configure Instructions</button>
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
              <span className="monitor-status-dot status-running" /> Network Connected
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
            {logs.map((log) => {
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
                disabled={deploying || !newAgent.name || !newAgent.role}
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
    </div>
  );
}
