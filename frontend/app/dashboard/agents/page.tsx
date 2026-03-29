"use client";

import { useEffect, useState, useRef } from "react";
import {
  getAgentSwarmStatus,
  getAgentLogs,
  type AgentSwarmStatus,
  type AgentLogEvent,
} from "@/lib/api";

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentSwarmStatus[]>([]);
  const [logs, setLogs] = useState<AgentLogEvent[]>([]);
  const [loading, setLoading] = useState(true);
  
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
          <h2 className="section-title">Active Swarm Agents</h2>
          
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
    </div>
  );
}
