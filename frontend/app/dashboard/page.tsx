"use client";

import { useEffect, useState } from "react";

import { getWorkspaceOverview, type WorkspaceOverview } from "@/lib/api";

type DashboardState = {
  workspace: WorkspaceOverview | null;
  loading: boolean;
  error: string | null;
};

export default function Dashboard() {
  const [state, setState] = useState<DashboardState>({
    workspace: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let mounted = true;
    
    async function loadWorkspace() {
      try {
        setState(prev => ({ ...prev, loading: true, error: null }));
        
        // Try instant workspace first, then load full data
        const workspace = await getWorkspaceOverview();
        
        if (mounted) {
          setState({ workspace, loading: false, error: null });
        }
      } catch (err) {
        if (mounted) {
          setState({ 
            workspace: null, 
            loading: false, 
            error: err instanceof Error ? err.message : "Failed to load workspace" 
          });
        }
      }
    }

    loadWorkspace();
    
    return () => { mounted = false; };
  }, []);

  if (state.loading) {
    return (
      <div className="dashboard-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading AI-Factory...</p>
        </div>
      </div>
    );
  }

  if (state.error) {
    return (
      <div className="dashboard-container">
        <div className="error-container">
          <h2>Connection Error</h2>
          <p>{state.error}</p>
          <button onClick={() => window.location.reload()} className="primary-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  const { workspace } = state;
  if (!workspace) return null;

  const readyChecks = workspace.readiness_checks?.filter((check: any) => check.ok).length || 0;
  const totalChecks = workspace.readiness_checks?.length || 0;

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-content">
          <h1>AI-Factory Dashboard</h1>
          <div className="status-indicator">
            <span className={`status-dot ${readyChecks === totalChecks ? 'ready' : 'partial'}`}></span>
            <span>{readyChecks}/{totalChecks} systems ready</span>
          </div>
        </div>
      </header>

      {/* Quick Stats */}
      <section className="quick-stats">
        <h2>System Overview</h2>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{workspace.summary?.datasets || 0}</div>
            <div className="stat-label">Datasets</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{workspace.summary?.models || 0}</div>
            <div className="stat-label">Models</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{workspace.summary?.runs || 0}</div>
            <div className="stat-label">Training Runs</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{workspace.summary?.benchmarks || 0}</div>
            <div className="stat-label">Benchmarks</div>
          </div>
        </div>
      </section>

      {/* Quick Actions */}
      <section className="quick-actions">
        <h2>Quick Actions</h2>
        <div className="actions-grid">
          <a href="/training" className="action-card">
            <div className="action-icon">🎯</div>
            <div className="action-content">
              <h3>Start Training</h3>
              <p>Launch a new training experiment</p>
            </div>
          </a>
          
          <a href="/evaluate" className="action-card">
            <div className="action-icon">📊</div>
            <div className="action-content">
              <h3>Evaluate Models</h3>
              <p>Run model benchmarks and evaluations</p>
            </div>
          </a>
          
          <a href="/datasets" className="action-card">
            <div className="action-icon">📚</div>
            <div className="action-content">
              <h3>Manage Datasets</h3>
              <p>Browse and curate training data</p>
            </div>
          </a>
          
          <a href="/generate" className="action-card">
            <div className="action-icon">🤖</div>
            <div className="action-content">
              <h3>Generate Content</h3>
              <p>Test model inference and generation</p>
            </div>
          </a>
        </div>
      </section>

      {/* System Status */}
      <section className="system-status">
        <h2>System Status</h2>
        <div className="status-list">
          {workspace.readiness_checks?.map((check: any) => (
            <div key={check.id} className={`status-item ${check.ok ? 'ok' : 'error'}`}>
              <span className={`status-indicator ${check.ok ? 'ok' : 'error'}`}></span>
              <div className="status-content">
                <div className="status-title">{check.label}</div>
                <div className="status-detail">{check.detail}</div>
              </div>
            </div>
          )) || []}
        </div>
      </section>

      {/* Available Commands */}
      {workspace.command_recipes && workspace.command_recipes.length > 0 && (
        <section className="recent-activity">
          <h2>Available Commands</h2>
          <div className="commands-grid">
            {workspace.command_recipes.slice(0, 6).map((recipe: any) => (
              <div key={recipe.id} className="command-card">
                <div className="command-title">{recipe.title}</div>
                <div className="command-description">{recipe.description}</div>
                <code className="command-code">{recipe.command}</code>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
