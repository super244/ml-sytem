import { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
  LayoutDashboard, Activity, Brain, MessageSquare,
  Database, Sparkles, Bot, Server,
  Lightbulb, Settings, Bell, ChevronLeft, ChevronRight,
} from 'lucide-react';
import StatusDot from './StatusDot';

const navGroups = [
  {
    label: 'LIFECYCLE',
    items: [
      { label: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
      { label: 'Monitoring', path: '/dashboard/monitoring', icon: Activity },
      { label: 'Models', path: '/models', icon: Brain },
      { label: 'Inference', path: '/inference', icon: MessageSquare },
    ],
  },
  {
    label: 'LABORATORY',
    items: [
      { label: 'Datasets', path: '/lab/datasets', icon: Database },
      { label: 'AutoML', path: '/lab/automl', icon: Sparkles },
      { label: 'Agents', path: '/lab/agents', icon: Bot },
      { label: 'Cluster', path: '/lab/cluster', icon: Server },
    ],
  },
  {
    label: 'WORKSPACE',
    items: [
      { label: 'Solve', path: '/solve', icon: Lightbulb },
    ],
  },
];

const AppSidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <aside
      className={`h-screen sticky top-0 flex flex-col border-r border-border bg-deep transition-all duration-300 ${
        collapsed ? 'w-[72px]' : 'w-[260px]'
      }`}
    >
      {/* Header */}
      <div className="flex items-center gap-3 px-4 h-14 border-b border-border shrink-0">
        <div className="w-8 h-8 rounded-lg bg-neon-green/20 flex items-center justify-center">
          <span className="text-neon-green font-bold text-xs font-mono">AI</span>
        </div>
        {!collapsed && (
          <div className="flex-1 min-w-0">
            <div className="text-sm font-display font-semibold text-foreground flex items-center gap-2">
              AI-FACTORY
              <span className="text-[9px] font-mono text-muted-foreground">v2.0</span>
            </div>
          </div>
        )}
        <button
          onClick={() => setCollapsed(c => !c)}
          className="p-1 rounded hover:bg-raised text-muted-foreground"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Nav Groups */}
      <nav className="flex-1 overflow-y-auto py-3 px-2">
        {navGroups.map(group => (
          <div key={group.label} className="mb-4">
            {!collapsed && (
              <div className="section-label px-3 mb-2">{group.label}</div>
            )}
            {group.items.map(item => {
              const active = location.pathname === item.path ||
                (item.path === '/dashboard' && location.pathname === '/');
              const Icon = item.icon;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg mb-0.5 transition-all text-sm ${
                    active
                      ? 'bg-neon-green/10 text-neon-green border-l-2 border-neon-green'
                      : 'text-muted-foreground hover:bg-raised hover:text-foreground'
                  }`}
                  title={collapsed ? item.label : undefined}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {!collapsed && <span className="font-display">{item.label}</span>}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Cluster Mini-Gauge */}
      <div className="border-t border-border p-3 space-y-2 shrink-0">
        {!collapsed ? (
          <>
            <MiniBar label="GPU" icon="⬡" value={82} color="neon-green" detail="2× A100 80GB" />
            <MiniBar label="MEM" icon="◆" value={71} color="neon-blue" detail="640GB" />
            <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono px-1">
              <StatusDot status="running" />
              <span>3 active · 7 queued</span>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <div className="w-6 h-6 rounded-full border border-neon-green/40 flex items-center justify-center">
              <span className="text-[8px] font-mono text-neon-green">82</span>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Actions */}
      <div className="border-t border-border p-2 flex items-center justify-around shrink-0">
        <Link to="/settings" className="p-2 rounded hover:bg-raised text-muted-foreground">
          <Settings className="w-4 h-4" />
        </Link>
        {!collapsed && (
          <button className="p-2 rounded hover:bg-raised text-muted-foreground relative">
            <Bell className="w-4 h-4" />
            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-neon-orange text-[8px] font-mono text-primary-foreground flex items-center justify-center animate-pulse-dot">
              3
            </span>
          </button>
        )}
      </div>
    </aside>
  );
};

const MiniBar = ({ label, icon, value, color, detail }: { label: string; icon: string; value: number; color: string; detail: string }) => (
  <div className="flex items-center gap-2 text-xs">
    <span className="text-muted-foreground font-mono w-8">{icon} {label}</span>
    <div className="flex-1 h-1.5 bg-raised rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full bg-${color} progress-bar-transition ${value > 80 ? 'animate-shimmer' : ''}`}
        style={{ width: `${value}%`, backgroundColor: color === 'neon-green' ? 'hsl(155,100%,50%)' : 'hsl(192,100%,50%)' }}
      />
    </div>
    <span className="font-mono text-muted-foreground w-8 text-right">{value}%</span>
  </div>
);

export default AppSidebar;
