import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, ArrowRight, Cpu, Brain, Database, Zap, BarChart3 } from 'lucide-react';

const routes = [
  { label: 'Dashboard', path: '/dashboard', icon: Zap, category: 'Navigate' },
  { label: 'Monitoring', path: '/dashboard/monitoring', icon: Cpu, category: 'Navigate' },
  { label: 'Models', path: '/models', icon: Brain, category: 'Navigate' },
  { label: 'Inference', path: '/inference', icon: ArrowRight, category: 'Navigate' },
  { label: 'Datasets', path: '/lab/datasets', icon: Database, category: 'Navigate' },
  { label: 'AutoML', path: '/lab/automl', icon: BarChart3, category: 'Navigate' },
  { label: 'Agents', path: '/lab/agents', icon: Brain, category: 'Navigate' },
  { label: 'Cluster', path: '/lab/cluster', icon: Cpu, category: 'Navigate' },
  { label: 'Solve', path: '/solve', icon: Zap, category: 'Workspace' },
  { label: 'Settings', path: '/settings', icon: Zap, category: 'System' },
];

const CommandPalette = () => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const navigate = useNavigate();

  const filtered = routes.filter(r =>
    r.label.toLowerCase().includes(query.toLowerCase()) ||
    r.category.toLowerCase().includes(query.toLowerCase())
  );

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      setOpen(o => !o);
      setQuery('');
      setSelectedIndex(0);
    }
    if (!open) return;
    if (e.key === 'Escape') setOpen(false);
    if (e.key === 'ArrowDown') { e.preventDefault(); setSelectedIndex(i => Math.min(i + 1, filtered.length - 1)); }
    if (e.key === 'ArrowUp') { e.preventDefault(); setSelectedIndex(i => Math.max(i - 1, 0)); }
    if (e.key === 'Enter' && filtered[selectedIndex]) {
      navigate(filtered[selectedIndex].path);
      setOpen(false);
    }
  }, [open, filtered, selectedIndex, navigate]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]" onClick={() => setOpen(false)}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative glass-panel w-full max-w-lg p-0 overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search className="w-4 h-4 text-muted-foreground" />
          <input
            autoFocus
            value={query}
            onChange={e => { setQuery(e.target.value); setSelectedIndex(0); }}
            placeholder="Search commands..."
            className="flex-1 bg-transparent text-sm font-display text-foreground placeholder:text-muted-foreground outline-none"
          />
          <kbd className="text-[10px] font-mono text-muted-foreground bg-raised px-1.5 py-0.5 rounded">ESC</kbd>
        </div>
        <div className="max-h-64 overflow-y-auto p-1">
          {filtered.map((route, i) => {
            const Icon = route.icon;
            return (
              <button
                key={route.path}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  i === selectedIndex ? 'bg-raised text-foreground' : 'text-muted-foreground hover:bg-raised/50'
                }`}
                onClick={() => { navigate(route.path); setOpen(false); }}
                onMouseEnter={() => setSelectedIndex(i)}
              >
                <Icon className="w-4 h-4" />
                <span className="font-display">{route.label}</span>
                <span className="ml-auto text-[10px] section-label">{route.category}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default CommandPalette;
