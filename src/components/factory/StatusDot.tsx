interface StatusDotProps {
  status: 'running' | 'warning' | 'critical' | 'idle' | 'offline' | 'completed' | 'failed' | 'queued' | 'degraded';
  size?: 'sm' | 'md';
}

const StatusDot = ({ status, size = 'sm' }: StatusDotProps) => {
  const sizeClass = size === 'sm' ? 'w-2 h-2' : 'w-3 h-3';
  
  const colorMap: Record<string, string> = {
    running: 'bg-neon-green animate-pulse-dot',
    completed: 'bg-neon-green',
    warning: 'bg-neon-orange animate-pulse-dot',
    degraded: 'bg-neon-orange animate-pulse-dot',
    critical: 'bg-neon-red animate-pulse-dot-fast',
    failed: 'bg-neon-red',
    idle: 'bg-muted-foreground/40',
    offline: 'bg-muted-foreground/20',
    queued: 'bg-neon-blue/50',
  };

  return (
    <span className={`inline-block rounded-full ${sizeClass} ${colorMap[status] || colorMap.idle}`} />
  );
};

export default StatusDot;
