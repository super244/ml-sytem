import { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import StatusDot from '@/components/factory/StatusDot';
import { LoadingSkeleton } from '@/components/factory/LoadingState';
import { useWebSocket } from '@/hooks/useWebSocket';
import { clusterNodes as mockNodes } from '@/data/mockData';
import type { ClusterNode } from '@/data/mockData';
import { Plus } from 'lucide-react';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const Monitoring = () => {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const { gpuTelemetry, isConnected } = useWebSocket();

  const { data: nodes, isLoading } = useQuery<ClusterNode[]>({
    queryKey: ['/cluster/nodes'],
  });

  const clusterNodes = nodes || mockNodes;
  const selected = clusterNodes.find(n => n.id === selectedNode);

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate">
        <PageHeader
          title="Cluster Fleet"
          subtitle={`${clusterNodes.filter(n => n.status !== 'offline').length} nodes online${isConnected ? ' · LIVE' : ''}`}
          actions={
            <button data-testid="button-add-node" className="text-xs font-mono px-3 py-1.5 rounded-lg bg-raised border border-border text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5">
              <Plus className="w-3 h-3" /> Add Node
            </button>
          }
        />
        <div className="p-6 space-y-6">
          <div>
            <div className="section-label px-1 mb-3">FLEET MAP</div>
            {isLoading && !clusterNodes.length ? (
              <LoadingSkeleton rows={2} />
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {clusterNodes.map(node => (
                  <GlassCard
                    key={node.id}
                    glow={node.status === 'running' ? 'green' : node.status === 'warning' ? 'orange' : node.status === 'critical' ? 'red' : 'none'}
                    hover
                    className="cursor-pointer"
                    onClick={() => setSelectedNode(node.id)}
                    data-testid={`card-node-${node.id}`}
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <StatusDot status={node.status} size="md" />
                      <span className="text-sm font-display font-medium text-foreground">{node.name}</span>
                    </div>
                    {node.gpus.map((gpu, i) => (
                      <div key={i} className="mb-2">
                        <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground mb-1">
                          <span>{gpu.name}</span>
                          <span>{gpu.util}% util</span>
                        </div>
                        <div className="h-1.5 bg-raised rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full progress-bar-transition"
                            style={{
                              width: `${gpu.util}%`,
                              backgroundColor: gpu.util >= 90 ? 'hsl(348,100%,50%)' : gpu.util >= 75 ? 'hsl(18,100%,57%)' : 'hsl(155,100%,50%)',
                            }}
                          />
                        </div>
                        <div className="flex justify-between text-[10px] font-mono text-muted-foreground mt-1">
                          <span>VRAM: {gpu.vram}/{gpu.vramTotal}GB</span>
                          <span className={gpu.temp >= 80 ? 'text-neon-orange' : ''}>{gpu.temp}°C</span>
                        </div>
                      </div>
                    ))}
                    {node.costPerHour > 0 && (
                      <div className="text-[10px] font-mono text-muted-foreground mt-2">Cost: ${node.costPerHour.toFixed(2)}/hr</div>
                    )}
                    {node.status === 'idle' && (
                      <div className="text-xs font-mono text-muted-foreground mt-1">IDLE · $0.00/hr</div>
                    )}
                  </GlassCard>
                ))}
              </div>
            )}
          </div>

          {selected && (
            <div>
              <div className="section-label px-1 mb-3">NODE DETAIL — {selected.name.toUpperCase()}</div>
              <GlassCard>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {selected.gpus.map((gpu, i) => (
                    <div key={i} className="text-center space-y-2">
                      <GaugeRing value={gpu.util} size={80} />
                      <div className="text-xs font-mono text-muted-foreground">{gpu.name}</div>
                      <div className="text-[10px] font-mono text-muted-foreground">
                        VRAM {gpu.vram}/{gpu.vramTotal}GB · {gpu.temp}°C
                      </div>
                    </div>
                  ))}
                </div>
              </GlassCard>
            </div>
          )}

          <div>
            <div className="section-label px-1 mb-3">TELEMETRY</div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <GlassCard>
                <div className="text-xs font-mono text-muted-foreground mb-2">GPU UTILIZATION — 60s</div>
                <FakeChart color="hsl(155,100%,50%)" />
              </GlassCard>
              <GlassCard>
                <div className="text-xs font-mono text-muted-foreground mb-2">VRAM PRESSURE — 60s</div>
                <FakeChart color="hsl(192,100%,50%)" />
              </GlassCard>
            </div>
          </div>
        </div>
      </motion.div>
    </Layout>
  );
};

const GaugeRing = ({ value, size }: { value: number; size: number }) => {
  const sw = 5, r = (size - sw * 2) / 2, c = 2 * Math.PI * r, o = c - (value / 100) * c;
  const color = value >= 90 ? 'hsl(348,100%,50%)' : value >= 75 ? 'hsl(18,100%,57%)' : 'hsl(155,100%,50%)';
  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={sw} />
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={sw}
          strokeDasharray={c} strokeDashoffset={o} strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 800ms cubic-bezier(0.4,0,0.2,1)' }} />
      </svg>
      <span className="absolute text-sm font-mono text-foreground">{value}%</span>
    </div>
  );
};

const FakeChart = ({ color }: { color: string }) => {
  const points = Array.from({ length: 30 }, (_, i) => 50 + Math.sin(i * 0.5) * 20 + Math.random() * 15);
  const max = Math.max(...points);
  const h = 80;
  const w = 100;
  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${(i / (points.length - 1)) * w} ${h - (p / max) * h}`).join(' ');

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-20" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`grad-${color}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${pathD} L ${w} ${h} L 0 ${h} Z`} fill={`url(#grad-${color})`} />
      <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
};

export default Monitoring;
