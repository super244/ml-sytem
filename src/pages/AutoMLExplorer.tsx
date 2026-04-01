import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import { LoadingSkeleton } from '@/components/factory/LoadingState';
import { automlRuns as mockRuns } from '@/data/mockData';
import type { AutoMLRun } from '@/data/mockData';
import { Plus, Play } from 'lucide-react';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const AutoMLExplorer = () => {
  const [hoveredRun, setHoveredRun] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<'loss' | 'lr' | 'rank'>('loss');

  const { data: apiSearches, isLoading } = useQuery<any>({
    queryKey: ['/automl/searches'],
  });

  const automlRuns: AutoMLRun[] = apiSearches?.runs || apiSearches || mockRuns;

  const bestRun = useMemo(
    () => automlRuns.filter(r => r.status === 'promoted').sort((a, b) => a.loss - b.loss)[0],
    [automlRuns]
  );

  const sortedRuns = useMemo(
    () => [...automlRuns].sort((a, b) => {
      if (sortKey === 'loss') return a.loss - b.loss;
      if (sortKey === 'lr') return a.lr - b.lr;
      return a.rank - b.rank;
    }),
    [automlRuns, sortKey]
  );

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate">
        <PageHeader
          title="AutoML Search Tree"
          actions={
            <div className="flex gap-2">
              <button data-testid="button-new-search" className="text-xs font-mono px-3 py-1.5 rounded-lg bg-raised border border-border text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5">
                <Plus className="w-3 h-3" /> New Search
              </button>
              <button data-testid="button-resume" className="text-xs font-mono px-3 py-1.5 rounded-lg bg-neon-green/20 border border-neon-green/30 text-neon-green hover:bg-neon-green/30 transition-colors flex items-center gap-1.5">
                <Play className="w-3 h-3" /> Resume
              </button>
            </div>
          }
        />
        <div className="p-6 space-y-6">
          {isLoading && !automlRuns.length ? (
            <LoadingSkeleton rows={4} />
          ) : (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
                <div className="lg:col-span-3">
                  <div className="section-label px-1 mb-3">SEARCH SPACE</div>
                  <GlassCard className="relative h-80">
                    <svg className="w-full h-full" viewBox="0 0 400 250">
                      <line x1="40" y1="230" x2="390" y2="230" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                      <line x1="40" y1="10" x2="40" y2="230" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                      <text x="215" y="248" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="9" fontFamily="JetBrains Mono">Learning Rate</text>
                      <text x="12" y="120" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="9" fontFamily="JetBrains Mono" transform="rotate(-90,12,120)">Eval Loss</text>

                      {automlRuns.map(run => {
                        const x = 40 + (Math.log10(run.lr) + 5) / 2 * 350;
                        const y = 10 + (run.loss / 1) * 220;
                        const r = Math.min(8, Math.max(3, run.steps / 2000));
                        const color = run.status === 'promoted' ? 'hsl(155,100%,50%)' :
                          run.status === 'running' ? 'hsl(192,100%,50%)' :
                          run.status === 'pruned' ? 'rgba(255,255,255,0.2)' : 'hsl(155,100%,50%)';

                        return (
                          <g key={run.id} onMouseEnter={() => setHoveredRun(run.id)} onMouseLeave={() => setHoveredRun(null)}>
                            {run.status === 'promoted' ? (
                              <polygon
                                points={`${x},${y-r-2} ${x-r-1},${y+r} ${x+r+1},${y+r}`}
                                fill={color} fillOpacity={hoveredRun === run.id ? 1 : 0.7}
                                stroke={color} strokeWidth="1"
                              />
                            ) : (
                              <circle cx={x} cy={y} r={r} fill={color} fillOpacity={hoveredRun === run.id ? 1 : 0.5}
                                stroke={hoveredRun === run.id ? color : 'none'} strokeWidth="2" />
                            )}
                            {hoveredRun === run.id && (
                              <g>
                                <rect x={x + 10} y={y - 30} width="110" height="50" rx="4" fill="hsl(240,24%,10%)" stroke="rgba(255,255,255,0.15)" />
                                <text x={x + 16} y={y - 14} fill="rgba(255,255,255,0.9)" fontSize="8" fontFamily="JetBrains Mono">lr: {run.lr.toExponential(1)}</text>
                                <text x={x + 16} y={y - 2} fill="rgba(255,255,255,0.9)" fontSize="8" fontFamily="JetBrains Mono">rank: {run.rank} α: {run.alpha}</text>
                                <text x={x + 16} y={y + 10} fill="rgba(255,255,255,0.9)" fontSize="8" fontFamily="JetBrains Mono">loss: {run.loss}</text>
                              </g>
                            )}
                          </g>
                        );
                      })}
                    </svg>
                    <div className="absolute bottom-3 right-3 flex gap-3 text-[10px] font-mono text-muted-foreground">
                      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-neon-green inline-block" /> completed</span>
                      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-neon-blue inline-block" /> running</span>
                      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-foreground/20 inline-block" /> pruned</span>
                      <span className="flex items-center gap-1"><span className="w-0 h-0 border-x-[4px] border-x-transparent border-b-[6px] border-b-neon-green inline-block" /> promoted</span>
                    </div>
                  </GlassCard>
                </div>

                <div className="lg:col-span-2">
                  <div className="section-label px-1 mb-3">SEARCH SUMMARY</div>
                  <GlassCard className="space-y-3">
                    <DetailRow label="Search ID" value={apiSearches?.id || "search-bayesian-001"} />
                    <DetailRow label="Strategy" value={apiSearches?.strategy || "Bayesian Optimization"} />
                    <DetailRow label="Total Runs" value={`${automlRuns.length}/100`} />
                    <DetailRow label="Pruned" value={String(automlRuns.filter(r => r.status === 'pruned').length)} />
                    <DetailRow label="Promoted" value={String(automlRuns.filter(r => r.status === 'promoted').length)} />

                    <div className="pt-3 border-t border-border">
                      <div className="text-[10px] font-mono text-muted-foreground mb-2 uppercase">Best Config</div>
                      {bestRun && (
                        <div className="space-y-1">
                          <DetailRow label="lr" value={bestRun.lr.toExponential(1)} />
                          <DetailRow label="rank" value={String(bestRun.rank)} />
                          <DetailRow label="alpha" value={String(bestRun.alpha)} />
                          <DetailRow label="loss" value={String(bestRun.loss)} />
                        </div>
                      )}
                    </div>

                    <button data-testid="button-promote" className="w-full text-xs font-mono px-3 py-2 rounded-lg bg-neon-green/20 border border-neon-green/30 text-neon-green hover:bg-neon-green/30 transition-colors">
                      → Promote Now
                    </button>
                  </GlassCard>
                </div>
              </div>

              <div>
                <div className="section-label px-1 mb-3">RUN TABLE</div>
                <GlassCard className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="border-b border-border">
                        {['ID', 'LR', 'Rank', 'Alpha', 'Loss', 'Status', 'Duration', 'Steps'].map(h => (
                          <th key={h} className="text-left py-2 px-3 text-muted-foreground font-normal cursor-pointer hover:text-foreground"
                            onClick={() => h.toLowerCase() === 'loss' || h.toLowerCase() === 'lr' || h.toLowerCase() === 'rank' ? setSortKey(h.toLowerCase() as any) : null}>
                            {h} {sortKey === h.toLowerCase() && '↓'}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sortedRuns.slice(0, 15).map(run => (
                        <tr key={run.id} className="border-b border-border/50 hover:bg-raised/50 transition-colors" data-testid={`row-run-${run.id}`}>
                          <td className="py-2 px-3 text-foreground">{run.id}</td>
                          <td className="py-2 px-3">{run.lr.toExponential(1)}</td>
                          <td className="py-2 px-3">{run.rank}</td>
                          <td className="py-2 px-3">{run.alpha}</td>
                          <td className="py-2 px-3 text-foreground">{run.loss}</td>
                          <td className="py-2 px-3">
                            <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                              run.status === 'promoted' ? 'bg-neon-green/20 text-neon-green' :
                              run.status === 'running' ? 'bg-neon-blue/20 text-neon-blue' :
                              run.status === 'pruned' ? 'bg-raised text-muted-foreground' :
                              'bg-neon-green/10 text-neon-green/70'
                            }`}>{run.status}</span>
                          </td>
                          <td className="py-2 px-3">{run.duration}</td>
                          <td className="py-2 px-3">{run.steps.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </GlassCard>
              </div>
            </>
          )}
        </div>
      </motion.div>
    </Layout>
  );
};

const DetailRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-center justify-between py-1">
    <span className="text-[10px] font-mono text-muted-foreground uppercase">{label}</span>
    <span className="text-xs font-mono text-foreground">{value}</span>
  </div>
);

export default AutoMLExplorer;
