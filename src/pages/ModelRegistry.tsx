import { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import { LoadingSkeleton } from '@/components/factory/LoadingState';
import { models as mockModels } from '@/data/mockData';
import type { ModelEntry } from '@/data/mockData';
import { Grid, List, Rocket, GitBranch } from 'lucide-react';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const ModelRegistry = () => {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filter, setFilter] = useState('');
  const [lineageModelId, setLineageModelId] = useState<string | null>(null);

  const { data: apiModels, isLoading } = useQuery<ModelEntry[]>({
    queryKey: ['/models'],
  });

  const { data: lineageData, isLoading: lineageLoading } = useQuery({
    queryKey: ['/models', lineageModelId, 'lineage'],
    enabled: !!lineageModelId,
  });

  const models = apiModels || mockModels;
  const filtered = models.filter(m => m.name.toLowerCase().includes(filter.toLowerCase()));

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate">
        <PageHeader
          title="Model Registry"
          actions={
            <div className="flex items-center gap-2">
              <input
                value={filter}
                onChange={e => setFilter(e.target.value)}
                placeholder="Filter models..."
                data-testid="input-filter-models"
                className="text-xs font-mono bg-raised border border-border rounded-lg px-3 py-1.5 text-foreground placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-neon-green/30 w-48"
              />
              <div className="flex border border-border rounded-lg overflow-hidden">
                <button
                  onClick={() => setViewMode('grid')}
                  data-testid="button-view-grid"
                  className={`p-1.5 ${viewMode === 'grid' ? 'bg-raised text-foreground' : 'text-muted-foreground'}`}
                >
                  <Grid className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  data-testid="button-view-list"
                  className={`p-1.5 ${viewMode === 'list' ? 'bg-raised text-foreground' : 'text-muted-foreground'}`}
                >
                  <List className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          }
        />
        <div className="p-6">
          {isLoading && !filtered.length ? (
            <LoadingSkeleton rows={4} />
          ) : viewMode === 'grid' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {filtered.map(model => (
                <GlassCard key={model.id} hover data-testid={`card-model-${model.id}`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-display font-semibold text-foreground">{model.name}</span>
                    <span className="text-[10px] font-mono bg-raised text-muted-foreground px-2 py-0.5 rounded">{model.size}</span>
                  </div>
                  <div className="flex gap-2 mb-3">
                    <span className="text-[10px] font-mono bg-neon-blue/20 text-neon-blue px-1.5 py-0.5 rounded">{model.method}</span>
                    <span className="text-[10px] font-mono bg-neon-purple/20 text-neon-purple px-1.5 py-0.5 rounded">{model.stage}</span>
                  </div>
                  <div className="space-y-1 mb-3">
                    {Object.entries(model.scores).map(([k, v]) => (
                      <div key={k} className="flex items-center justify-between text-xs font-mono">
                        <span className="text-muted-foreground">{k}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1 bg-raised rounded-full overflow-hidden">
                            <div className="h-full bg-neon-green rounded-full" style={{ width: `${v}%` }} />
                          </div>
                          <span className="text-foreground w-10 text-right">{v}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="text-[10px] font-mono text-muted-foreground mb-3">
                    {model.children} children · {model.createdAt}
                  </div>
                  <div className="flex gap-2">
                    <button data-testid={`button-deploy-${model.id}`} className="flex-1 text-[10px] font-mono px-2 py-1.5 rounded-lg bg-neon-green/20 border border-neon-green/30 text-neon-green hover:bg-neon-green/30 transition-colors flex items-center justify-center gap-1">
                      <Rocket className="w-3 h-3" /> Deploy
                    </button>
                    <button
                      data-testid={`button-lineage-${model.id}`}
                      onClick={() => setLineageModelId(lineageModelId === model.id ? null : model.id)}
                      className="flex-1 text-[10px] font-mono px-2 py-1.5 rounded-lg bg-raised border border-border text-muted-foreground hover:text-foreground transition-colors flex items-center justify-center gap-1"
                    >
                      <GitBranch className="w-3 h-3" /> Lineage
                    </button>
                  </div>
                  {lineageModelId === model.id && (
                    <div className="mt-3 pt-3 border-t border-border">
                      {lineageLoading ? (
                        <div className="text-xs font-mono text-muted-foreground animate-pulse">Loading lineage...</div>
                      ) : lineageData ? (
                        <div className="text-xs font-mono text-foreground/80">
                          <pre className="whitespace-pre-wrap">{JSON.stringify(lineageData, null, 2)}</pre>
                        </div>
                      ) : (
                        <div className="text-xs font-mono text-muted-foreground">No lineage data available</div>
                      )}
                    </div>
                  )}
                </GlassCard>
              ))}
            </div>
          ) : (
            <GlassCard className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-border">
                    {['Name', 'Base', 'Method', 'Stage', 'MMLU', 'GSM8K', 'HumanEval', 'Size', 'Created'].map(h => (
                      <th key={h} className="text-left py-2 px-3 text-muted-foreground font-normal">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map(m => (
                    <tr key={m.id} className="border-b border-border/50 hover:bg-raised/50 transition-colors" data-testid={`row-model-${m.id}`}>
                      <td className="py-2 px-3 text-foreground font-medium">{m.name}</td>
                      <td className="py-2 px-3 text-muted-foreground">{m.base}</td>
                      <td className="py-2 px-3"><span className="bg-neon-blue/20 text-neon-blue px-1.5 py-0.5 rounded">{m.method}</span></td>
                      <td className="py-2 px-3"><span className="bg-neon-purple/20 text-neon-purple px-1.5 py-0.5 rounded">{m.stage}</span></td>
                      <td className="py-2 px-3">{m.scores.MMLU}%</td>
                      <td className="py-2 px-3">{m.scores.GSM8K}%</td>
                      <td className="py-2 px-3">{m.scores.HumanEval}%</td>
                      <td className="py-2 px-3">{m.size}</td>
                      <td className="py-2 px-3 text-muted-foreground">{m.createdAt}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </GlassCard>
          )}
        </div>
      </motion.div>
    </Layout>
  );
};

export default ModelRegistry;
