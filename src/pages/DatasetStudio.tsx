import { useState } from 'react';
import { motion } from 'framer-motion';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import { datasetPacks } from '@/data/mockData';
import { Plus, Package } from 'lucide-react';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const pipelineStages = [
  { id: 'source', label: 'Source', status: 'completed' as const },
  { id: 'synthesis', label: 'Synthesis', status: 'completed' as const },
  { id: 'quality', label: 'Quality Filters', status: 'running' as const },
  { id: 'pack', label: 'Pack', status: 'idle' as const },
];

const DatasetStudio = () => {
  const [selectedPack, setSelectedPack] = useState(datasetPacks[0]);

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate">
        <PageHeader
          title="Dataset Studio"
          actions={
            <div className="flex gap-2">
              <button className="text-xs font-mono px-3 py-1.5 rounded-lg bg-raised border border-border text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5">
                <Plus className="w-3 h-3" /> New Pipeline
              </button>
              <button className="text-xs font-mono px-3 py-1.5 rounded-lg bg-neon-green/20 border border-neon-green/30 text-neon-green hover:bg-neon-green/30 transition-colors flex items-center gap-1.5">
                <Package className="w-3 h-3" /> Pack
              </button>
            </div>
          }
        />
        <div className="p-6 space-y-6">
          {/* Pipeline DAG */}
          <div>
            <div className="section-label px-1 mb-3">PIPELINE DAG</div>
            <GlassCard>
              <div className="flex items-center justify-center gap-0 py-6 overflow-x-auto">
                {pipelineStages.map((stage, i) => (
                  <div key={stage.id} className="flex items-center">
                    <div className={`px-6 py-3 rounded-lg border text-sm font-display font-medium transition-all ${
                      stage.status === 'completed' ? 'border-neon-green/40 text-neon-green bg-neon-green/10' :
                      stage.status === 'running' ? 'border-neon-blue/40 text-neon-blue bg-neon-blue/10 animate-glow-pulse' :
                      'border-border text-muted-foreground bg-raised'
                    }`}>
                      {stage.label}
                    </div>
                    {i < pipelineStages.length - 1 && (
                      <div className="flex items-center mx-2">
                        <div className={`w-8 h-px ${stage.status === 'completed' ? 'bg-neon-green/40' : 'bg-border'}`} />
                        <div className={`w-0 h-0 border-y-[4px] border-y-transparent border-l-[6px] ${
                          stage.status === 'completed' ? 'border-l-neon-green/40' : 'border-l-border'
                        }`} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>

          {/* Registry + Detail */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <div className="section-label px-1 mb-3">DATASET REGISTRY</div>
              <div className="space-y-2">
                {datasetPacks.map(pack => (
                  <GlassCard
                    key={pack.id}
                    hover
                    className={`cursor-pointer ${selectedPack?.id === pack.id ? 'glow-green' : ''}`}
                    onClick={() => setSelectedPack(pack)}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-display font-medium text-foreground">{pack.name}</span>
                      <span className="text-[10px] font-mono bg-raised text-muted-foreground px-2 py-0.5 rounded">{pack.domain}</span>
                    </div>
                    <div className="text-xs font-mono text-muted-foreground mb-2">
                      {pack.samples.toLocaleString()} samples · {pack.createdAt}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-muted-foreground">Quality: {pack.quality}%</span>
                      <div className="flex-1 h-1 bg-raised rounded-full overflow-hidden">
                        <div className="h-full bg-neon-green rounded-full progress-bar-transition" style={{ width: `${pack.quality}%` }} />
                      </div>
                    </div>
                  </GlassCard>
                ))}
              </div>
            </div>

            <div>
              <div className="section-label px-1 mb-3">PACK DETAIL</div>
              {selectedPack && (
                <GlassCard>
                  <div className="text-lg font-display font-semibold text-foreground mb-4">{selectedPack.name}</div>
                  <div className="space-y-3">
                    <DetailRow label="Pack ID" value={selectedPack.id} />
                    <DetailRow label="Samples" value={selectedPack.samples.toLocaleString()} />
                    <DetailRow label="Quality" value={`${selectedPack.quality}%`} />
                    <DetailRow label="Created" value={selectedPack.createdAt} />
                    <div>
                      <div className="text-[10px] font-mono text-muted-foreground mb-2">SOURCE BREAKDOWN</div>
                      {selectedPack.sources.map(s => (
                        <div key={s.name} className="flex items-center justify-between py-1 text-xs font-mono">
                          <span className="text-foreground/80">{s.name}</span>
                          <span className="text-muted-foreground">{s.count.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                    {/* Quality Distribution */}
                    <div>
                      <div className="text-[10px] font-mono text-muted-foreground mb-2">QUALITY DISTRIBUTION</div>
                      <div className="flex items-end gap-[2px] h-12">
                        {Array.from({ length: 20 }, (_, i) => {
                          const h = Math.max(5, Math.random() * (i > 14 ? 100 : i > 10 ? 60 : 20));
                          const hue = i < 6 ? 348 : i < 14 ? 18 : 155;
                          return <div key={i} className="flex-1 rounded-t-sm" style={{ height: `${h}%`, backgroundColor: `hsl(${hue},100%,50%)`, opacity: 0.7 }} />;
                        })}
                      </div>
                    </div>
                    <button className="w-full text-xs font-mono px-3 py-2 rounded-lg bg-neon-blue/20 border border-neon-blue/30 text-neon-blue hover:bg-neon-blue/30 transition-colors">
                      Inspect Samples
                    </button>
                  </div>
                </GlassCard>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </Layout>
  );
};

const DetailRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-center justify-between py-1 border-b border-border">
    <span className="text-[10px] font-mono text-muted-foreground uppercase">{label}</span>
    <span className="text-xs font-mono text-foreground">{value}</span>
  </div>
);

export default DatasetStudio;
