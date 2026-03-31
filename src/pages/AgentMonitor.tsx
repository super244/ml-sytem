import { useState } from 'react';
import { motion } from 'framer-motion';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import StatusDot from '@/components/factory/StatusDot';
import { agentDecisions } from '@/data/mockData';
import { Pause } from 'lucide-react';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const agents = [
  { id: 'evaluator-01', type: 'EVALUATOR', status: 'running' as const, detail: 'Monitoring 3 runs', lastAction: '2s ago' },
  { id: 'pruner-01', type: 'PRUNER', status: 'running' as const, detail: 'Pruned 14 runs today', lastAction: '4m ago' },
  { id: 'promoter-01', type: 'PROMOTER', status: 'idle' as const, detail: 'Waiting for eval completion', lastAction: '23m ago' },
];

const AgentMonitor = () => {
  const [selectedDecision, setSelectedDecision] = useState(agentDecisions[0]);

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate">
        <PageHeader
          title="Agent Swarm"
          subtitle="3 agents active"
          actions={
            <button className="text-xs font-mono px-3 py-1.5 rounded-lg bg-neon-orange/20 border border-neon-orange/30 text-neon-orange hover:bg-neon-orange/30 transition-colors flex items-center gap-1.5">
              <Pause className="w-3 h-3" /> Pause All
            </button>
          }
        />
        <div className="p-6 space-y-6">
          {/* Agent Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {agents.map(agent => (
              <GlassCard key={agent.id} glow={agent.status === 'running' ? 'purple' : 'none'} hover>
                <div className="flex items-center gap-2 mb-2">
                  <StatusDot status={agent.status} size="md" />
                  <span className="text-sm font-display font-medium text-foreground">{agent.type}</span>
                  <span className="text-[10px] font-mono text-neon-purple ml-auto uppercase">{agent.status === 'running' ? 'ACTIVE' : 'STANDBY'}</span>
                </div>
                <div className="text-xs font-mono text-muted-foreground">{agent.detail}</div>
                <div className="text-[10px] font-mono text-muted-foreground/60 mt-1">Last action: {agent.lastAction}</div>
              </GlassCard>
            ))}
          </div>

          {/* Decision Feed + Detail */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <div className="lg:col-span-3">
              <div className="section-label px-1 mb-3">DECISION FEED</div>
              <div className="space-y-2">
                {agentDecisions.map(dec => (
                  <GlassCard
                    key={dec.id}
                    hover
                    className={`cursor-pointer ${selectedDecision?.id === dec.id ? 'glow-purple' : ''}`}
                    onClick={() => setSelectedDecision(dec)}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                        dec.agentType === 'pruner' ? 'bg-neon-orange/20 text-neon-orange' :
                        dec.agentType === 'evaluator' ? 'bg-neon-blue/20 text-neon-blue' :
                        'bg-neon-green/20 text-neon-green'
                      }`}>{dec.agentType.toUpperCase()}</span>
                      <span className="text-xs font-mono font-medium text-foreground">{dec.action}</span>
                      <span className="text-xs font-mono text-muted-foreground">{dec.targetJobId}</span>
                      <span className="text-[10px] font-mono text-muted-foreground/60 ml-auto">{dec.timestamp}</span>
                    </div>
                    <div className="text-xs font-mono text-muted-foreground leading-relaxed">{dec.reasoning}</div>
                  </GlassCard>
                ))}
              </div>
            </div>

            <div className="lg:col-span-2">
              <div className="section-label px-1 mb-3">DECISION DETAIL</div>
              {selectedDecision && (
                <GlassCard glow="purple">
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-mono px-2 py-1 rounded ${
                        selectedDecision.agentType === 'pruner' ? 'bg-neon-orange/20 text-neon-orange' :
                        selectedDecision.agentType === 'evaluator' ? 'bg-neon-blue/20 text-neon-blue' :
                        'bg-neon-green/20 text-neon-green'
                      }`}>{selectedDecision.agentType.toUpperCase()}</span>
                      <span className="text-sm font-display font-semibold text-foreground">{selectedDecision.action}</span>
                    </div>
                    <div className="text-xs font-mono text-muted-foreground">{selectedDecision.targetJobId}</div>
                    <div className="border-t border-border pt-3">
                      <div className="text-[10px] font-mono text-muted-foreground uppercase mb-1">Reasoning</div>
                      <div className="text-sm font-mono text-foreground/80 leading-relaxed">{selectedDecision.reasoning}</div>
                    </div>
                    <div className="text-[10px] font-mono text-muted-foreground/60">{selectedDecision.timestamp}</div>
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

export default AgentMonitor;
