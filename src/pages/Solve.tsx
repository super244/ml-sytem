import { motion } from 'framer-motion';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const Solve = () => (
  <Layout>
    <motion.div variants={pageVariants} initial="initial" animate="animate">
      <PageHeader title="Solve Workspace" subtitle="Interactive problem solving environment" />
      <div className="p-6">
        <GlassCard className="flex flex-col items-center justify-center py-20 text-center">
          <div className="text-3xl font-display font-bold text-foreground mb-2">Solve Workspace</div>
          <div className="text-sm font-mono text-muted-foreground max-w-md">
            Interactive environment for testing hypotheses, running quick experiments, and exploring model behavior with live feedback loops.
          </div>
          <div className="mt-6 flex gap-3">
            <button className="text-xs font-mono px-4 py-2 rounded-lg bg-neon-green/20 border border-neon-green/30 text-neon-green hover:bg-neon-green/30 transition-colors">
              Start Session
            </button>
            <button className="text-xs font-mono px-4 py-2 rounded-lg bg-raised border border-border text-muted-foreground hover:text-foreground transition-colors">
              Load Template
            </button>
          </div>
        </GlassCard>
      </div>
    </motion.div>
  </Layout>
);

export default Solve;
