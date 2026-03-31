import { motion } from 'framer-motion';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const SettingsPage = () => (
  <Layout>
    <motion.div variants={pageVariants} initial="initial" animate="animate">
      <PageHeader title="Settings" subtitle="System configuration" />
      <div className="p-6 space-y-4 max-w-2xl">
        {[
          { label: 'Cluster WebSocket', value: 'ws://localhost:8000/ws/telemetry', desc: 'Real-time telemetry endpoint' },
          { label: 'API Base URL', value: 'http://localhost:8000/api/v1', desc: 'Backend API endpoint' },
          { label: 'Auto-prune Threshold', value: 'Δloss < 0.001 for 500 steps', desc: 'When agents should prune stalled runs' },
          { label: 'Promotion Score', value: '0.65 composite minimum', desc: 'Minimum score to promote to next stage' },
          { label: 'GPU Temp Warning', value: '80°C', desc: 'Temperature threshold for warning state' },
          { label: 'GPU Temp Critical', value: '90°C', desc: 'Temperature threshold for critical state' },
        ].map(setting => (
          <GlassCard key={setting.label} hover>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-display font-medium text-foreground">{setting.label}</div>
                <div className="text-[10px] font-mono text-muted-foreground mt-0.5">{setting.desc}</div>
              </div>
              <div className="text-xs font-mono text-foreground bg-raised px-3 py-1.5 rounded-lg border border-border">
                {setting.value}
              </div>
            </div>
          </GlassCard>
        ))}
      </div>
    </motion.div>
  </Layout>
);

export default SettingsPage;
