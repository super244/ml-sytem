import { useState } from 'react';
import { motion } from 'framer-motion';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import { getApiBaseUrl, setApiBaseUrl } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { Save } from 'lucide-react';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const SettingsPage = () => {
  const { toast } = useToast();
  const [apiBase, setApiBase] = useState(getApiBaseUrl());
  const [wsEndpoint, setWsEndpoint] = useState(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/telemetry`;
  });

  const handleSaveApiBase = () => {
    setApiBaseUrl(apiBase);
    toast({
      title: 'Settings saved',
      description: `API Base URL updated to ${apiBase}`,
    });
  };

  const settings = [
    { label: 'Auto-prune Threshold', value: 'Δloss < 0.001 for 500 steps', desc: 'When agents should prune stalled runs', editable: false },
    { label: 'Promotion Score', value: '0.65 composite minimum', desc: 'Minimum score to promote to next stage', editable: false },
    { label: 'GPU Temp Warning', value: '80°C', desc: 'Temperature threshold for warning state', editable: false },
    { label: 'GPU Temp Critical', value: '90°C', desc: 'Temperature threshold for critical state', editable: false },
  ];

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate">
        <PageHeader title="Settings" subtitle="System configuration" />
        <div className="p-6 space-y-4 max-w-2xl">
          <GlassCard hover>
            <div className="flex items-center justify-between gap-4">
              <div className="flex-1">
                <div className="text-sm font-display font-medium text-foreground">API Base URL</div>
                <div className="text-[10px] font-mono text-muted-foreground mt-0.5">Backend API endpoint</div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  value={apiBase}
                  onChange={e => setApiBase(e.target.value)}
                  data-testid="input-api-base-url"
                  className="text-xs font-mono text-foreground bg-raised px-3 py-1.5 rounded-lg border border-border outline-none focus:ring-1 focus:ring-neon-green/30 w-64"
                />
                <button
                  onClick={handleSaveApiBase}
                  data-testid="button-save-api-url"
                  className="text-xs font-mono px-3 py-1.5 rounded-lg bg-neon-green/20 border border-neon-green/30 text-neon-green hover:bg-neon-green/30 transition-colors flex items-center gap-1"
                >
                  <Save className="w-3 h-3" /> Save
                </button>
              </div>
            </div>
          </GlassCard>

          <GlassCard hover>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-display font-medium text-foreground">Cluster WebSocket</div>
                <div className="text-[10px] font-mono text-muted-foreground mt-0.5">Real-time telemetry endpoint</div>
              </div>
              <div className="text-xs font-mono text-foreground bg-raised px-3 py-1.5 rounded-lg border border-border" data-testid="text-ws-endpoint">
                {wsEndpoint}
              </div>
            </div>
          </GlassCard>

          {settings.map(setting => (
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
};

export default SettingsPage;
