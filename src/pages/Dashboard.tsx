import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import StatusDot from '@/components/factory/StatusDot';
import CountUp from '@/components/factory/CountUp';
import { LoadingSkeleton, ErrorState } from '@/components/factory/LoadingState';
import { useWebSocket } from '@/hooks/useWebSocket';
import { trainingJobs as mockJobs, logLines as mockLogs, timelineEntries as mockTimeline } from '@/data/mockData';
import type { TrainingJob, LogLine } from '@/data/mockData';
import { ExternalLink, FileText } from 'lucide-react';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

const Dashboard = () => {
  const { gpuTelemetry, isConnected } = useWebSocket();

  const { data: jobs, isLoading: jobsLoading, error: jobsError, refetch: refetchJobs } = useQuery<TrainingJob[]>({
    queryKey: ['/jobs'],
  });

  const { data: nodes, isLoading: nodesLoading } = useQuery<any[]>({
    queryKey: ['/cluster/nodes'],
  });

  const trainingJobs = jobs || mockJobs;
  const clusterNodes = nodes || [];
  const logLines = mockLogs;
  const timelineEntries = mockTimeline;

  const wsTelemetry = gpuTelemetry as any;

  const allGpus = clusterNodes.flatMap((n: any) =>
    (n.gpus || []).map((g: any, i: number) => ({
      label: `GPU-${i}`,
      util: g.util,
      vram: g.vram,
      temp: g.temp,
    }))
  );
  const defaultGpus = [
    { label: 'GPU-0', util: 91, vram: 74, temp: 79 },
    { label: 'GPU-1', util: 88, vram: 71, temp: 76 },
    { label: 'GPU-2', util: 38, vram: 31, temp: 58 },
    { label: 'GPU-3', util: 82, vram: 18, temp: 71 },
  ];
  const wsGpus = wsTelemetry?.gpus?.map?.((g: any, i: number) => ({
    label: g.label || `GPU-${i}`,
    util: g.util ?? g.utilization ?? 0,
    vram: g.vram ?? g.memory_used ?? 0,
    temp: g.temp ?? g.temperature ?? 0,
  }));
  const gpuList = (wsGpus || (allGpus.length > 0 ? allGpus : null) || defaultGpus).slice(0, 4);

  const activeJobs = trainingJobs.filter(j => j.status === 'running').length;
  const nodeCount = clusterNodes.length || 4;
  const activeNodes = clusterNodes.filter((n: any) => n.status !== 'offline').length || 3;

  const kpis = [
    { label: 'ACTIVE TRAINING JOBS', value: activeJobs || 3, glow: 'green' as const, change: `${activeJobs} running` },
    { label: 'MODELS TRAINED', value: 47, glow: 'blue' as const, change: '↑ 3 today' },
    { label: 'GPU FLEET', value: nodeCount, glow: 'none' as const, change: `${activeNodes}/${nodeCount} nodes active`, suffix: ' nodes' },
    { label: 'DATASET VOLUME', value: 205, glow: 'none' as const, change: '50K new samples', suffix: 'K samples' },
  ];

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate">
        <PageHeader title="Mission Control" subtitle={new Date().toLocaleString()} />

        <div className="p-6 space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" data-testid="kpi-row">
            {kpis.map(kpi => (
              <GlassCard key={kpi.label} glow={kpi.glow} hover>
                <div className="section-label mb-3">{kpi.label}</div>
                <div className="text-3xl font-mono font-bold text-foreground">
                  <CountUp end={kpi.value} suffix={kpi.suffix} />
                </div>
                <div className="text-xs text-muted-foreground font-mono mt-2">{kpi.change}</div>
              </GlassCard>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <div className="lg:col-span-3 space-y-3">
              <div className="section-label px-1">ACTIVE OPERATIONS</div>
              {jobsLoading ? (
                <LoadingSkeleton rows={3} />
              ) : jobsError ? (
                <ErrorState message={(jobsError as Error).message} onRetry={() => refetchJobs()} />
              ) : (
                trainingJobs.filter(j => j.status === 'running' || j.status === 'queued').map(job => (
                  <JobCard key={job.id} job={job} />
                ))
              )}
            </div>

            <div className="lg:col-span-2">
              <div className="section-label px-1 mb-3">
                SYSTEM HEALTH
                {isConnected && <span className="ml-2 text-neon-green text-[10px]">● LIVE</span>}
              </div>
              <GlassCard>
                <div className="grid grid-cols-2 gap-6">
                  {gpuList.map((gpu: any) => (
                    <div key={gpu.label} className="text-center">
                      <GaugeRing value={gpu.util} size={72} />
                      <div className="text-[10px] font-mono text-muted-foreground mt-1">{gpu.label}</div>
                      <div className="text-[10px] font-mono text-muted-foreground">
                        {gpu.vram}GB · {gpu.temp}°C
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 space-y-2">
                  <MetricBar label="Memory Pressure" value={71} color="hsl(192,100%,50%)" />
                  <MetricBar label="Network I/O" value={45} color="hsl(263,100%,77%)" />
                  <MetricBar label="Disk Throughput" value={33} color="hsl(155,100%,50%)" />
                </div>
              </GlassCard>
            </div>
          </div>

          <div>
            <div className="section-label px-1 mb-3">EXPERIMENT TIMELINE — LAST 24H</div>
            <GlassCard>
              <div className="space-y-2">
                {timelineEntries.map(entry => (
                  <div key={entry.id} className="flex items-center gap-3">
                    <span className="text-xs font-mono text-muted-foreground w-20 shrink-0">{entry.name}</span>
                    <div className="flex-1 h-5 bg-raised rounded relative overflow-hidden">
                      <div
                        className="absolute top-0 h-full rounded progress-bar-transition"
                        style={{
                          left: `${entry.start}%`,
                          width: `${entry.end - entry.start}%`,
                          backgroundColor: entry.status === 'completed' ? 'hsl(155,100%,50%)' : entry.status === 'running' ? 'hsl(192,100%,50%)' : 'hsl(348,100%,50%)',
                          opacity: 0.7,
                        }}
                      />
                    </div>
                    <StatusDot status={entry.status} />
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <div className="lg:col-span-2">
              <div className="section-label px-1 mb-3">RECENT MODELS</div>
              <GlassCard className="space-y-3">
                {[
                  { name: 'llama-3.1-8b-math', score: 'MMLU: 61.2%', time: '2h ago' },
                  { name: 'mistral-7b-code', score: 'HumanEval: 52.1%', time: '6h ago' },
                  { name: 'llama-3.1-8b-general', score: 'MMLU: 63.4%', time: '1d ago' },
                ].map(m => (
                  <div key={m.name} className="flex items-center justify-between py-2 border-b border-border last:border-0">
                    <div>
                      <div className="text-sm font-display text-foreground">{m.name}</div>
                      <div className="text-xs font-mono text-muted-foreground">{m.score}</div>
                    </div>
                    <span className="text-[10px] font-mono text-muted-foreground">{m.time}</span>
                  </div>
                ))}
              </GlassCard>
            </div>

            <div className="lg:col-span-3">
              <div className="section-label px-1 mb-3">LIVE LOG STREAM</div>
              <GlassCard className="max-h-64 overflow-y-auto">
                <div className="space-y-0.5">
                  {logLines.map((log: LogLine) => (
                    <div key={log.id} className="flex gap-2 text-xs font-mono py-0.5">
                      <span className="text-muted-foreground shrink-0">{log.timestamp}</span>
                      <span className={`shrink-0 w-14 ${
                        log.level === 'ERROR' ? 'text-neon-red' :
                        log.level === 'WARN' ? 'text-neon-orange' :
                        log.level === 'AGENT' ? 'text-neon-purple' :
                        log.level === 'METRIC' ? 'text-neon-blue' :
                        'text-muted-foreground'
                      }`}>[{log.level}]</span>
                      <span className="text-foreground/80">{log.message}</span>
                    </div>
                  ))}
                </div>
              </GlassCard>
            </div>
          </div>
        </div>
      </motion.div>
    </Layout>
  );
};

const JobCard = ({ job }: { job: TrainingJob }) => (
  <GlassCard glow={job.status === 'running' ? 'green' : 'none'} hover data-testid={`card-job-${job.id}`}>
    <div className="flex items-center gap-2 mb-2">
      <StatusDot status={job.status} size="md" />
      <span className="text-xs font-mono uppercase text-neon-green">{job.status}</span>
      <span className="text-[10px] font-mono bg-raised text-muted-foreground px-2 py-0.5 rounded">{job.type}</span>
      <span className="text-sm font-display text-foreground ml-auto">{job.model}</span>
    </div>
    <div className="text-xs font-mono text-muted-foreground mb-2">{job.name}</div>
    {job.status === 'running' && (
      <>
        <div className="flex gap-6 text-xs font-mono mb-2">
          <span>Loss: <span className="text-foreground">{job.loss}</span> <span className={job.lossChange < 0 ? 'text-neon-green' : 'text-neon-red'}>{job.lossChange < 0 ? '▼' : '▲'}{Math.abs(job.lossChange)}</span></span>
          <span>Step: <span className="text-foreground">{job.step.toLocaleString()}/{job.totalSteps.toLocaleString()}</span></span>
        </div>
        <div className="flex items-center gap-3 mb-2">
          <div className="flex-1 h-2 bg-raised rounded-full overflow-hidden">
            <div className="h-full bg-neon-green rounded-full progress-bar-transition" style={{ width: `${job.progress}%` }} />
          </div>
          <span className="text-xs font-mono text-foreground">{job.progress}%</span>
          <span className="text-xs font-mono text-muted-foreground">ETA: {job.eta}</span>
        </div>
        <div className="flex items-center justify-between">
          <div className="text-[10px] font-mono text-muted-foreground">
            {job.gpuUtil.map((u, i) => `GPU-${i}: ${u}%`).join(' · ')} VRAM: {job.vram}
          </div>
          <div className="flex gap-2">
            <button data-testid={`button-stop-${job.id}`} className="text-[10px] font-mono px-2 py-1 rounded bg-neon-red/20 text-neon-red border border-neon-red/30 hover:bg-neon-red/30 transition-colors">■ Stop</button>
            <button data-testid={`button-details-${job.id}`} className="text-[10px] font-mono px-2 py-1 rounded bg-raised text-muted-foreground border border-border hover:bg-overlay transition-colors flex items-center gap-1">
              <ExternalLink className="w-3 h-3" /> Details
            </button>
            <button data-testid={`button-logs-${job.id}`} className="text-[10px] font-mono px-2 py-1 rounded bg-raised text-muted-foreground border border-border hover:bg-overlay transition-colors flex items-center gap-1">
              <FileText className="w-3 h-3" /> Logs
            </button>
          </div>
        </div>
      </>
    )}
    {job.status === 'queued' && (
      <div className="text-xs font-mono text-muted-foreground">{job.startedAt}</div>
    )}
  </GlassCard>
);

const GaugeRing = ({ value, size }: { value: number; size: number }) => {
  const strokeWidth = 4;
  const r = (size - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (value / 100) * circumference;
  const color = value >= 90 ? 'hsl(348,100%,50%)' : value >= 75 ? 'hsl(18,100%,57%)' : 'hsl(155,100%,50%)';

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth} />
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={strokeWidth}
          strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 800ms cubic-bezier(0.4,0,0.2,1)' }} />
      </svg>
      <span className="absolute text-xs font-mono text-foreground">{value}%</span>
    </div>
  );
};

const MetricBar = ({ label, value, color }: { label: string; value: number; color: string }) => (
  <div className="flex items-center gap-2">
    <span className="text-[10px] font-mono text-muted-foreground w-28 shrink-0">{label}</span>
    <div className="flex-1 h-1.5 bg-raised rounded-full overflow-hidden">
      <div className="h-full rounded-full progress-bar-transition" style={{ width: `${value}%`, backgroundColor: color }} />
    </div>
    <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{value}%</span>
  </div>
);

export default Dashboard;
