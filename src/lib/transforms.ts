import type { TrainingJob, ClusterNode, ModelEntry, DatasetPack, AutoMLRun, AgentDecision } from '@/data/mockData';

function formatEta(seconds: number | null | undefined): string {
  if (!seconds || seconds <= 0) return 'N/A';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}

function formatTimeAgo(isoString: string | null | undefined): string {
  if (!isoString) return '';
  const diff = Date.now() - new Date(isoString).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function transformJob(raw: any): TrainingJob {
  return {
    id: raw.id,
    name: raw.name,
    type: raw.type?.replace('_', ' ').toUpperCase() || raw.type,
    model: raw.base_model || '',
    status: raw.status === 'stopped' ? 'failed' : raw.status,
    progress: Math.round((raw.progress || 0) * 100),
    loss: raw.current_loss ?? 0,
    lossChange: raw.loss_delta ?? 0,
    step: raw.current_step || 0,
    totalSteps: raw.total_steps || 0,
    eta: formatEta(raw.eta_seconds),
    gpuUtil: raw.gpu_utilization || [],
    vram: `${Math.round(raw.vram_used_gb || 0)}/${Math.round(raw.vram_total_gb || 0)}GB`,
    startedAt: raw.started_at ? new Date(raw.started_at).toLocaleString() : '',
  };
}

export function transformJobs(raw: any[]): TrainingJob[] {
  return (raw || []).map(transformJob);
}

function mapNodeStatus(status: string): ClusterNode['status'] {
  switch (status) {
    case 'online': return 'running';
    case 'offline': return 'offline';
    case 'degraded': return 'warning';
    case 'maintenance': return 'idle';
    default: return 'idle';
  }
}

export function transformNode(raw: any): ClusterNode {
  return {
    id: raw.id,
    name: raw.name,
    status: mapNodeStatus(raw.status),
    gpus: (raw.gpus || []).map((g: any) => ({
      name: g.name,
      util: Math.round(g.utilization || 0),
      vram: Math.round(g.vram_used_gb || 0),
      vramTotal: Math.round(g.vram_total_gb || 0),
      temp: Math.round(g.temperature_celsius || 0),
    })),
    costPerHour: raw.cost_per_hour ?? 0,
  };
}

export function transformNodes(raw: any[]): ClusterNode[] {
  return (raw || []).map(transformNode);
}

export function transformModel(raw: any): ModelEntry {
  const typeMap: Record<string, string> = {
    lora: 'LoRA',
    dpo: 'DPO',
    rlhf: 'RLHF',
    full_finetune: 'Full FT',
  };
  const stageMap: Record<string, string> = {
    lora: 'SFT',
    dpo: 'DPO',
    rlhf: 'RLHF',
    full_finetune: 'SFT',
  };
  return {
    id: raw.id,
    name: raw.name,
    base: raw.base_model || '',
    method: typeMap[raw.training_type] || raw.training_type || '',
    stage: stageMap[raw.training_type] || 'SFT',
    scores: raw.eval_scores || {},
    children: raw.children_count || 0,
    createdAt: raw.created_at ? formatTimeAgo(raw.created_at) : '',
    size: raw.size_gb ? `${raw.size_gb.toFixed(1)} GB` : '0 GB',
  };
}

export function transformModels(raw: any[]): ModelEntry[] {
  return (raw || []).map(transformModel);
}

export function transformDataset(raw: any): DatasetPack {
  const ps = raw.pack_summary;
  const sources = ps?.source_distribution
    ? Object.entries(ps.source_distribution).map(([name, pct]) => ({
        name,
        count: Math.round((pct as number) * (raw.sample_count || 0)),
      }))
    : [{ name: raw.domain || 'unknown', count: raw.sample_count || 0 }];

  return {
    id: raw.id,
    name: raw.name,
    domain: raw.domain || '',
    samples: raw.sample_count || 0,
    quality: raw.quality_score_mean || 0,
    createdAt: raw.created_at ? formatTimeAgo(raw.created_at) : '',
    sources,
  };
}

export function transformDatasets(raw: any[]): DatasetPack[] {
  return (raw || []).map(transformDataset);
}

export function transformAutoMLRun(raw: any): AutoMLRun {
  const hp = raw.hyperparams || {};
  return {
    id: raw.id,
    lr: hp.lr || hp.learning_rate || 0,
    rank: hp.rank || 0,
    alpha: hp.alpha || 0,
    loss: raw.eval_loss ?? raw.composite_score ?? 0,
    status: raw.status,
    duration: raw.training_minutes ? `${Math.round(raw.training_minutes)}m` : 'N/A',
    steps: raw.step_pruned || 0,
  };
}

export function transformAutoMLSearchToRuns(searchesRaw: any[]): AutoMLRun[] {
  if (!searchesRaw || searchesRaw.length === 0) return [];
  const allRuns: AutoMLRun[] = [];
  for (const search of searchesRaw) {
    if (search.runs) {
      allRuns.push(...search.runs.map(transformAutoMLRun));
    }
  }
  return allRuns;
}

export function transformAgentDecision(raw: any): AgentDecision {
  return {
    id: raw.id,
    agentId: raw.agent_id,
    agentType: raw.agent_type,
    action: raw.action?.toUpperCase() || raw.action,
    targetJobId: raw.target_id,
    reasoning: raw.reasoning || '',
    timestamp: raw.timestamp ? formatTimeAgo(typeof raw.timestamp === 'number' ? new Date(raw.timestamp * 1000).toISOString() : raw.timestamp) : '',
  };
}

export function transformAgentDecisions(raw: any[]): AgentDecision[] {
  return (raw || []).map(transformAgentDecision);
}
