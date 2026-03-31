// ===== MOCK DATA FOR AI-FACTORY =====

export interface TrainingJob {
  id: string;
  name: string;
  type: string;
  model: string;
  status: 'running' | 'completed' | 'failed' | 'queued' | 'degraded';
  progress: number;
  loss: number;
  lossChange: number;
  step: number;
  totalSteps: number;
  eta: string;
  gpuUtil: number[];
  vram: string;
  startedAt: string;
}

export interface ClusterNode {
  id: string;
  name: string;
  status: 'running' | 'idle' | 'warning' | 'critical' | 'offline';
  gpus: { name: string; util: number; vram: number; vramTotal: number; temp: number }[];
  costPerHour: number;
}

export interface AgentDecision {
  id: string;
  agentId: string;
  agentType: 'evaluator' | 'pruner' | 'promoter';
  action: string;
  targetJobId: string;
  reasoning: string;
  timestamp: string;
}

export interface ModelEntry {
  id: string;
  name: string;
  base: string;
  method: string;
  stage: string;
  scores: Record<string, number>;
  children: number;
  createdAt: string;
  size: string;
}

export interface DatasetPack {
  id: string;
  name: string;
  domain: string;
  samples: number;
  quality: number;
  createdAt: string;
  sources: { name: string; count: number }[];
}

export interface AutoMLRun {
  id: string;
  lr: number;
  rank: number;
  alpha: number;
  loss: number;
  status: 'completed' | 'running' | 'pruned' | 'promoted';
  duration: string;
  steps: number;
}

export interface LogLine {
  id: string;
  level: 'INFO' | 'WARN' | 'ERROR' | 'AGENT' | 'METRIC';
  message: string;
  source: string;
  timestamp: string;
}

// ===== DATA =====

export const trainingJobs: TrainingJob[] = [
  {
    id: 'job-2847',
    name: 'run-2847-bayesian-lr-search',
    type: 'LORA FINETUNE',
    model: 'llama-3.1-8b',
    status: 'running',
    progress: 48,
    loss: 0.3421,
    lossChange: -0.0012,
    step: 4821,
    totalSteps: 10000,
    eta: '23m 14s',
    gpuUtil: [91, 88],
    vram: '74/80GB',
    startedAt: '2h 14m ago',
  },
  {
    id: 'job-2846',
    name: 'run-2846-dpo-alignment',
    type: 'DPO TRAINING',
    model: 'mistral-7b',
    status: 'running',
    progress: 72,
    loss: 0.1893,
    lossChange: -0.0034,
    step: 7200,
    totalSteps: 10000,
    eta: '11m 42s',
    gpuUtil: [85],
    vram: '22/24GB',
    startedAt: '3h 01m ago',
  },
  {
    id: 'job-2845',
    name: 'run-2845-math-sft',
    type: 'SFT',
    model: 'llama-3.1-8b',
    status: 'running',
    progress: 91,
    loss: 0.2104,
    lossChange: -0.0002,
    step: 9100,
    totalSteps: 10000,
    eta: '4m 23s',
    gpuUtil: [78, 82],
    vram: '71/80GB',
    startedAt: '5h 33m ago',
  },
  {
    id: 'job-2844',
    name: 'run-2844-code-lora',
    type: 'LORA FINETUNE',
    model: 'codellama-13b',
    status: 'queued',
    progress: 0,
    loss: 0,
    lossChange: 0,
    step: 0,
    totalSteps: 15000,
    eta: 'Queued',
    gpuUtil: [],
    vram: '-',
    startedAt: 'Queued 12m ago',
  },
];

export const clusterNodes: ClusterNode[] = [
  {
    id: 'node-1',
    name: 'mac-studio-01',
    status: 'running',
    gpus: [{ name: 'M2 Ultra', util: 82, vram: 18, vramTotal: 24, temp: 71 }],
    costPerHour: 0,
  },
  {
    id: 'node-2',
    name: 'linux-rig-01',
    status: 'warning',
    gpus: [
      { name: 'A100 80GB', util: 97, vram: 76, vramTotal: 80, temp: 83 },
      { name: 'A100 80GB', util: 91, vram: 74, vramTotal: 80, temp: 79 },
    ],
    costPerHour: 0,
  },
  {
    id: 'node-3',
    name: 'ec2-p4d-01',
    status: 'idle',
    gpus: [{ name: 'A100 40GB', util: 0, vram: 0, vramTotal: 40, temp: 34 }],
    costPerHour: 0,
  },
  {
    id: 'node-4',
    name: 'lambda-A100-02',
    status: 'running',
    gpus: [{ name: 'A100 80GB', util: 38, vram: 31, vramTotal: 80, temp: 58 }],
    costPerHour: 2.14,
  },
];

export const agentDecisions: AgentDecision[] = [
  { id: 'dec-1', agentId: 'pruner-01', agentType: 'pruner', action: 'PRUNED', targetJobId: 'run-2841', reasoning: 'Loss plateau detected at 2000 steps. Δloss < 0.001 for 500 steps. Killing to free resources.', timestamp: '14m ago' },
  { id: 'dec-2', agentId: 'evaluator-01', agentType: 'evaluator', action: 'SCORED', targetJobId: 'run-2839', reasoning: 'MMLU: 61.2% · HumanEval: 43.8% · GSM8K: 78.4% → Composite: 0.71', timestamp: '22m ago' },
  { id: 'dec-3', agentId: 'promoter-01', agentType: 'promoter', action: 'PROMOTED', targetJobId: 'run-2839', reasoning: 'Composite score 0.71 exceeds threshold 0.65. Promoting to DPO stage.', timestamp: '23m ago' },
  { id: 'dec-4', agentId: 'evaluator-01', agentType: 'evaluator', action: 'SCORED', targetJobId: 'run-2838', reasoning: 'MMLU: 54.1% · HumanEval: 31.2% · GSM8K: 62.8% → Composite: 0.52', timestamp: '1h ago' },
  { id: 'dec-5', agentId: 'pruner-01', agentType: 'pruner', action: 'PRUNED', targetJobId: 'run-2836', reasoning: 'Loss divergence detected. Current loss 2.41 exceeds baseline 0.89 by 170%.', timestamp: '2h ago' },
  { id: 'dec-6', agentId: 'evaluator-01', agentType: 'evaluator', action: 'SCORED', targetJobId: 'run-2835', reasoning: 'MMLU: 58.8% · HumanEval: 52.1% · GSM8K: 71.2% → Composite: 0.64', timestamp: '3h ago' },
];

export const models: ModelEntry[] = [
  { id: 'mdl-1', name: 'llama-3.1-8b-math', base: 'llama-3.1-8b', method: 'LoRA', stage: 'DPO', scores: { MMLU: 61.2, GSM8K: 78.4, HumanEval: 43.8 }, children: 2, createdAt: '2h ago', size: '4.2GB' },
  { id: 'mdl-2', name: 'mistral-7b-code', base: 'mistral-7b', method: 'LoRA', stage: 'SFT', scores: { MMLU: 58.8, HumanEval: 52.1, GSM8K: 71.2 }, children: 0, createdAt: '6h ago', size: '3.8GB' },
  { id: 'mdl-3', name: 'llama-3.1-8b-general', base: 'llama-3.1-8b', method: 'Full FT', stage: 'SFT', scores: { MMLU: 63.4, GSM8K: 69.1, HumanEval: 38.2 }, children: 1, createdAt: '1d ago', size: '15.2GB' },
  { id: 'mdl-4', name: 'codellama-13b-instruct', base: 'codellama-13b', method: 'LoRA', stage: 'DPO', scores: { MMLU: 55.1, HumanEval: 67.3, GSM8K: 44.8 }, children: 0, createdAt: '2d ago', size: '7.1GB' },
];

export const datasetPacks: DatasetPack[] = [
  { id: 'pack-a1b2c3d4', name: 'math-reasoning-v3', domain: 'math', samples: 50000, quality: 98.2, createdAt: '2h ago', sources: [{ name: 'GSM8K-synth', count: 20000 }, { name: 'MATH-aug', count: 18000 }, { name: 'custom-cot', count: 12000 }] },
  { id: 'pack-e5f6g7h8', name: 'code-instruct-v2', domain: 'code', samples: 35000, quality: 95.7, createdAt: '1d ago', sources: [{ name: 'HumanEval-synth', count: 15000 }, { name: 'MBPP-aug', count: 12000 }, { name: 'custom', count: 8000 }] },
  { id: 'pack-i9j0k1l2', name: 'general-chat-v4', domain: 'general', samples: 120000, quality: 91.3, createdAt: '3d ago', sources: [{ name: 'OpenAssistant', count: 60000 }, { name: 'ShareGPT-filt', count: 40000 }, { name: 'synth-dialog', count: 20000 }] },
];

export const automlRuns: AutoMLRun[] = Array.from({ length: 47 }, (_, i) => {
  const statuses: AutoMLRun['status'][] = ['completed', 'completed', 'completed', 'pruned', 'running', 'promoted'];
  const status = i < 3 ? 'promoted' : i < 8 ? 'running' : i < 31 ? 'completed' : 'pruned';
  return {
    id: `aml-${String(i + 1).padStart(3, '0')}`,
    lr: parseFloat((Math.random() * 5e-4 + 1e-5).toExponential(1)),
    rank: [4, 8, 16, 32, 64][Math.floor(Math.random() * 5)],
    alpha: [8, 16, 32, 64][Math.floor(Math.random() * 4)],
    loss: status === 'pruned' ? parseFloat((Math.random() * 0.5 + 0.5).toFixed(4)) : parseFloat((Math.random() * 0.3 + 0.2).toFixed(4)),
    status,
    duration: status === 'running' ? 'In progress' : `${Math.floor(Math.random() * 120 + 10)}m`,
    steps: status === 'running' ? Math.floor(Math.random() * 5000) : status === 'pruned' ? Math.floor(Math.random() * 3000) : 10000,
  };
});

export const logLines: LogLine[] = [
  { id: 'l1', level: 'INFO', message: 'Training step 4821/10000 completed', source: 'trainer', timestamp: '00:14:23' },
  { id: 'l2', level: 'METRIC', message: 'loss=0.3421 lr=2.4e-4 grad_norm=1.23', source: 'metrics', timestamp: '00:14:23' },
  { id: 'l3', level: 'INFO', message: 'Checkpoint saved: ckpt-4800', source: 'trainer', timestamp: '00:14:20' },
  { id: 'l4', level: 'AGENT', message: '[PRUNER] Analyzing run-2841 loss curve...', source: 'agent', timestamp: '00:14:18' },
  { id: 'l5', level: 'WARN', message: 'GPU-1 temperature 83°C exceeds soft limit', source: 'cluster', timestamp: '00:14:15' },
  { id: 'l6', level: 'AGENT', message: '[PRUNER] Decision: PRUNE run-2841 (plateau)', source: 'agent', timestamp: '00:14:12' },
  { id: 'l7', level: 'INFO', message: 'Run-2841 terminated. Resources freed.', source: 'scheduler', timestamp: '00:14:10' },
  { id: 'l8', level: 'METRIC', message: 'loss=0.3433 lr=2.4e-4 grad_norm=1.18', source: 'metrics', timestamp: '00:14:03' },
  { id: 'l9', level: 'INFO', message: 'Eval checkpoint queued: run-2839-ckpt-10000', source: 'evaluator', timestamp: '00:13:58' },
  { id: 'l10', level: 'ERROR', message: 'OOM warning on node linux-rig-01 GPU-0', source: 'cluster', timestamp: '00:13:44' },
  { id: 'l11', level: 'INFO', message: 'Gradient accumulation adjusted: 4 → 8', source: 'trainer', timestamp: '00:13:42' },
  { id: 'l12', level: 'AGENT', message: '[EVALUATOR] Scoring run-2839...', source: 'agent', timestamp: '00:13:30' },
];

export const timelineEntries = [
  { id: 't1', name: 'run-2845', status: 'running' as const, start: 0, end: 91 },
  { id: 't2', name: 'run-2846', status: 'running' as const, start: 15, end: 72 },
  { id: 't3', name: 'run-2847', status: 'running' as const, start: 28, end: 48 },
  { id: 't4', name: 'run-2841', status: 'failed' as const, start: 5, end: 35 },
  { id: 't5', name: 'run-2839', status: 'completed' as const, start: 0, end: 100 },
  { id: 't6', name: 'run-2838', status: 'completed' as const, start: 0, end: 100 },
  { id: 't7', name: 'run-2836', status: 'failed' as const, start: 10, end: 45 },
  { id: 't8', name: 'run-2835', status: 'completed' as const, start: 0, end: 100 },
];
