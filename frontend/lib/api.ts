export type ModelVariant = string;
export type Difficulty = "easy" | "medium" | "hard" | "olympiad";
export type SolverMode = "rigorous" | "exam" | "concise" | "verification";
export type OutputFormat = "text" | "json";
export type TrainingOrigin = "existing_model" | "from_scratch";
export type LearningMode =
  | "supervised"
  | "unsupervised"
  | "rlhf"
  | "dpo"
  | "orpo"
  | "ppo"
  | "lora"
  | "qlora"
  | "full_finetune";
export type DeploymentTarget =
  | "huggingface"
  | "ollama"
  | "lmstudio"
  | "api"
  | "openai_compatible_api"
  | "custom_api";
export type LifecycleStage =
  | "prepare"
  | "train"
  | "evaluate"
  | "decide"
  | "finetune"
  | "infer"
  | "publish";

export type ArchitectureSpec = {
  family?: string | null;
  hidden_size?: number | null;
  num_layers?: number | null;
  num_attention_heads?: number | null;
  max_position_embeddings?: number | null;
  vocab_size?: number | null;
  notes?: string | null;
};

export type EvaluationSuiteSpec = {
  id?: string | null;
  label?: string | null;
  benchmark_config?: string | null;
  compare_to_models?: string[];
  custom_dataset_paths?: string[];
  notes?: string | null;
};

export type LifecycleProfile = {
  stage?: LifecycleStage | null;
  origin?: TrainingOrigin | null;
  learning_mode?: LearningMode | null;
  source_model?: string | null;
  architecture?: ArchitectureSpec;
  evaluation_suite?: EvaluationSuiteSpec | null;
  deployment_targets?: DeploymentTarget[];
  notes?: string[];
};

export type MetricPoint = {
  timestamp: string;
  name: string;
  value: number | boolean | null;
  unit?: string | null;
  tags?: Record<string, string>;
  metadata?: Record<string, unknown>;
};

export type DecisionResult = {
  action: string;
  rule: string;
  thresholds: Record<string, number | null>;
  summary: Record<string, unknown>;
  explanation: string;
};

export type FeedbackRecommendation = {
  action: string;
  reason: string;
  priority: number;
  target_instance_type?: string | null;
  config_path?: string | null;
  deployment_target?: DeploymentTarget | null;
  command?: string[] | null;
  metadata?: Record<string, unknown>;
};

export type InstanceActionDescriptor = {
  action: string;
  label: string;
  description: string;
  target_instance_type?: string | null;
  config_path?: string | null;
  deployment_target?: DeploymentTarget | null;
};

export type EnvironmentSpec = {
  kind: string;
  profile_name?: string | null;
  host?: string | null;
  user?: string | null;
  port?: number;
  key_path?: string | null;
  remote_repo_root?: string | null;
  python_bin?: string | null;
  env?: Record<string, string>;
  port_forwards?: Array<{
    local_port: number;
    remote_port: number;
    bind_host?: string;
    description?: string | null;
  }>;
};

export type CandidateVerification = {
  final_answer?: string | null;
  equivalent?: boolean | null;
  step_correctness?: number | null;
  verifier_agreement?: boolean | null;
  formatting_failure?: boolean;
  arithmetic_slip?: boolean;
  error_type?: string;
};

export type Candidate = {
  text: string;
  display_text: string;
  reasoning: string;
  final_answer?: string | null;
  calculator_trace?: Array<{ expression: string; result: string }>;
  vote_count?: number;
  score?: number;
  verification?: CandidateVerification | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
};

export type GenerateResult = {
  model_variant: ModelVariant;
  prompt: string;
  answer: string;
  raw_text: string;
  final_answer?: string | null;
  reasoning_steps: string[];
  selected_score: number;
  candidates: Candidate[];
  verification?: CandidateVerification | null;
  structured?: {
    reasoning: string;
    final_answer?: string | null;
    verification?: CandidateVerification | null;
  } | null;
  cache_hit?: boolean;
  telemetry_id?: string | null;
  latency_s?: number | null;
  prompt_preset?: string | null;
  candidate_agreement?: number;
};

export type GenerateResponse = GenerateResult & {
  comparison?: GenerateResult | null;
};

export type CompareResponse = {
  primary: GenerateResult;
  secondary: GenerateResult;
};

export type GenerateRequest = {
  question: string;
  model_variant: ModelVariant;
  compare_to_base: boolean;
  compare_to_model?: string | null;
  prompt_preset: string;
  temperature: number;
  top_p: number;
  max_new_tokens: number;
  show_reasoning: boolean;
  difficulty_target: Difficulty;
  num_samples: number;
  use_calculator: boolean;
  solver_mode: SolverMode;
  output_format: OutputFormat;
  use_cache: boolean;
};

export type CompareRequest = {
  question: string;
  primary_model: ModelVariant;
  secondary_model: ModelVariant;
  prompt_preset: string;
  temperature: number;
  top_p: number;
  max_new_tokens: number;
  show_reasoning: boolean;
  difficulty_target: Difficulty;
  num_samples: number;
  use_calculator: boolean;
  solver_mode: SolverMode;
  output_format: OutputFormat;
  use_cache: boolean;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type DatasetPreview = {
  id: string;
  question: string;
  difficulty: string;
  topic: string;
  final_answer: string;
};

export type DatasetEntry = {
  id: string;
  title: string;
  kind: "custom" | "public";
  family: string;
  topic: string;
  path: string;
  num_rows: number;
  size_bytes: number;
  description: string;
  usage?: string;
  default_weight?: number;
  benchmark_tags?: string[];
  reasoning_style?: string;
  preview_examples: DatasetPreview[];
};

export type PackEntry = {
  id: string;
  description: string;
  num_rows: number;
  size_bytes: number;
  path: string;
};

export type DatasetDashboard = {
  generated_at: string | null;
  summary: {
    num_datasets: number;
    custom_datasets: number;
    public_datasets: number;
    total_bytes: number;
    total_rows: number;
  };
  datasets: DatasetEntry[];
  packs?: PackEntry[];
};

export type PromptPreset = {
  id: string;
  title: string;
  description: string;
  style_instructions: string;
};

export type PromptExample = {
  dataset_id: string;
  dataset_title: string;
  question: string;
  difficulty: string;
  topic: string;
};

export type PromptLibrary = {
  presets: PromptPreset[];
  examples: PromptExample[];
};

export type ModelInfo = {
  name: ModelVariant;
  label?: string;
  description?: string | null;
  base_model: string;
  adapter_path?: string | null;
  available: boolean;
  tags?: string[];
};

export type BenchmarkInfo = {
  id: string;
  title: string;
  path: string;
  description: string;
  tags: string[];
};

export type RunInfo = {
  run_id?: string;
  run_name: string;
  profile_name?: string;
  base_model: string;
  output_dir: string;
  metrics: Record<string, number>;
  model_report: {
    trainable_parameters?: number;
    total_parameters?: number;
    trainable_ratio?: number;
  };
  dataset_report: Record<string, unknown>;
};

export type StatusInfo = {
  title: string;
  version: string;
  models: ModelInfo[];
  cache: {
    enabled?: boolean;
    entries?: number;
    size_bytes?: number;
    root?: string;
  };
  benchmarks: number;
  runs: number;
};

export type WorkspaceCheck = {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
};

export type WorkspaceRecipe = {
  id: string;
  title: string;
  description: string;
  command: string;
  category: string;
};

export type WorkspaceTrainingProfile = {
  id: string;
  title: string;
  path: string;
  dry_run_command: string;
  train_command: string;
};

export type WorkspaceEvaluationConfig = {
  id: string;
  title: string;
  path: string;
  run_command: string;
};

export type WorkspaceCapability = {
  id: string;
  title: string;
  detail: string;
};

export type WorkspaceOrchestrationTemplate = {
  id: string;
  title: string;
  path: string;
  instance_type: string;
  user_level: string;
  orchestration_mode: string;
  command: string;
};

export type WorkspaceOverview = {
  repo_root: string;
  summary: {
    datasets: number;
    packs: number;
    models: number;
    benchmarks: number;
    runs: number;
    training_profiles: number;
    evaluation_configs: number;
    orchestration_templates: number;
    ready_checks: number;
    total_checks: number;
  };
  models: ModelInfo[];
  readiness_checks: WorkspaceCheck[];
  command_recipes: WorkspaceRecipe[];
  orchestration_capabilities: WorkspaceCapability[];
  orchestration_templates: WorkspaceOrchestrationTemplate[];
  training_profiles: WorkspaceTrainingProfile[];
  evaluation_configs: WorkspaceEvaluationConfig[];
};

export type OrchestrationRun = {
  id: string;
  legacy_instance_id?: string | null;
  name: string;
  status: string;
  root_run_id?: string | null;
  parent_run_id?: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
};

export type OrchestrationTask = {
  id: string;
  run_id: string;
  legacy_instance_id?: string | null;
  task_type: string;
  agent_type: string;
  status: string;
  current_attempt: number;
  resource_class: string;
  available_at: string;
  checkpoint_hint?: string | null;
  metadata: Record<string, unknown>;
};

export type OrchestrationEvent = {
  id: string;
  run_id: string;
  task_id?: string | null;
  attempt_id?: string | null;
  event_type: string;
  level: string;
  message: string;
  created_at: string;
  payload: Record<string, unknown>;
};

export type OrchestrationRunDetail = {
  run: OrchestrationRun;
  tasks: OrchestrationTask[];
  events: OrchestrationEvent[];
  summary: Record<string, unknown>;
};

export type InstanceSummary = {
  id: string;
  type: string;
  status: string;
  name: string;
  created_at: string;
  updated_at: string;
  parent_instance_id?: string | null;
  orchestration_run_id?: string | null;
  environment: EnvironmentSpec;
  lifecycle: LifecycleProfile;
  metrics_summary: Record<string, unknown>;
  task_summary: Record<string, unknown>;
  recommendations?: FeedbackRecommendation[];
  decision?: DecisionResult | null;
  progress?: {
    stage: string;
    status_message?: string | null;
    percent?: number | null;
  } | null;
};

export type OrchestrationSummary = {
  runs?: number;
  tasks?: number;
  task_status_counts?: Record<string, number>;
  open_circuits?: string[];
};

export type InstanceDetail = InstanceSummary & {
  config_snapshot: Record<string, unknown>;
  logs?: {
    stdout: string;
    stderr: string;
    stdout_path?: string | null;
    stderr_path?: string | null;
  } | null;
  metrics: {
    summary: Record<string, unknown>;
    points: MetricPoint[];
  };
  children: InstanceSummary[];
  events: Array<Record<string, unknown>>;
  available_actions: InstanceActionDescriptor[];
};

export type CreateManagedInstanceRequest = {
  config_path: string;
  start?: boolean;
  environment?: EnvironmentSpec | null;
  parent_instance_id?: string | null;
  name?: string | null;
  user_level?: "beginner" | "hobbyist" | "dev" | null;
  lifecycle?: LifecycleProfile | null;
  subsystem_overrides?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    ...options,
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed: ${path}`);
  }
  return (await response.json()) as T;
}

export async function generateAnswer(payload: GenerateRequest): Promise<GenerateResponse> {
  return fetchJson<GenerateResponse>("/v1/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function compareModels(payload: CompareRequest): Promise<CompareResponse> {
  return fetchJson<CompareResponse>("/v1/compare", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function getDatasetDashboard(): Promise<DatasetDashboard> {
  return fetchJson<DatasetDashboard>("/v1/datasets");
}

export async function getPromptLibrary(): Promise<PromptLibrary> {
  return fetchJson<PromptLibrary>("/v1/prompts");
}

export async function getModels(): Promise<ModelInfo[]> {
  const payload = await fetchJson<{ models: ModelInfo[] }>("/v1/models");
  return payload.models;
}

export async function getBenchmarks(): Promise<BenchmarkInfo[]> {
  const payload = await fetchJson<{ benchmarks: BenchmarkInfo[] }>("/v1/benchmarks");
  return payload.benchmarks;
}

export async function getRuns(): Promise<RunInfo[]> {
  const payload = await fetchJson<{ runs: RunInfo[] }>("/v1/runs");
  return payload.runs;
}

export async function getStatus(): Promise<StatusInfo> {
  return fetchJson<StatusInfo>("/v1/status");
}

export async function getWorkspaceOverview(): Promise<WorkspaceOverview> {
  return fetchJson<WorkspaceOverview>("/v1/workspace");
}

export async function getOrchestrationRuns(): Promise<OrchestrationRun[]> {
  const payload = await fetchJson<{ runs: OrchestrationRun[] }>("/v1/orchestration/runs");
  return payload.runs;
}

export async function getOrchestrationRun(runId: string): Promise<OrchestrationRunDetail> {
  return fetchJson<OrchestrationRunDetail>(`/v1/orchestration/runs/${runId}`);
}

export async function getInstances(): Promise<InstanceSummary[]> {
  const payload = await fetchJson<{ instances: InstanceSummary[] }>("/v1/instances");
  return payload.instances;
}

export async function getInstanceDetail(instanceId: string): Promise<InstanceDetail> {
  return fetchJson<InstanceDetail>(`/v1/instances/${instanceId}`);
}

export async function createManagedInstance(
  payload: CreateManagedInstanceRequest,
): Promise<InstanceDetail> {
  return fetchJson<InstanceDetail>("/v1/instances", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function evaluateManagedInstance(
  instanceId: string,
  payload?: { config_path?: string | null; start?: boolean },
): Promise<InstanceDetail> {
  return fetchJson<InstanceDetail>(`/v1/instances/${instanceId}/evaluate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload ?? {}),
  });
}

export async function startManagedInference(
  instanceId: string,
  payload?: { config_path?: string | null; start?: boolean },
): Promise<InstanceDetail> {
  return fetchJson<InstanceDetail>(`/v1/instances/${instanceId}/inference`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload ?? {}),
  });
}

export async function deployManagedInstance(
  instanceId: string,
  payload: { target: DeploymentTarget; config_path?: string | null; start?: boolean },
): Promise<InstanceDetail> {
  return fetchJson<InstanceDetail>(`/v1/instances/${instanceId}/deploy`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function getOrchestrationSummary(): Promise<OrchestrationSummary> {
  const payload = await fetchJson<{ summary: OrchestrationSummary }>("/v1/orchestration/summary");
  return payload.summary;
}
