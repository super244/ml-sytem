import { titanStatusSchema, type TitanStatus } from "@/lib/titan-schema";

export type { TitanStatus } from "@/lib/titan-schema";

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
  base_model: string;
  context_window?: number;
  parameter_size_b?: number | null;
  quantization?: "4bit" | "8bit" | "16bit" | "none";
  lora_target_modules?: string[] | null;
  lora_rank?: number | null;
  lora_alpha?: number | null;
  metadata?: Record<string, any>;
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

const REQUEST_TIMEOUT_MS = 20_000;

function normalizeBase(base: string): string {
  return base.replace(/\/$/, "");
}

function resolveApiBases(): string[] {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (configured) {
    return [normalizeBase(configured)];
  }

  const bases = ["http://127.0.0.1:8000", "http://localhost:8000"];
  if (typeof window !== "undefined") {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    bases.unshift(`${protocol}//${window.location.hostname}:8000`);
  }

  return Array.from(new Set(bases.map(normalizeBase)));
}

function extractErrorMessage(payload: unknown): string | null {
  if (typeof payload === "string") {
    return payload.trim() || null;
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }

  const record = payload as Record<string, unknown>;
  const detail = record.detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((entry) => {
        if (typeof entry === "string") {
          return entry;
        }
        if (entry && typeof entry === "object" && !Array.isArray(entry)) {
          const nested = entry as Record<string, unknown>;
          const location = Array.isArray(nested.loc) ? nested.loc.join(".") : null;
          const message = typeof nested.msg === "string" ? nested.msg : null;
          return [location, message].filter(Boolean).join(": ");
        }
        return "";
      })
      .filter(Boolean)
      .join("; ");
  }

  const error = record.error;
  if (error && typeof error === "object" && !Array.isArray(error)) {
    const nested = error as Record<string, unknown>;
    if (typeof nested.message === "string" && nested.message.trim()) {
      return nested.message;
    }
  }

  if (typeof record.message === "string" && record.message.trim()) {
    return record.message;
  }

  return null;
}

async function readResponsePayload(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text.trim()) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export type DatasetPreview = {
  id: string;
  question: string;
  difficulty: string;
  topic: string;
  final_answer: string;
};

export type DatasetBuildInfo = {
  build_id: string;
  created_at?: string;
  git_sha?: string | null;
  config_path?: string | null;
  config_sha256?: string | null;
  seed?: number | null;
  notes?: string[];
};

export type DatasetFileInfo = {
  path: string;
  sha256: string;
  size_bytes: number;
  num_rows: number;
};

export type DatasetLineageSummaryGroup = {
  source_id: string;
  loader: string;
  version?: string | null;
  origin_path?: string | null;
  dataset_split: string;
  failure_case: boolean;
  record_count: number;
  exact_matches: number;
  near_matches: number;
  contaminated_records: number;
  max_similarity: number;
};

export type DatasetLineageSummary = {
  total_records: number;
  contamination: {
    exact_matches: number;
    near_matches: number;
    contaminated_records: number;
    failure_cases: number;
  };
  by_split: Record<string, number>;
  by_loader: Record<string, number>;
  groups: DatasetLineageSummaryGroup[];
};

export type DatasetManifest = {
  schema_version: string;
  manifest_type: "dataset" | "pack" | "benchmark";
  build: DatasetBuildInfo;
  pack_id?: string | null;
  description?: string | null;
  inputs: DatasetFileInfo[];
  outputs: DatasetFileInfo[];
  source_lineage: Array<{
    dataset_id: string;
    dataset_family: string;
    origin_path?: string | null;
    loader: string;
    source_url?: string | null;
    license?: string | null;
    filters: Record<string, unknown>;
    source_record_id?: string | null;
    notes: string[];
  }>;
  stats: Record<string, unknown>;
  metadata: Record<string, unknown>;
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
  manifest_path?: string | null;
  card_path?: string | null;
  build_id?: string | null;
  build?: DatasetBuildInfo | null;
  stats?: Record<string, unknown>;
};

export type DatasetProvenance = {
  processed_manifest: DatasetManifest | null;
  pack_summary: {
    packs: PackEntry[];
  };
  pack_manifests: DatasetManifest[];
  lineage_summary: DatasetLineageSummary | null;
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
  provenance?: DatasetProvenance;
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
  status?: "available" | "degraded";
  errors?: string[];
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
  status?: "available" | "degraded";
  errors?: string[];
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

export type WorkspaceInterfaceSurface = {
  id: string;
  label: string;
  entrypoint: string;
  backend_contract: string;
  description: string;
  status: string;
};

export type WorkspaceExperienceTier = {
  id: string;
  label: string;
  description: string;
  visible_controls: string[];
  recommended_modes: string[];
  safe_defaults: string[];
};

export type WorkspaceExtensionPoint = {
  id: string;
  kind: string;
  label: string;
  description: string;
  supported_instance_types: string[];
  source: string;
  maturity: string;
  config_hint?: string | null;
  future_ready?: boolean;
};

export type WorkspaceOverview = {
  status?: "available" | "degraded";
  errors?: string[];
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
    interfaces?: number;
    experience_tiers?: number;
    extension_points?: number;
    ready_checks: number;
    total_checks: number;
  };
  models: ModelInfo[];
  readiness_checks: WorkspaceCheck[];
  interfaces?: WorkspaceInterfaceSurface[];
  experience_tiers?: WorkspaceExperienceTier[];
  extension_points?: WorkspaceExtensionPoint[];
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
  active_runs?: number;
  run_status_counts?: Record<string, number>;
  task_status_counts?: Record<string, number>;
  task_type_counts?: Record<string, number>;
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

export type MissionControlSnapshot = {
  generated_at: string;
  repo_root: string;
  workspace: WorkspaceOverview;
  orchestration: OrchestrationSummary;
  autonomous?: AutonomousCampaignSnapshot;
  autonomy: AutonomyOverview;
  titan: TitanStatus;
  criticality: {
    level: "critical" | "warning" | "opportunity" | "info";
    counts: Record<string, number>;
  };
  recommendations: MissionControlRecommendation[];
  watchlist: {
    instances: InstanceSummary[];
    running_instances: InstanceSummary[];
    failed_instances: InstanceSummary[];
    agents: AgentSwarmStatus[];
    sweeps: AutoMLSweep[];
    cluster_nodes: ClusterNodeHardware[];
    telemetry: TelemetryRecord[];
  };
  control_plane: {
    instances: InstanceSummary[];
    runs: OrchestrationRun[];
    orchestration_summary: OrchestrationSummary;
  };
  agents: {
    path: string;
    count: number;
    status_counts: Record<string, number>;
    swarm: AgentSwarmStatus[];
  };
  automl: {
    path: string;
    count: number;
    status_counts: Record<string, number>;
    latest: AutoMLSweep | null;
    sweeps: AutoMLSweep[];
  };
  cluster: {
    nodes: ClusterNodeHardware[];
    status_counts: Record<string, number>;
    placements?: Array<Record<string, unknown>>;
    pressure?: Record<string, unknown>;
  };
  lineage?: {
    records?: number;
    roots?: number;
    covered_instances?: number;
    gap_count?: number;
    gaps?: Array<Record<string, unknown>>;
    latest?: Array<Record<string, unknown>>;
  };
  telemetry: {
    flagged: {
      path: string;
      count: number;
      recent: TelemetryRecord[];
      latest: TelemetryRecord | null;
    };
    requests: {
      path: string;
      count: number;
      by_model: Record<string, number>;
      recent: Array<Record<string, unknown>>;
    };
  };
  summary: {
    workspace_ready_checks: number;
    workspace_total_checks: number;
    instances: number;
    running_instances: number;
    failed_instances: number;
    orchestration_runs: number;
    autonomous_campaigns?: number;
    ready_autonomous_actions?: number;
    lineage_records?: number;
    lineage_gaps?: number;
    agents: number;
    automl_sweeps: number;
    cluster_nodes: number;
    telemetry_flags: number;
    telemetry_requests: number;
    active_agents: number;
    running_sweeps: number;
    telemetry_backlog: number;
    ready_checks: number;
    total_checks: number;
    datasets: number;
    training_profiles: number;
    open_circuits: number;
    autonomous_actions?: number;
    autonomous_executable_actions?: number;
    autonomous_ready?: boolean;
    autonomous_blockers: number;
    titan_backend?: string;
    titan_mode?: string;
    autonomy_status?: AutonomyOverview["status"];
    autonomy_mode?: AutonomyOverview["mode"];
    stalled_runs?: number;
  };
};

export type MissionControlRecommendation = {
  id: string;
  severity: "critical" | "warning" | "opportunity" | "info";
  title: string;
  detail: string;
  surface: string;
  href: string;
  metric_label?: string | null;
  metric_value?: string | null;
  command?: string | null;
};

export type AutonomyStageSnapshot = {
  id: string;
  title: string;
  status: "blocked" | "active" | "attention" | "ready" | "idle";
  headline: string;
  detail: string;
  href: string;
  metric_label?: string | null;
  metric_value?: string | null;
  counts: Record<string, number>;
};

export type AutonomyAgentCoverage = {
  agent_type: string;
  label: string;
  status: "blocked" | "active" | "attention" | "ready" | "idle";
  queued_tasks: number;
  running_tasks: number;
  active_swarm_agents: number;
  open_circuit: boolean;
  resource_classes: string[];
  recommended_action: string;
};

export type AutonomyLineageAlert = {
  id: string;
  severity: "critical" | "warning" | "opportunity" | "info";
  title: string;
  detail: string;
  href: string;
  instance_id?: string | null;
};

export type AutonomyNextAction = {
  id: string;
  title: string;
  detail: string;
  href: string;
  category: "stabilize" | "dispatch" | "optimize" | "lineage";
  blocking: boolean;
  command?: string | null;
};

export type AutonomyCapacitySnapshot = {
  status: "blocked" | "active" | "ready" | "idle";
  idle_nodes: number;
  busy_nodes: number;
  offline_nodes: number;
  active_gpu_tasks: number;
  active_cpu_tasks: number;
  schedulable_trials: number;
  suggested_parallelism: number;
  bottleneck: string;
  execution_modes: Record<string, number>;
};

export type AutonomyOverview = {
  status: "blocked" | "degraded" | "active" | "ready" | "idle";
  mode: "manual" | "assisted" | "autonomous";
  summary: string;
  open_circuits: number;
  telemetry_backlog: number;
  active_runs: number;
  running_sweeps: number;
  stalled_runs: number;
  stages: AutonomyStageSnapshot[];
  agent_coverage: AutonomyAgentCoverage[];
  capacity: AutonomyCapacitySnapshot;
  lineage_alerts: AutonomyLineageAlert[];
  next_actions: AutonomyNextAction[];
};

export type AutonomousLoopAction = {
  id: string;
  kind: "launch_training" | "run_action" | "advisory";
  title: string;
  detail: string;
  priority: number;
  executable: boolean;
  status: "planned" | "executed" | "blocked" | "failed" | "skipped";
  source_instance_id?: string | null;
  source_instance_name?: string | null;
  action?: string | null;
  config_path?: string | null;
  deployment_target?: string | null;
  surface: string;
  href: string;
  command?: string | null;
  created_instance_id?: string | null;
  error?: string | null;
  metadata: Record<string, unknown>;
};

export type AutonomousLoopRun = {
  id: string;
  created_at: string;
  status: "planned" | "executed" | "partial" | "blocked" | "failed";
  dry_run: boolean;
  start_instances: boolean;
  blockers: string[];
  summary: Record<string, unknown>;
  actions: AutonomousLoopAction[];
};

export type AutonomousLoopSnapshot = {
  generated_at: string;
  ready: boolean;
  blockers: string[];
  summary: {
    workspace_ready: boolean;
    ready_checks: number;
    total_checks: number;
    instances: number;
    completed_instances: number;
    running_instances: number;
    failed_instances: number;
    completed_evaluations: number;
    completed_training_branches: number;
    telemetry_backlog: number;
    idle_nodes: number;
    open_circuits: number;
    total_actions: number;
    executable_actions: number;
    advisory_actions: number;
    [key: string]: unknown;
  };
  actions: AutonomousLoopAction[];
  recent_loops: AutonomousLoopRun[];
  latest_loop: AutonomousLoopRun | null;
};

export type AutonomousCampaignAction = {
  id: string;
  kind: "prepare" | "train" | "finetune" | "evaluate" | "inference" | "deploy" | "lineage";
  title: string;
  detail: string;
  status: "planned" | "started" | "blocked" | "completed" | "failed" | "skipped";
  config_path?: string | null;
  source_instance_id?: string | null;
  instance_id?: string | null;
  depends_on: string[];
  metadata: Record<string, unknown>;
};

export type AutonomousCampaign = {
  campaign_id: string;
  experiment_name: string;
  goal: string;
  status: "planned" | "running" | "completed" | "degraded";
  created_at: string;
  updated_at: string;
  parameters: Record<string, unknown>;
  plan: AutonomousCampaignAction[];
  execution: Array<Record<string, unknown>>;
  summary: Record<string, unknown>;
};

export type AutonomousCampaignSnapshot = {
  status: "available";
  write_enabled: boolean;
  path: string;
  count: number;
  status_counts: Record<string, number>;
  active_campaigns: number;
  campaigns: AutonomousCampaign[];
  ready_actions: AutonomousCampaignAction[];
  actions: AutonomousLoopAction[];
  loop_health: Record<string, unknown>;
  lineage: {
    records: number;
    roots: number;
    covered_instances: number;
    gap_count: number;
    gaps: Array<Record<string, unknown>>;
    latest: Array<Record<string, unknown>>;
  };
  cluster: {
    nodes: ClusterNodeHardware[];
    idle_nodes: number;
    placements: Array<Record<string, unknown>>;
    pressure: Record<string, unknown>;
  };
  summary: {
    total_actions: number;
    executable_actions: number;
    advisory_actions: number;
    telemetry_backlog: number;
    idle_nodes: number;
    [key: string]: unknown;
  };
  ready: boolean;
  blockers: string[];
};

export type CreateAutonomousCampaignRequest = {
  experiment_name: string;
  goal: string;
  parameters?: Record<string, string | number | boolean>;
  auto_start?: boolean;
  max_actions?: number;
};

export type AutonomousCampaignResponse = {
  experiment_id: string;
  status: string;
  message: string;
  campaign: AutonomousCampaign;
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
  const externalSignal = options?.signal;
  let lastNetworkError: unknown = null;

  for (const base of resolveApiBases()) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    if (externalSignal) {
      if (externalSignal.aborted) {
        controller.abort();
      } else {
        externalSignal.addEventListener("abort", () => controller.abort(), { once: true });
      }
    }

    try {
      const response = await fetch(`${base}${path}`, {
        cache: "no-store",
        ...options,
        signal: controller.signal,
      });

      if (!response.ok) {
        const payload = await readResponsePayload(response);
        const message =
          extractErrorMessage(payload) ??
          `Request failed (${response.status}): ${path}`;
        throw new Error(message);
      }

      const payload = await readResponsePayload(response);
      return payload as T;
    } catch (error) {
      if (error instanceof Error && error.message.startsWith("Request failed")) {
        throw error;
      }
      if (error instanceof DOMException && error.name === "AbortError") {
        lastNetworkError = new Error(`Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s: ${path}`);
        continue;
      }
      if (error instanceof TypeError) {
        lastNetworkError = error;
        continue;
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  const errorMessage =
    lastNetworkError instanceof Error
      ? lastNetworkError.message
      : `Network request failed for ${path}. Check that the AI-Factory API is reachable.`;
  throw new Error(`${errorMessage} Tried: ${resolveApiBases().join(", ")}`);
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

export async function runManagedInstanceAction(
  instanceId: string,
  payload: { action: string; config_path?: string | null; deployment_target?: DeploymentTarget | null; start?: boolean },
): Promise<InstanceDetail> {
  return fetchJson<InstanceDetail>(`/v1/instances/${instanceId}/actions`, {
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

export async function getMissionControl(): Promise<MissionControlSnapshot> {
  return fetchJson<MissionControlSnapshot>("/v1/lab/mission-control");
}

export async function getAutonomousSnapshot(): Promise<AutonomousLoopSnapshot> {
  return fetchJson<AutonomousLoopSnapshot>("/v1/experiments/autonomous");
}

export async function getAutonomousOverview(): Promise<AutonomyOverview> {
  return fetchJson<AutonomyOverview>("/v1/experiments/autonomous/overview");
}

export async function getAutonomousCampaigns(): Promise<AutonomousCampaignSnapshot> {
  return fetchJson<AutonomousCampaignSnapshot>("/v1/experiments/autonomous/campaigns");
}

export async function runAutonomousCampaign(
  payload: CreateAutonomousCampaignRequest,
): Promise<AutonomousCampaignResponse> {
  return fetchJson<AutonomousCampaignResponse>("/v1/experiments/autonomous/campaigns/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getAutonomousCampaign(experimentId: string): Promise<AutonomousCampaignResponse> {
  return fetchJson<AutonomousCampaignResponse>(`/v1/experiments/autonomous/campaigns/${experimentId}`);
}

export async function planAutonomousLoop(payload?: { max_actions?: number }): Promise<AutonomousLoopRun> {
  return fetchJson<AutonomousLoopRun>("/v1/experiments/autonomous/plan", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload ?? {}),
  });
}

export async function executeAutonomousLoop(payload?: {
  max_actions?: number;
  dry_run?: boolean;
  start_instances?: boolean;
}): Promise<AutonomousLoopRun> {
  return fetchJson<AutonomousLoopRun>("/v1/experiments/autonomous/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload ?? {}),
  });
}

export async function getTitanStatus(): Promise<TitanStatus> {
  const payload = await fetchJson<unknown>("/v1/titan/status");
  return titanStatusSchema.parse(payload);
}

export type FlagTelemetryRequest = {
  prompt: string;
  assistant_output: string;
  expected_output: string;
  model_variant: string;
  latency_s?: number | null;
};

export async function flagTelemetry(payload: FlagTelemetryRequest): Promise<{status: string, message: string}> {
  return fetchJson<{status: string, message: string}>("/v1/telemetry/flag", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export type ClusterNodeHardware = {
  id: string;
  name: string;
  type: string;
  memory: string;
  usage: number;
  status: "online" | "idle" | "offline";
  activeJobs: number;
};

export async function getClusterNodes(): Promise<ClusterNodeHardware[]> {
  const payload = await fetchJson<{ nodes: ClusterNodeHardware[] }>("/v1/cluster/nodes");
  return payload.nodes;
}

export type TelemetryRecord = {
  id: string;
  timestamp: number;
  prompt: string;
  assistant_output: string;
  expected_output: string;
  model_variant: string;
  latency_s?: number | null;
};

export type TelemetryActionResult = {
  status: string;
  record: TelemetryRecord;
  destination?: string | null;
  message?: string | null;
};

export async function getTelemetryBacklog(): Promise<TelemetryRecord[]> {
  const payload = await fetchJson<{ telemetry: TelemetryRecord[] }>("/v1/datasets/telemetry");
  return payload.telemetry;
}

export async function promoteTelemetryRecord(recordId: string): Promise<TelemetryActionResult> {
  return fetchJson<TelemetryActionResult>(`/v1/datasets/telemetry/${recordId}/promote`, {
    method: "POST",
  });
}

export async function discardTelemetryRecord(recordId: string): Promise<TelemetryActionResult> {
  return fetchJson<TelemetryActionResult>(`/v1/datasets/telemetry/${recordId}/discard`, {
    method: "POST",
  });
}

export type SynthesizeRequest = {
  seed_prompt: string;
  num_variants: number;
  model_variant: string;
};

export type SynthesizeResponse = {
  status: string;
  job_id: string;
  message: string;
  estimated_time_s: number;
};

export type SynthesisJob = {
  job_id: string;
  status: string;
  seed_prompt: string;
  num_variants: number;
  model_variant: string;
  created_at: number;
  estimated_time_s: number;
  completed_rows: number;
  output_path: string;
};

export async function synthesizeDataset(payload: SynthesizeRequest): Promise<SynthesizeResponse> {
  return fetchJson<SynthesizeResponse>("/v1/datasets/synthesize", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function getSynthesisJob(jobId: string): Promise<SynthesisJob> {
  return fetchJson<SynthesisJob>(`/v1/datasets/synthesize/${jobId}`);
}

export type AgentSwarmStatus = {
  id: string;
  name: string;
  role: string;
  model: string;
  status: "active" | "sleeping" | "offline";
  uptime_s: number;
  tokens_used: number;
};

export type AgentLogEvent = {
  timestamp: number;
  message: string;
  level: string;
};

export type AgentSwarmSnapshot = {
  status?: string;
  write_enabled?: boolean;
  deploy_enabled?: boolean;
  swarm: AgentSwarmStatus[];
};

export type AgentLogsSnapshot = {
  status?: string;
  simulation_enabled?: boolean;
  logs: AgentLogEvent[];
};

export async function getAgentSwarmSnapshot(): Promise<AgentSwarmSnapshot> {
  return fetchJson<AgentSwarmSnapshot>("/v1/agents/swarm");
}

export async function getAgentSwarmStatus(): Promise<AgentSwarmStatus[]> {
  const payload = await getAgentSwarmSnapshot();
  return payload.swarm;
}

export async function getAgentLogsSnapshot(limit: number = 20): Promise<AgentLogsSnapshot> {
  return fetchJson<AgentLogsSnapshot>(`/v1/agents/logs?limit=${limit}`);
}

export async function getAgentLogs(limit: number = 20): Promise<AgentLogEvent[]> {
  const payload = await getAgentLogsSnapshot(limit);
  return payload.logs;
}

export type AgentDeployRequest = {
  name: string;
  role: string;
  model: string;
};

export type AgentUpdateRequest = {
  name?: string;
  role?: string;
  model?: string;
  status?: AgentSwarmStatus["status"];
};

export async function deployAgent(payload: AgentDeployRequest): Promise<{ status: string; agent: AgentSwarmStatus }> {
  return fetchJson<{ status: string; agent: AgentSwarmStatus }>("/v1/agents/deploy", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function updateAgent(
  agentId: string,
  payload: AgentUpdateRequest,
): Promise<{ status: string; agent: AgentSwarmStatus }> {
  return fetchJson<{ status: string; agent: AgentSwarmStatus }>(`/v1/agents/${agentId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export type AutoMLTrialParams = {
  learning_rate: number;
  batch_size: number;
  warmup_ratio: number;
  lora_rank: number;
};

export type AutoMLTrialMetrics = {
  final_loss: number;
  accuracy: number;
  perplexity: number;
};

export type AutoMLTrial = {
  trial_id: string;
  status: string;
  params: AutoMLTrialParams;
  metrics: AutoMLTrialMetrics;
  duration_s: number;
};

export type AutoMLSweep = {
  id: string;
  name: string;
  base_model: string;
  strategy: string;
  status: string;
  num_trials: number;
  completed_trials: number;
  created_at: number;
  best_trial: AutoMLTrial;
  trials: AutoMLTrial[];
};

export type AutoMLSweepSnapshot = {
  status?: string;
  write_enabled?: boolean;
  sweeps: AutoMLSweep[];
};

export type LaunchSweepRequest = {
  name: string;
  base_model: string;
  strategy: string;
  num_trials: number;
  search_space?: {
    learning_rate?: number[];
    batch_size?: number[];
    warmup_ratio?: number[];
    lora_rank?: number[];
  };
};

export async function getSweepsSnapshot(): Promise<AutoMLSweepSnapshot> {
  return fetchJson<AutoMLSweepSnapshot>("/v1/automl/sweeps");
}

export async function getSweeps(): Promise<AutoMLSweep[]> {
  const payload = await getSweepsSnapshot();
  return payload.sweeps;
}

export async function launchSweep(payload: LaunchSweepRequest): Promise<AutoMLSweep> {
  return fetchJson<AutoMLSweep>("/v1/automl/sweeps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getSweepDetail(sweepId: string): Promise<AutoMLSweep> {
  return fetchJson<AutoMLSweep>(`/v1/automl/sweeps/${sweepId}`);
}

export type DatasetSpec = {
  name: string;
  description: string;
  path: string;
  domain: string;
  subdomain?: string | null;
  difficulty_range?: string[];
  size?: number;
  format?: string;
  metadata?: Record<string, any>;
};

export type MetricSpec = {
  name: string;
  description: string;
  type: string;
  domain: string;
  subdomain?: string | null;
  range?: number[] | null;
  metadata?: Record<string, any>;
};

export type EvaluationSpec = {
  name: string;
  description: string;
  domain: string;
  subdomain?: string | null;
  datasets: string[];
  metrics: string[];
  splits?: string[];
  size?: number;
  metadata?: Record<string, any>;
};

export type TrainingProfileSpec = {
  name: string;
  description: string;
  domain: string;
  subdomain?: string | null;
  training_method: string;
  datasets: string[];
  config_path: string;
  curriculum_order?: string[] | null;
  model_requirements?: Record<string, any>;
  metadata?: Record<string, any>;
};

export type ModelArtifact = {
  name: string;
  version: string;
  path: string;
  domain?: string | null;
  architecture: string;
  parameters: number;
  format: string;
  metadata?: Record<string, any>;
};

export type DeploymentSpec = {
  target: string;
  model_name: string;
  config?: Record<string, any>;
  public?: boolean;
  metadata?: Record<string, any>;
};

export type ScalingConfig = {
  max_nodes?: number;
  default_resources?: Record<string, any>;
  cluster_type?: string;
  metadata?: Record<string, any>;
};

export type MonitoringConfig = {
  collection_interval_seconds?: number;
  storage_backend?: string;
  alert_channels?: string[];
  thresholds?: Record<string, any>;
  metadata?: Record<string, any>;
};

export type TrainingJob = {
  name: string;
  profile: string;
  resource_requirements?: Record<string, any>;
  estimated_duration_hours?: number;
  priority?: string;
  metadata?: Record<string, any>;
};

export type Alert = {
  id: string;
  severity: string;
  message: string;
  source: string;
  timestamp: string;
  metadata?: Record<string, any>;
};

export type DomainType = "mathematics" | "coding" | "reasoning" | "vision" | "general";

export type ResourceProfile = {
  vram_required_gb: number;
  cpu_cores_required?: number;
  system_memory_gb?: number;
  recommended_gpus?: number;
  storage_gb?: number;
};

export type PerformanceProfile = {
  throughput_tokens_per_sec?: number | null;
  latency_ms_per_token?: number | null;
  memory_footprint_gb?: number | null;
  power_consumption_w?: number | null;
};

export type ModelLineage = {
  parent_model?: string | null;
  training_dataset_ids?: string[];
  training_run_ids?: string[];
  creation_timestamp?: string;
};

export type ModelCapability = {
  domain: DomainType;
  score: number;
  benchmark_name: string;
  verified?: boolean;
};

export type UniversalModelSpec = {
  id: string;
  name: string;
  version: string;
  domain: DomainType;
  architecture: ArchitectureSpec;
  resource_profile: ResourceProfile;
  performance_profile?: PerformanceProfile | null;
  lineage?: ModelLineage;
  capabilities?: ModelCapability[];
  tags?: string[];
  metadata?: Record<string, any>;
};

export type SearchSpaceSpec = {
  hyperparameters?: Record<string, any[]>;
  architectures?: string[] | null;
  datasets?: string[] | null;
};

export type OptimizationObjective = {
  metric: string;
  maximize?: boolean;
  target_value?: number | null;
  weight?: number;
};

export type IterationStrategy = {
  max_iterations?: number;
  early_stopping_patience?: number;
  exploration_factor?: number;
  batch_size?: number;
};

export type ResourceBudget = {
  max_compute_hours?: number | null;
  max_cost_usd?: number | null;
  max_gpu_hours?: number | null;
};

export type EvaluationCriterion = {
  metric_name: string;
  min_threshold: number;
  critical?: boolean;
};

export type AutoDeploymentPolicy = {
  enabled?: boolean;
  targets?: DeploymentTarget[];
  approval_required?: boolean;
  rollback_on_failure?: boolean;
};

export type AutonomousExperimentConfig = {
  experiment_id: string;
  name: string;
  domains: DomainType[];
  search_space: SearchSpaceSpec;
  objectives: OptimizationObjective[];
  strategy?: IterationStrategy;
  budget?: ResourceBudget;
  evaluation_criteria?: EvaluationCriterion[];
  deployment_policy?: AutoDeploymentPolicy;
  metadata?: Record<string, any>;
};
