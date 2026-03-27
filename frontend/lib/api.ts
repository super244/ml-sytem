export type ModelVariant = string;
export type Difficulty = "easy" | "medium" | "hard" | "olympiad";
export type SolverMode = "rigorous" | "exam" | "concise" | "verification";
export type OutputFormat = "text" | "json";

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
