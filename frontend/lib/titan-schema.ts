import { z } from "zod";

const tensorLayoutSchema = z.object({
  format: z.string(),
  block_size: z.number(),
  bytes_per_block: z.number(),
  alignment_bytes: z.number(),
  storage: z.string(),
});

const schedulerSchema = z.object({
  runtime: z.string(),
  queue_policy: z.string(),
  ui_frame_budget_hz: z.number(),
  max_inflight_tasks: z.number().optional(),
  priority_bands: z.number().optional(),
});

const quantizationSchema = z.object({
  formats: z.array(z.string()),
  memory_layout: z.string(),
  default_layout: tensorLayoutSchema.optional(),
  layout: tensorLayoutSchema.nullable().optional(),
});

const telemetrySchema = z.object({
  bridge: z.string(),
  target_latency_ms: z.number(),
  metrics: z.array(z.string()),
  runtime_flag: z.string().optional(),
  runtime_enabled: z.boolean().optional(),
});

const rustCoreSchema = z.object({
  crate_root: z.string(),
  cargo_toml: z.string(),
  toolchain_available: z.boolean(),
  python_bridge_stub: z.string(),
  build_script: z.string().optional(),
  build_rs: z.string().optional(),
  cpp_bridge_stub: z.string().optional(),
  cpp_bridge: z.string().optional(),
  runtime_flag: z.string().optional(),
  titan_status_bin: z.string().optional(),
  gguf_module: z.string().optional(),
  kv_cache_module: z.string().optional(),
  runtime_module: z.string().optional(),
  sampler_module: z.string().optional(),
  features: z.array(z.string()).optional(),
  cpp_feature_available: z.boolean().optional(),
  status_binary: z.string().nullable().optional(),
  status_binary_available: z.boolean().optional(),
});

const runtimeSchema = z.object({
  selected: z.string().optional(),
  env_var: z.string().optional(),
  runtime_flag: z.string().optional(),
  runtime_enabled: z.boolean().optional(),
  status_source: z.string().optional(),
  status_binary_available: z.boolean().optional(),
  gguf_support: z.boolean(),
  kv_cache: z.union([
    z.boolean(),
    z.object({
      enabled: z.boolean().optional(),
      strategy: z.string(),
    }),
    z.object({
      strategy: z.string(),
      page_size_tokens: z.number(),
      capacity_tokens: z.number(),
      stored_tokens: z.number(),
    }),
  ]),
  sampler: z
    .union([
      z.object({
        stack: z.array(z.string()),
      }),
      z.object({
        temperature: z.number(),
        top_k: z.number(),
        top_p: z.number(),
        repetition_penalty: z.number(),
      }),
    ])
    .optional(),
  sampler_stack: z.array(z.string()).optional(),
});

const engineSchema = z.object({
  architecture: z.string(),
  decode_model: z.string(),
  runtime_mode: z.string().optional(),
  runtime_ready: z.boolean().optional(),
  runtime_reason: z.string().optional(),
  max_context_tokens: z.number(),
  max_batch_tokens: z.number(),
  cache_strategy: z.string(),
  scheduler_queue_depth: z.number(),
  runtime_env: z.string(),
  supports_gguf: z.boolean().optional(),
  supports_kv_cache: z.boolean().optional(),
  supports_sampler_stack: z.boolean().optional(),
  gguf_support: z.boolean(),
  kv_cache: z.boolean(),
  sampler_stack: z.array(z.string()),
  supported_quantizations: z.array(z.string()),
  default_layout: tensorLayoutSchema,
  acceleration: z.object({
    rust_fallback: z.boolean(),
    cpp_kernels: z.boolean(),
    metal_backend: z.boolean(),
    cuda_backend: z.boolean(),
  }),
});

export const titanStatusSchema = z.object({
  silicon: z.string(),
  platform: z.string(),
  backend: z.string(),
  mode: z.string(),
  unified_memory_gb: z.number().nullable(),
  bandwidth_gbps: z.number().nullable(),
  gpu_name: z.string().nullable(),
  gpu_vendor: z.string().nullable(),
  gpu_count: z.number(),
  cpu_threads: z.number(),
  cpu_fallback_threads: z.number(),
  zero_copy_supported: z.boolean(),
  supports_metal: z.boolean(),
  supports_cuda: z.boolean(),
  supports_mlx: z.boolean(),
  supports_pyo3_bridge: z.boolean(),
  remote_execution: z.boolean(),
  cloud_provider: z.string().nullable(),
  cuda_compute_capability: z.string().nullable(),
  cuda_memory_gb: z.number().nullable(),
  cuda_driver_version: z.string().nullable(),
  silent_mode: z.boolean(),
  gpu_cap_pct: z.number(),
  preferred_training_backend: z.string(),
  runtime_source: z.string(),
  scheduler: schedulerSchema,
  quantization: quantizationSchema,
  telemetry: telemetrySchema,
  rust_core: rustCoreSchema,
  runtime: runtimeSchema,
  engine: engineSchema,
});

export type TitanStatus = z.infer<typeof titanStatusSchema>;
