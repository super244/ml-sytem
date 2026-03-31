import { z } from "zod";

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
  scheduler: z.object({
    runtime: z.string(),
    queue_policy: z.string(),
    ui_frame_budget_hz: z.number(),
  }),
  quantization: z.object({
    formats: z.array(z.string()),
    memory_layout: z.string(),
  }),
  telemetry: z.object({
    bridge: z.string(),
    target_latency_ms: z.number(),
    metrics: z.array(z.string()),
  }),
  rust_core: z.object({
    crate_root: z.string(),
    cargo_toml: z.string(),
    toolchain_available: z.boolean(),
    python_bridge_stub: z.string(),
  }),
});

export type TitanStatus = z.infer<typeof titanStatusSchema>;
