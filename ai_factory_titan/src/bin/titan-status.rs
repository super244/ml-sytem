use ai_factory_titan::{
    detect_hardware,
    quantization::default_q4_layout,
    KvCache,
    KvCacheConfig,
    SamplerConfig,
    TitanEngineDescriptor,
    TitanRuntimePlan,
    TitanScheduler,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let hardware = detect_hardware();
    let scheduler = TitanScheduler::status();
    let engine = TitanEngineDescriptor::local_default();
    let kv_cache = KvCache::new(KvCacheConfig {
        page_size: 32,
        max_tokens: 2048,
        heads: 32,
        head_dim: 128,
    });
    let runtime = TitanRuntimePlan::current();
    let sampler = SamplerConfig::default();
    let payload = serde_json::json!({
        "hardware": hardware,
        "scheduler": scheduler,
        "engine": engine,
        "runtime": {
            "selected": runtime.mode.as_str(),
            "env_var": "AI_FACTORY_TITAN_RUNTIME",
            "runtime_flag": "AI_FACTORY_TITAN_RUNTIME",
            "runtime_enabled": runtime.can_generate,
            "status_source": "rust-binary",
            "status_binary_available": true,
            "gguf_support": runtime.gguf_enabled,
            "kv_cache": {
                "strategy": "paged-kv",
                "page_size_tokens": 32,
                "capacity_tokens": kv_cache.stats().max_tokens,
                "stored_tokens": kv_cache.stats().stored_tokens
            },
            "sampler": sampler,
            "sampler_stack": ["argmax", "temperature", "top_k", "top_p", "repetition_penalty"]
        },
        "neural_accelerator": {
            "target": "apple-amx",
            "mode": "prompt-preprocess",
            "speedup_goal_vs_m4": 4
        },
        "quantization": {
            "formats": ["q4_0", "q4_k", "q8_0", "f16"],
            "layout": default_q4_layout()
        }
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}
