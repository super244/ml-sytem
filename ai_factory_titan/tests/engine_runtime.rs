use ai_factory_titan::{
    dot_f32,
    matmul_f32,
    parse_gguf_header,
    quantization::estimate_q4_bytes,
    sample_token,
    KvCache,
    KvCacheConfig,
    QuantizationFormat,
    QuantizedTensorLayout,
    SamplerConfig,
    TensorShape,
    TitanEngineDescriptor,
    TitanRuntimeMode,
    TitanRuntimePlan,
};

#[test]
fn titan_engine_descriptor_reports_expected_capabilities() {
    let descriptor = TitanEngineDescriptor::local_default();
    assert_eq!(descriptor.decode_model, "llama.cpp-inspired");
    assert!(descriptor.acceleration.rust_fallback);
    assert!(descriptor.supported_quantizations.len() >= 4);
    assert!(descriptor.gguf_support);
    assert!(descriptor.kv_cache);
}

#[test]
fn quantized_layout_estimates_storage() {
    let layout = QuantizedTensorLayout::for_format(QuantizationFormat::Q4K);
    let bytes = layout.estimated_bytes(TensorShape { rows: 128, cols: 256 });
    assert_eq!(bytes, estimate_q4_bytes(128, 256));
    assert!(bytes > 0);
}

#[test]
fn cpu_dot_and_matmul_are_consistent() {
    let dot = dot_f32(&[1.0, 2.0, 3.0], &[0.5, 1.5, -1.0]).expect("dot should succeed");
    assert!((dot - 0.5).abs() < 1e-5);

    let out = matmul_f32(
        &[1.0, 2.0, 3.0, 4.0],
        2,
        2,
        &[5.0, 6.0, 7.0, 8.0],
        2,
    )
    .expect("matmul should succeed");
    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn gguf_header_probe_reads_minimal_header() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&11u64.to_le_bytes());
    bytes.extend_from_slice(&7u64.to_le_bytes());

    let header = parse_gguf_header(&bytes).expect("gguf header should parse");
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 11);
    assert_eq!(header.metadata_kv_count, 7);
}

#[test]
fn kv_cache_tracks_recent_tokens_and_stats() {
    let mut cache = KvCache::new(KvCacheConfig {
        page_size: 2,
        max_tokens: 4,
        heads: 4,
        head_dim: 8,
    });
    cache.append(11).expect("append one");
    cache.append(17).expect("append two");
    cache.append(23).expect("append three");

    assert_eq!(cache.recent(2), &[17, 23]);
    assert_eq!(cache.stats().stored_tokens, 3);
    assert_eq!(cache.stats().pages_in_use, 2);
}

#[test]
fn sampler_prefers_best_scoring_token_with_repetition_penalty() {
    let params = SamplerConfig {
        temperature: 0.7,
        top_k: 3,
        top_p: 0.9,
        repetition_penalty: 2.5,
    };
    let pick = sample_token(&[0.2, 1.4, 1.1, 0.7], &[1], &params).expect("sampler should pick a token");
    assert_eq!(pick, 2);
}

#[test]
fn runtime_plan_tracks_runtime_flag_modes() {
    let runtime = TitanRuntimePlan::current();
    assert!(matches!(runtime.mode, TitanRuntimeMode::PythonFallback | TitanRuntimeMode::RustCanary | TitanRuntimeMode::RustPrimary));
    assert!(runtime.reason.contains("Titan runtime") || runtime.reason.contains("Transformers path"));
}
