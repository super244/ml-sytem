use ai_factory_titan::{
    adamw_step_f32, batch_softmax_f32, dot_f32, fused_rms_norm_silu_f32, gelu_f32, matmul_f32,
    parse_gguf_header,
    quantization::estimate_q4_bytes,
    sample_token, KvCache, KvCacheConfig, Priority, QuantizationFormat, QuantizedTensorLayout,
    SamplerConfig, TensorShape, TitanEngineDescriptor, TitanRuntimeMode, TitanRuntimePlan,
    TitanScheduler,
};

// ─── Engine descriptor ────────────────────────────────────────────────────────

#[test]
fn titan_engine_descriptor_reports_expected_capabilities() {
    let d = TitanEngineDescriptor::local_default();
    assert_eq!(d.decode_model, "llama.cpp-inspired");
    assert!(d.acceleration.rust_fallback);
    assert!(d.supported_quantizations.len() >= 4);
    assert!(d.gguf_support);
    assert!(d.kv_cache);
    assert!(d.acceleration.rayon_parallel);
    assert!(d.acceleration.paged_kv_cache);
    assert!(d.acceleration.cpu_adamw);
}

#[test]
fn cloud_profiles_have_larger_context() {
    let local  = TitanEngineDescriptor::local_default();
    let a100   = TitanEngineDescriptor::cloud_a100();
    assert!(a100.max_batch_tokens >= local.max_batch_tokens);
}

// ─── Quantization Layout ─────────────────────────────────────────────────────

#[test]
fn quantized_layout_estimates_storage() {
    let layout = QuantizedTensorLayout::for_format(QuantizationFormat::Q4K);
    let bytes  = layout.estimated_bytes(TensorShape { rows: 128, cols: 256 });
    assert_eq!(bytes, estimate_q4_bytes(128, 256));
    assert!(bytes > 0);
}

#[test]
fn bf16_layout_is_2_bytes_per_element() {
    let layout = QuantizedTensorLayout::for_format(QuantizationFormat::BF16);
    // BF16 is a float format: bytes = element_count × 2.
    let bytes = layout.estimated_bytes(TensorShape { rows: 4, cols: 8 });
    assert_eq!(bytes, 4 * 8 * 2);
}

// ─── CPU kernels ──────────────────────────────────────────────────────────────

#[test]
fn cpu_dot_and_matmul_are_consistent() {
    let dot = dot_f32(&[1.0, 2.0, 3.0], &[0.5, 1.5, -1.0]).expect("dot should succeed");
    assert!((dot - 0.5).abs() < 1e-5);

    let out = matmul_f32(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[5.0, 6.0, 7.0, 8.0], 2)
        .expect("matmul should succeed");
    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn gelu_is_positive_for_positive_input() {
    let vals = gelu_f32(&[0.5, 1.0, 2.0]);
    for v in vals {
        assert!(v > 0.0, "GELU should be positive for positive input");
    }
}

#[test]
fn batch_softmax_rows_sum_to_one() {
    let input = vec![1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0];
    let out = batch_softmax_f32(&input, 2, 3).unwrap();
    let row0: f32 = out[..3].iter().sum();
    let row1: f32 = out[3..].iter().sum();
    assert!((row0 - 1.0).abs() < 1e-6);
    assert!((row1 - 1.0).abs() < 1e-6);
}

#[test]
fn fused_rms_norm_silu_matches_separate_ops() {
    use ai_factory_titan::{rms_norm_f32, silu_f32};
    let input = vec![1.0f32, -2.0, 3.0, -4.0];
    let fused    = fused_rms_norm_silu_f32(&input, 1e-6);
    let separate = silu_f32(&rms_norm_f32(&input, 1e-6));
    for (f, s) in fused.iter().zip(separate.iter()) {
        assert!((f - s).abs() < 1e-5, "fused={f} vs separate={s}");
    }
}

#[test]
fn adamw_decreases_parameter_toward_gradient() {
    let mut p = vec![1.0f32];
    let g = vec![0.5f32];
    let mut m = vec![0.0f32];
    let mut v = vec![0.0f32];
    adamw_step_f32(&mut p, &g, &mut m, &mut v, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1).unwrap();
    assert!(p[0] < 1.0, "parameter should decrease after gradient step");
}

// ─── GGUF ─────────────────────────────────────────────────────────────────────

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

// ─── KV Cache ─────────────────────────────────────────────────────────────────

#[test]
fn kv_cache_tracks_recent_tokens_and_stats() {
    let mut cache = KvCache::new(KvCacheConfig {
        page_size: 2,
        max_tokens: 4,
        heads: 4,
        head_dim: 8,
        max_pages: None,
    });
    cache.append(11).expect("append one");
    cache.append(17).expect("append two");
    cache.append(23).expect("append three");

    assert_eq!(cache.recent(2), vec![17, 23]);
    assert_eq!(cache.stats().stored_tokens, 3);
    assert_eq!(cache.stats().pages_in_use, 2);
}

#[test]
fn kv_cache_sliding_window_evicts_oldest_page() {
    let mut cache = KvCache::new(KvCacheConfig {
        page_size: 2,
        max_tokens: 100,
        heads: 1,
        head_dim: 4,
        max_pages: Some(2),
    });
    for id in 0u32..6 {
        cache.append(id).expect("append");
    }
    // 3 pages were created; first should have been evicted.
    assert_eq!(cache.stats().evicted_pages, 1);
    assert_eq!(cache.stats().pages_in_use, 2);
}

// ─── Sampler ─────────────────────────────────────────────────────────────────

#[test]
fn sampler_prefers_best_scoring_token_with_repetition_penalty() {
    let params = SamplerConfig {
        temperature: 0.7,
        top_k: 1, // greedy: argmax after penalty
        top_p: 0.9,
        min_p: 0.0,
        typical_p: 0.0,
        repetition_penalty: 2.5,
        seed: None,
    };
    let pick =
        sample_token(&[0.2, 1.4, 1.1, 0.7], &[1], &params).expect("sampler should pick a token");
    assert_eq!(pick, 2);
}

#[test]
fn greedy_sampler_picks_argmax() {
    let pick = sample_token(&[0.1, 0.9, 0.5], &[], &SamplerConfig::greedy())
        .expect("greedy should pick");
    assert_eq!(pick, 1);
}

// ─── Runtime ─────────────────────────────────────────────────────────────────

#[test]
fn runtime_plan_tracks_runtime_flag_modes() {
    let runtime = TitanRuntimePlan::current();
    assert!(matches!(
        runtime.mode,
        TitanRuntimeMode::PythonFallback
            | TitanRuntimeMode::RustCanary
            | TitanRuntimeMode::RustPrimary
    ));
    assert!(!runtime.version.is_empty());
    assert!(runtime.rayon_kernels);
}

// ─── Scheduler ───────────────────────────────────────────────────────────────

#[test]
fn scheduler_can_be_created_without_a_tokio_runtime() {
    let _ = TitanScheduler::new(2);
}

#[tokio::test]
async fn scheduler_completes_submitted_work() {
    let scheduler = TitanScheduler::new(2);
    let (task, rx) = ai_factory_titan::GpuTask::new("decode-step", Priority::Normal);
    scheduler.try_submit(task).expect("submit should succeed");
    let reply = rx.await.expect("reply");
    assert!(reply.contains("decode-step"));
}

#[tokio::test]
async fn scheduler_high_priority_fast_path() {
    let scheduler = TitanScheduler::new(4);
    let reply = scheduler.submit_high("prefill").await.expect("high priority");
    assert!(reply.contains("prefill"));
}
