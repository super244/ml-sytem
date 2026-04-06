//! Criterion benchmarks for Titan CPU/SIMD kernel throughput.
//!
//! Run with: `cargo bench -p ai_factory_titan`
//! HTML report: `target/criterion/report/index.html`

use ai_factory_titan::cpu_kernels::{
    dot_f32, matmul_f32, rms_norm_f32, silu_f32, softmax_f32, vec_add_f32,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ─── Helper ───────────────────────────────────────────────────────────────────

fn rand_vec(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.1 + 1.0).sin())
        .collect()
}

// ─── Dot product ──────────────────────────────────────────────────────────────

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f32");
    for size in [256usize, 1024, 4096, 16384] {
        let a = rand_vec(size);
        let b = rand_vec(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &_| {
            bench.iter(|| dot_f32(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

// ─── Matrix multiply ──────────────────────────────────────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");
    for dim in [32usize, 64, 128, 256] {
        let a = rand_vec(dim * dim);
        let b = rand_vec(dim * dim);
        group.throughput(Throughput::Elements((dim * dim * dim * 2) as u64)); // FLOPs
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, &d| {
            bench.iter(|| matmul_f32(black_box(&a), d, d, black_box(&b), d).unwrap())
        });
    }
    group.finish();
}

// ─── RMS norm ─────────────────────────────────────────────────────────────────

fn bench_rms_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_norm_f32");
    for size in [512usize, 2048, 8192] {
        let v = rand_vec(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &_| {
            bench.iter(|| rms_norm_f32(black_box(&v), 1e-6))
        });
    }
    group.finish();
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_f32");
    for size in [128usize, 1024, 8192, 32768] {
        let v = rand_vec(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &_| {
            bench.iter(|| softmax_f32(black_box(&v)))
        });
    }
    group.finish();
}

// ─── SiLU ─────────────────────────────────────────────────────────────────────

fn bench_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu_f32");
    for size in [512usize, 4096, 16384] {
        let v = rand_vec(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &_| {
            bench.iter(|| silu_f32(black_box(&v)))
        });
    }
    group.finish();
}

// ─── Vec add ──────────────────────────────────────────────────────────────────

fn bench_vec_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_add_f32");
    for size in [1024usize, 16384, 65536] {
        let a = rand_vec(size);
        let b = rand_vec(size);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &_| {
            bench.iter(|| vec_add_f32(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

// ─── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    kernel_benches,
    bench_dot,
    bench_matmul,
    bench_rms_norm,
    bench_softmax,
    bench_silu,
    bench_vec_add,
);
criterion_main!(kernel_benches);
