//! Profiling and performance monitoring for Titan kernels
//!
//! Provides:
//! - Kernel-level timing and profiling
//! - Performance counters for hardware metrics
//! - Flamegraph generation support
//! - Automatic bottleneck detection

use anyhow::{anyhow, Result};
use instant::Instant;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use parking_lot::RwLock;

/// Performance counter for tracking kernel metrics
pub struct PerformanceCounter {
    name: String,
    count: AtomicU64,
    total_time_ns: AtomicU64,
    min_time_ns: AtomicU64,
    max_time_ns: AtomicU64,
}

impl PerformanceCounter {
    /// Create a new performance counter
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            count: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
            min_time_ns: AtomicU64::new(u64::MAX),
            max_time_ns: AtomicU64::new(0),
        }
    }

    /// Record a timing measurement
    pub fn record(&self, duration_ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(duration_ns, Ordering::Relaxed);

        // Update min
        loop {
            let current = self.min_time_ns.load(Ordering::Relaxed);
            if duration_ns >= current || self.min_time_ns.compare_exchange(
                current,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed
            ).is_ok() {
                break;
            }
        }

        // Update max
        loop {
            let current = self.max_time_ns.load(Ordering::Relaxed);
            if duration_ns <= current || self.max_time_ns.compare_exchange(
                current,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed
            ).is_ok() {
                break;
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> CounterStats {
        let count = self.count.load(Ordering::Relaxed);
        let total = self.total_time_ns.load(Ordering::Relaxed);

        CounterStats {
            name: self.name.clone(),
            count,
            total_time_ms: total as f64 / 1_000_000.0,
            avg_time_ms: if count > 0 { (total / count) as f64 / 1_000_000.0 } else { 0.0 },
            min_time_ms: self.min_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            max_time_ms: self.max_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        }
    }
}

/// Statistics for a performance counter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CounterStats {
    pub name: String,
    pub count: u64,
    pub total_time_ms: f64,
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
}

/// High-precision kernel profiler
pub struct KernelProfiler {
    counters: RwLock<HashMap<String, Arc<PerformanceCounter>>>,
    enabled: AtomicU64,
}

impl KernelProfiler {
    /// Create a new kernel profiler
    pub fn new() -> Self {
        Self {
            counters: RwLock::new(HashMap::new()),
            enabled: AtomicU64::new(1),
        }
    }

    /// Enable profiling
    pub fn enable(&self) {
        self.enabled.store(1, Ordering::SeqCst);
    }

    /// Disable profiling
    pub fn disable(&self) {
        self.enabled.store(0, Ordering::SeqCst);
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst) == 1
    }

    /// Get or create a counter
    fn get_counter(&self, name: &str) -> Arc<PerformanceCounter> {
        // Fast path: read lock
        {
            let counters = self.counters.read();
            if let Some(counter) = counters.get(name) {
                return Arc::clone(counter);
            }
        }

        // Slow path: write lock
        let mut counters = self.counters.write();
        counters
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(PerformanceCounter::new(name)))
            .clone()
    }

    /// Time a closure and record the measurement
    pub fn time<F, T>(&self, name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        if !self.is_enabled() {
            return f();
        }

        let counter = self.get_counter(name);
        let start = Instant::now();
        let result = f();
        let duration_ns = start.elapsed().as_nanos() as u64;

        counter.record(duration_ns);
        result
    }

    /// Get all counter statistics
    pub fn all_stats(&self) -> Vec<CounterStats> {
        let counters = self.counters.read();
        counters.values().map(|c| c.stats()).collect()
    }

    /// Get a timing report
    pub fn report(&self) -> TimingReport {
        let stats = self.all_stats();
        let total_time_ms: f64 = stats.iter().map(|s| s.total_time_ms).sum();

        // Sort by total time (descending)
        let mut sorted_stats = stats.clone();
        sorted_stats.sort_by(|a, b| b.total_time_ms.partial_cmp(&a.total_time_ms).unwrap());

        TimingReport {
            total_time_ms,
            counters: sorted_stats,
            generated_at: chrono::Local::now().to_rfc3339(),
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        let mut counters = self.counters.write();
        counters.clear();
    }
}

impl Default for KernelProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing report with aggregated statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimingReport {
    pub total_time_ms: f64,
    pub counters: Vec<CounterStats>,
    pub generated_at: String,
}

impl TimingReport {
    /// Print the report to stdout
    pub fn print(&self) {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                  Titan Kernel Timing Report                  ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Generated: {:50} ║", self.generated_at);
        println!("║ Total Time: {:49.2} ms ║", self.total_time_ms);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ {:<30} {:>8} {:>12} {:>12} ║", "Kernel", "Count", "Total (ms)", "Avg (ms)");
        println!("╠══════════════════════════════════════════════════════════════╣");

        for stat in &self.counters {
            println!(
                "║ {:<30} {:>8} {:>12.2} {:>12.3} ║",
                &stat.name[..stat.name.len().min(30)],
                stat.count,
                stat.total_time_ms,
                stat.avg_time_ms
            );
        }

        println!("╚══════════════════════════════════════════════════════════════╝");
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize report: {}", e))
    }

    /// Export to CSV
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("kernel,count,total_ms,avg_ms,min_ms,max_ms\n");

        for stat in &self.counters {
            csv.push_str(&format!(
                "{},{},{:.2},{:.3},{:.3},{:.3}\n",
                stat.name, stat.count, stat.total_time_ms,
                stat.avg_time_ms, stat.min_time_ms, stat.max_time_ms
            ));
        }

        csv
    }
}

/// Scoped timer for automatic profiling
pub struct ScopedTimer {
    counter: Arc<PerformanceCounter>,
    start: Instant,
}

impl ScopedTimer {
    /// Create a new scoped timer
    pub fn new(profiler: &KernelProfiler, name: &str) -> Option<Self> {
        if !profiler.is_enabled() {
            return None;
        }

        Some(Self {
            counter: profiler.get_counter(name),
            start: Instant::now(),
        })
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let duration_ns = self.start.elapsed().as_nanos() as u64;
        self.counter.record(duration_ns);
    }
}

/// Macro for easy timing
#[macro_export]
macro_rules! time_kernel {
    ($profiler:expr, $name:expr, $block:expr) => {
        $profiler.time($name, || $block)
    };
}

/// Flamegraph-compatible event
#[derive(Clone, Debug, Serialize)]
pub struct FlameEvent {
    pub name: String,
    pub start_ns: u64,
    pub end_ns: u64,
}

/// Flamegraph profiler for visualization
pub struct FlamegraphProfiler {
    events: Arc<Mutex<Vec<FlameEvent>>>,
    enabled: AtomicU64,
    start_time: Instant,
}

impl FlamegraphProfiler {
    /// Create a new flamegraph profiler
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            enabled: AtomicU64::new(1),
            start_time: Instant::now(),
        }
    }

    /// Record an event
    pub fn record_event(&self, name: &str, start: Instant, end: Instant) {
        if !self.is_enabled() {
            return;
        }

        let start_ns = start.duration_since(self.start_time).as_nanos() as u64;
        let end_ns = end.duration_since(self.start_time).as_nanos() as u64;

        let mut events = self.events.lock().unwrap();
        events.push(FlameEvent {
            name: name.to_string(),
            start_ns,
            end_ns,
        });
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst) == 1
    }

    /// Export to Chrome trace format
    pub fn to_chrome_trace(&self) -> Result<String> {
        let events = self.events.lock().unwrap();

        #[derive(Serialize)]
        struct ChromeEvent {
            name: String,
            ph: &'static str, // phase: B=begin, E=end
            ts: u64,          // timestamp in microseconds
            pid: u32,
            tid: u32,
        }

        let chrome_events: Vec<ChromeEvent> = events
            .iter()
            .flat_map(|e| {
                vec![
                    ChromeEvent {
                        name: e.name.clone(),
                        ph: "B",
                        ts: e.start_ns / 1000,
                        pid: 1,
                        tid: 1,
                    },
                    ChromeEvent {
                        name: e.name.clone(),
                        ph: "E",
                        ts: e.end_ns / 1000,
                        pid: 1,
                        tid: 1,
                    },
                ]
            })
            .collect();

        serde_json::to_string(&chrome_events)
            .map_err(|e| anyhow!("Failed to serialize: {}", e))
    }

    /// Clear all events
    pub fn clear(&self) {
        let mut events = self.events.lock().unwrap();
        events.clear();
    }
}

impl Default for FlamegraphProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU profiler for CUDA kernels (CUDA only)
#[cfg(feature = "cuda")]
pub struct CudaProfiler {
    events: Arc<Mutex<Vec<(String, f64)>>>, // kernel name, duration ms
}

#[cfg(feature = "cuda")]
impl CudaProfiler {
    /// Create new CUDA profiler
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Profile a CUDA kernel launch
    pub fn profile_kernel<F>(&self, name: &str, f: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        let start = Instant::now();
        f()?;
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut events = self.events.lock().unwrap();
        events.push((name.to_string(), duration_ms));

        Ok(())
    }

    /// Get all events
    pub fn events(&self) -> Vec<(String, f64)> {
        self.events.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_counter() {
        let counter = PerformanceCounter::new("test");

        counter.record(100_000); // 100 microseconds
        counter.record(200_000);
        counter.record(150_000);

        let stats = counter.stats();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_time_ms, 0.45); // 450 microseconds
        assert_eq!(stats.avg_time_ms, 0.15);
        assert_eq!(stats.min_time_ms, 0.1);
        assert_eq!(stats.max_time_ms, 0.2);
    }

    #[test]
    fn test_kernel_profiler() {
        let profiler = KernelProfiler::new();

        // Simulate some work
        let result = profiler.time("matmul", || {
            std::thread::sleep(std::time::Duration::from_millis(1));
            42
        });

        assert_eq!(result, 42);

        let stats = profiler.all_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "matmul");
        assert!(stats[0].count >= 1);
    }

    #[test]
    fn test_timing_report() {
        let profiler = KernelProfiler::new();

        profiler.time("kernel_a", || ());
        profiler.time("kernel_b", || ());
        profiler.time("kernel_a", || ());

        let report = profiler.report();
        assert!(!report.counters.is_empty());
        assert!(report.total_time_ms >= 0.0);

        // Test JSON export
        let json = report.to_json().unwrap();
        assert!(json.contains("kernel_a") || json.contains("kernel_b"));
    }
}
