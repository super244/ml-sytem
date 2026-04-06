//! Real-time inference telemetry frame.
//!
//! v0.3.0: added `tokens_per_sec`, `batch_size`, `queue_depth_pct`,
//!          and structured `ThermalState`.

use serde::Serialize;

/// Coarse thermal state classification.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThermalState {
    Normal,
    Warm,
    Throttled,
    Emergency,
}

impl ThermalState {
    /// Classify from a temperature in Celsius.
    #[must_use]
    pub fn from_celsius(c: f32) -> Self {
        if c < 75.0 {
            Self::Normal
        } else if c < 85.0 {
            Self::Warm
        } else if c < 92.0 {
            Self::Throttled
        } else {
            Self::Emergency
        }
    }
}

/// Single telemetry snapshot, collected at the cadence of `ui_budget_hz`.
#[derive(Clone, Debug, Serialize)]
pub struct TelemetryFrame {
    /// Core/die temperature in Celsius.
    pub thermals_c: f32,
    /// Memory pressure as a ratio 0.0–1.0  (1.0 = fully saturated).
    pub memory_pressure: f32,
    /// Sustained TFLOPS measured over the last inference window.
    pub flops_tflops: f32,
    /// Number of waiting tasks in the priority scheduler.
    pub queue_depth: usize,
    /// Effective token generation throughput (tokens / second).
    pub tokens_per_sec: f32,
    /// Active batch size (number of sequences in-flight).
    pub batch_size: u16,
    /// Queue depth as a percentage of `max_inflight_tasks` (0–100).
    pub queue_depth_pct: u8,
    /// Structured thermal classification.
    pub thermal_state: ThermalState,
}

impl TelemetryFrame {
    /// Returns `true` when the system should throttle new work.
    #[must_use]
    #[inline]
    pub fn degraded(&self) -> bool {
        self.memory_pressure > 0.9
            || matches!(
                self.thermal_state,
                ThermalState::Throttled | ThermalState::Emergency
            )
    }

    /// Build a zero-value frame (useful for initialisation before the first tick).
    #[must_use]
    pub fn zero(_max_inflight: usize) -> Self {
        Self {
            thermals_c: 0.0,
            memory_pressure: 0.0,
            flops_tflops: 0.0,
            queue_depth: 0,
            tokens_per_sec: 0.0,
            batch_size: 0,
            queue_depth_pct: 0,
            thermal_state: ThermalState::Normal,
        }
    }

    /// Update thermal fields from a new temperature reading.
    pub fn update_thermals(&mut self, celsius: f32) {
        self.thermals_c = celsius;
        self.thermal_state = ThermalState::from_celsius(celsius);
    }

    /// Update queue-depth percentage given the configured maximum.
    pub fn update_queue_pct(&mut self, max_inflight: usize) {
        self.queue_depth_pct = if max_inflight > 0 {
            ((self.queue_depth as f32 / max_inflight as f32) * 100.0).min(100.0) as u8
        } else {
            0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_classification_boundaries() {
        assert_eq!(ThermalState::from_celsius(60.0), ThermalState::Normal);
        assert_eq!(ThermalState::from_celsius(80.0), ThermalState::Warm);
        assert_eq!(ThermalState::from_celsius(88.0), ThermalState::Throttled);
        assert_eq!(ThermalState::from_celsius(95.0), ThermalState::Emergency);
    }

    #[test]
    fn degraded_triggers_on_emergency_thermal() {
        let mut frame = TelemetryFrame::zero(64);
        frame.update_thermals(95.0);
        assert!(frame.degraded());
    }
}
