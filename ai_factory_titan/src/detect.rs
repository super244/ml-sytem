use crate::backend::{BackendKind, TitanBackend};
use serde::Serialize;
use std::process::Command;

#[derive(Clone, Debug, Serialize)]
pub struct HardwareProfile {
    pub silicon: String,
    pub cpu_cores: usize,
    pub memory_gb: u64,
    pub gpu_name: Option<String>,
    pub gpu_vendor: Option<String>,
    pub unified_memory: bool,
    pub bandwidth_gbps: Option<u64>,
    pub simd_features: Vec<String>,
    pub vram_usage_mb: Option<u64>,
    pub backend: TitanBackend,
}

fn detect_simd() -> Vec<String> {
    let mut features = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") { features.push("avx".to_string()); }
        if std::is_x86_feature_detected!("avx2") { features.push("avx2".to_string()); }
        if std::is_x86_feature_detected!("avx512f") { features.push("avx512f".to_string()); }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") { features.push("neon".to_string()); }
    }
    features
}

pub fn detect_hardware() -> HardwareProfile {
    let cpu_cores = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let memory_gb = sys_info::mem_info()
        .map(|info| info.total / 1024 / 1024)
        .unwrap_or(0);
    let simd_features = detect_simd();

    #[cfg(target_os = "macos")]
    if let Some(mut profile) = detect_apple(cpu_cores, memory_gb) {
        profile.simd_features = simd_features.clone();
        return profile;
    }

    #[cfg(feature = "cuda")]
    if let Some(mut profile) = detect_cuda(cpu_cores, memory_gb) {
        profile.simd_features = simd_features.clone();
        return profile;
    }

    HardwareProfile {
        silicon: std::env::consts::ARCH.to_string(),
        cpu_cores,
        memory_gb,
        gpu_name: None,
        gpu_vendor: None,
        unified_memory: false,
        bandwidth_gbps: None,
        simd_features,
        vram_usage_mb: None,
        backend: TitanBackend::new(BackendKind::CpuFallback, 100),
    }
}

#[cfg(target_os = "macos")]
fn detect_apple(cpu_cores: usize, memory_gb: u64) -> Option<HardwareProfile> {
    #[cfg(feature = "metal")]
    {
        let device = metal::Device::system_default()?;
        let name = device.name().to_string();
        let bandwidth_gbps = if name.to_lowercase().contains("m5 max") {
            Some(614)
        } else {
            None
        };
        // Very basic mock of VRAM usage for Apple Silicon (unified memory)
        let vram_usage_mb = Some(0); // We'd need IOKit to get real memory usage
        return Some(HardwareProfile {
            silicon: name.clone(),
            cpu_cores,
            memory_gb,
            gpu_name: Some(name),
            gpu_vendor: Some("Apple".to_string()),
            unified_memory: true,
            bandwidth_gbps,
            simd_features: Vec::new(),
            vram_usage_mb,
            backend: TitanBackend::new(BackendKind::Metal, 90),
        });
    }

    #[cfg(not(feature = "metal"))]
    {
        let name = probe_apple_silicon_name()?;
        return Some(HardwareProfile {
            silicon: name.clone(),
            cpu_cores,
            memory_gb,
            gpu_name: Some(name.clone()),
            gpu_vendor: Some("Apple".to_string()),
            unified_memory: true,
            bandwidth_gbps: apple_bandwidth_gbps(&name),
            simd_features: Vec::new(),
            vram_usage_mb: None,
            backend: TitanBackend::new(BackendKind::CpuFallback, 90),
        });
    }
}

#[cfg(not(target_os = "macos"))]
fn detect_apple(_cpu_cores: usize, _memory_gb: u64) -> Option<HardwareProfile> {
    None
}

#[cfg(feature = "cuda")]
fn detect_cuda(cpu_cores: usize, memory_gb: u64) -> Option<HardwareProfile> {
    use cudarc::driver::sys::CUdevice_attribute;

    let device = cudarc::driver::CudaDevice::new(0).ok()?;
    let major = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    let minor = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .ok()?;
    let bus_width_bits = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
        .ok()
        .map(|value| value as u64);
    Some(HardwareProfile {
        silicon: std::env::consts::ARCH.to_string(),
        cpu_cores,
        memory_gb,
        gpu_name: Some(device.name().ok()?),
        gpu_vendor: Some("NVIDIA".to_string()),
        unified_memory: false,
        bandwidth_gbps: bus_width_bits.or(Some(((major * 100) + minor) as u64)),
        simd_features: Vec::new(), // will be overwritten in detect_hardware
        vram_usage_mb: None, // could be obtained via nvml or cudarc
        backend: TitanBackend::new(BackendKind::Cuda, 100),
    })
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn detect_cuda(_cpu_cores: usize, _memory_gb: u64) -> Option<HardwareProfile> {
    None
}

#[cfg(target_os = "macos")]
fn probe_apple_silicon_name() -> Option<String> {
    let output = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[cfg(target_os = "macos")]
fn apple_bandwidth_gbps(name: &str) -> Option<u64> {
    let normalized = name.to_lowercase();
    if normalized.contains("m5 max") {
        Some(614)
    } else if normalized.contains("m4 max") {
        Some(546)
    } else if normalized.contains("m3 max") || normalized.contains("m2 max") || normalized.contains("m1 max") {
        Some(400)
    } else {
        None
    }
}
