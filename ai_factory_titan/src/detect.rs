use crate::backend::{BackendKind, TitanBackend};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct HardwareProfile {
    pub silicon: String,
    pub cpu_cores: usize,
    pub memory_gb: u64,
    pub gpu_name: Option<String>,
    pub gpu_vendor: Option<String>,
    pub unified_memory: bool,
    pub bandwidth_gbps: Option<u64>,
    pub backend: TitanBackend,
}

pub fn detect_hardware() -> HardwareProfile {
    let cpu_cores = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let memory_gb = sys_info::mem_info()
        .map(|info| info.total / 1024 / 1024)
        .unwrap_or(0);

    #[cfg(target_os = "macos")]
    if let Some(profile) = detect_apple(cpu_cores, memory_gb) {
        return profile;
    }

    #[cfg(feature = "cuda")]
    if let Some(profile) = detect_cuda(cpu_cores, memory_gb) {
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
        return Some(HardwareProfile {
            silicon: name.clone(),
            cpu_cores,
            memory_gb,
            gpu_name: Some(name),
            gpu_vendor: Some("Apple".to_string()),
            unified_memory: true,
            bandwidth_gbps,
            backend: TitanBackend::new(BackendKind::Metal, 90),
        });
    }

    #[cfg(not(feature = "metal"))]
    {
        let _ = (cpu_cores, memory_gb);
        None
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
        backend: TitanBackend::new(BackendKind::Cuda, 100),
    })
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn detect_cuda(_cpu_cores: usize, _memory_gb: u64) -> Option<HardwareProfile> {
    None
}
