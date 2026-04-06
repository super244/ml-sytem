use std::env;

fn main() {
    println!("cargo:rerun-if-changed=src/cpp/kernels.cpp");
    println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
    println!("cargo:rerun-if-changed=src/metal/shaders.metal");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=METAL_SDK");

    // C++ kernel compilation with feature detection
    if env::var_os("CARGO_FEATURE_CPP").is_some() {
        compile_cpp_kernels();
    }

    // CUDA kernel compilation
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        compile_cuda_kernels();
    }

    // Metal shader compilation (macOS only)
    #[cfg(target_os = "macos")]
    if env::var_os("CARGO_FEATURE_METAL").is_some() {
        compile_metal_shaders();
    }

    // Link flags for different platforms
    configure_link_flags();
}

fn compile_cpp_kernels() {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("-funroll-loops")
        .flag_if_supported("-fopenmp")
        .flag_if_supported("-fPIC")
        .file("src/cpp/kernels.cpp");

    // Architecture-specific optimizations
    let target_arch = env::var("TARGET").unwrap_or_default();
    
    if target_arch.contains("x86_64") {
        // x86_64: Enable AVX-512 if available
        build
            .flag_if_supported("-mavx512f")
            .flag_if_supported("-mavx512bw")
            .flag_if_supported("-mavx512vnni")
            .flag_if_supported("-mavx2")
            .flag_if_supported("-mfma")
            .flag_if_supported("-mf16c");
    } else if target_arch.contains("aarch64") || target_arch.contains("arm64") {
        // ARM64: Enable NEON and SVE
        build
            .flag_if_supported("-march=armv8.2-a+fp16+dotprod")
            .flag_if_supported("-mfpu=neon-fp16");
    }

    // OpenMP for parallelization
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "macos" {
        build.flag_if_supported("-fopenmp");
        println!("cargo:rustc-link-lib=gomp");
    }

    build.flag_if_supported("-Wno-unused-parameter");
    build.flag_if_supported("-Wno-ignored-attributes");

    build.compile("titan_cpp_kernels");
    println!("cargo:rustc-link-lib=static=titan_cpp_kernels");
    println!("cargo:rustc-cfg=feature=\"cpp-kernels\"");
}

fn compile_cuda_kernels() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .or_else(|_| env::var("CUDA_ROOT"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Set CUDA compiler
    let nvcc = format!("{}/bin/nvcc", cuda_path);
    
    if std::path::Path::new(&nvcc).exists() {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-cfg=feature=\"cuda-kernels\"");
        
        // Note: Actual CUDA compilation would happen here
        // For now, we assume pre-compiled PTX or JIT compilation
    }
}

#[cfg(target_os = "macos")]
fn compile_metal_shaders() {
    // Metal shaders are compiled at runtime on macOS
    // Just verify the SDK is available
    let sdk_path = env::var("METAL_SDK_PATH")
        .unwrap_or_else(|_| "/System/Library/Frameworks/Metal.framework".to_string());
    
    if std::path::Path::new(&sdk_path).exists() {
        println!("cargo:rustc-cfg=feature=\"metal-shaders\"");
    }
}

fn configure_link_flags() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        "linux" => {
            // Linux-specific linking
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=pthread");
        }
        _ => {}
    }
}
