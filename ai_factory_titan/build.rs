fn main() {
    // Re-run when C++ source changes.
    println!("cargo:rerun-if-changed=src/cpp/kernels.cpp");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=CXXFLAGS");

    if std::env::var_os("CARGO_FEATURE_CPP").is_some() {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O3")
            .flag_if_supported("-march=native") // enable AVX2/NEON automatically
            .flag_if_supported("-ffast-math")   // unsafe but dramatically faster for kernels
            .flag_if_supported("-funroll-loops")
            .file("src/cpp/kernels.cpp");

        // Silence unused-parameter warnings from generated FFI glue.
        build.flag_if_supported("-Wno-unused-parameter");

        build.compile("titan_cpp_kernels");
        println!("cargo:rustc-link-lib=static=titan_cpp_kernels");
    }
}
