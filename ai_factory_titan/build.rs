fn main() {
    if std::env::var_os("CARGO_FEATURE_CPP").is_some() {
        cc::Build::new()
            .cpp(true)
            .flag_if_supported("-std=c++17")
            .file("src/cpp/kernels.cpp")
            .compile("titan_cpp_kernels");
    }
}
