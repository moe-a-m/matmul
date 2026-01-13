fn main() {
    let mut build = cc::Build::new();
    build.file("kernels/matmul_kernel.c").opt_level(3);

    // Only add RISC-V flags when cross-compiling to RISC-V
    if std::env::var("TARGET")
        .unwrap_or_default()
        .contains("riscv")
    {
        build.flag("-march=rv64gcv").flag("-mtune=generic");
    }

    build.compile("matmul_kernel");

    // GPU kernel compilation
    if std::env::var("CARGO_FEATURE_GPU").is_ok() {
        let mut gpu_build = cc::Build::new();
        gpu_build
            .file("kernels/tt_xla_backend.cpp")
            .cpp(true)
            .opt_level(3)
            .flag("-std=c++17")
            .flag("-march=native")
            .flag("-mtune=native")
            .flag("-funroll-loops")
            .compile("tt_xla_backend");
    }

    // Add Homebrew library paths for macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-search=/opt/homebrew/lib");
        println!("cargo:rustc-link-search=/urs/local/lib");
    }

    // Link BLIS if feature is enabled
    if std::env::var("CARGO_FEATURE_BLIS").is_ok() {
        println!("cargo:rustc-link-lib=blis");
    }

    println!("cargo:rerun-if-changed=kernels/matmul_kernel.c");
    println!("cargo:rerun-if-changed=kernels/tt_xla_backend.cpp");
    println!("cargo:rerun-if-changed=kernels/matmul_tt.cpp");
}
