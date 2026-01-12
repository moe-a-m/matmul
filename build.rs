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

    // Add Homebrew library paths for macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-search=/opt/homebrew/lib");
        println!("cargo:rustc-link-search=/urs/local/lib");
    }

    // Link BLIS if feature is enabled
    if std::env::var("CARGO_FEATURE_BLIS").is_ok() {
        println!("cargo:rustc-link-lib=blis");
    }

    print!("cargo:rerun-if-changed=kernels/matmul_kernel.c");
}
