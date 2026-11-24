use std::process::Command;

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_x86 = target.starts_with("x86_64");
    let is_arm = target.starts_with("aarch64") || target.starts_with("arm");
    let is_riscv = target.starts_with("riscv");
    let is_macos = target_os == "macos";

    // ── c_packed kernel (always built) ───────────────────────────────────────
    let mut build = cc::Build::new();
    build
        .file("kernels/matmul_kernel.c")
        .opt_level(3)
        .flag("-funroll-loops")
        .flag("-ffast-math");
    if is_x86 {
        build.flag("-march=native").flag("-mtune=native");
    } else if is_riscv {
        // Enables the RVV path in matmul_kernel.c.
        build.flag("-march=rv64gcv").flag("-mtune=generic");
    } else if is_arm && is_macos {
        build.flag_if_supported("-mcpu=apple-m1");
    }
    build.compile("matmul_kernel");

    // ── avx2_blis kernel (feature `avx2`, x86-64 only) ───────────────────────
    if std::env::var("CARGO_FEATURE_AVX2").is_ok() {
        if !is_x86 {
            println!(
                "cargo:warning=feature `avx2` ignored: avx2_blis_kernel.c needs x86-64 with AVX2/FMA (target is {target})"
            );
        } else {
            let mut avx2 = cc::Build::new();
            avx2.file("kernels/avx2_blis_kernel.c")
                .opt_level(3)
                .flag("-funroll-loops")
                .flag("-ffast-math")
                .flag("-march=native")
                .flag("-mtune=native")
                .flag("-mavx2")
                .flag("-mfma");
            if is_macos {
                // Apple clang needs Homebrew's libomp (untested; see README).
                avx2.flag("-Xpreprocessor").flag("-fopenmp");
                avx2.include("/usr/local/opt/libomp/include");
                println!("cargo:rustc-link-search=/usr/local/opt/libomp/lib");
                println!("cargo:rustc-link-lib=omp");
            } else {
                avx2.flag("-fopenmp");
                println!("cargo:rustc-link-lib=gomp");
            }
            avx2.compile("avx2_blis_kernel");
        }
    }

    // ── BLIS library (feature `blis`) ────────────────────────────────────────
    if std::env::var("CARGO_FEATURE_BLIS").is_ok() {
        if is_macos {
            println!("cargo:rustc-link-search=/opt/homebrew/lib");
            println!("cargo:rustc-link-search=/usr/local/lib");
        }
        println!("cargo:rustc-link-lib=blis");
    }

    // ── Build metadata recorded in every result file ─────────────────────────
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = run(&rustc, &["--version"]).unwrap_or_else(|| "unknown".into());
    let commit = run("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // core.fileMode=false: permission-only changes don't count as modifications.
    let dirty = run(
        "git",
        &[
            "-c",
            "core.fileMode=false",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
    )
    .map(|s| !s.is_empty())
    .unwrap_or(false);
    println!("cargo:rustc-env=MATMUL_RUSTC_VERSION={rustc_version}");
    println!(
        "cargo:rustc-env=MATMUL_GIT_COMMIT={commit}{}",
        if dirty { "-dirty" } else { "" }
    );
    println!("cargo:rustc-env=MATMUL_TARGET={target}");
    println!(
        "cargo:rustc-env=MATMUL_RUSTFLAGS={}",
        std::env::var("CARGO_ENCODED_RUSTFLAGS")
            .unwrap_or_default()
            .replace('\x1f', " ")
    );

    println!("cargo:rerun-if-changed=kernels/matmul_kernel.c");
    println!("cargo:rerun-if-changed=kernels/matmul_kernel.h");
    println!("cargo:rerun-if-changed=kernels/avx2_blis_kernel.c");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}
