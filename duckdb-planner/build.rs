use std::path::PathBuf;
use std::process::Command;

/// The `flatbuffers` runtime crate's major.minor (keep in sync with Cargo.toml).
/// `flatc` generates code that calls into that runtime and the two are released
/// in lockstep, so a `flatc` from a different major.minor can emit code that no
/// longer matches the crate API.
const FLATBUFFERS_VERSION: &str = "25.12";

/// Fail the build early with a clear message when the host `flatc` major.minor
/// does not match the runtime crate, rather than letting flatc emit code that
/// fails to compile (or, worse, silently disagrees) against `flatbuffers`.
fn check_flatc_version() {
    let out = Command::new("flatc")
        .arg("--version")
        .output()
        .expect("failed to run flatc (is it installed and on PATH?)");
    let version = String::from_utf8_lossy(&out.stdout);
    // e.g. "flatc version 25.12.19"
    let reported = version.split_whitespace().last().unwrap_or("").trim();
    let host_major_minor = reported.split('.').take(2).collect::<Vec<_>>().join(".");
    assert_eq!(
        host_major_minor, FLATBUFFERS_VERSION,
        "host flatc {reported} does not match the flatbuffers crate {FLATBUFFERS_VERSION}.x; \
         the generated code may not compile. Install a matching flatc \
         (e.g. `brew install flatbuffers`) or bump both together."
    );
}

/// Resolve the FlatBuffers C++ runtime include directory (the one containing
/// `flatbuffers/flatbuffers.h`). Honors `FLATBUFFERS_INCLUDE_DIR`, then falls
/// back to `pkg-config`, so the bridge compiles wherever the C++ headers live.
fn flatbuffers_include_dir() -> String {
    if let Ok(dir) = std::env::var("FLATBUFFERS_INCLUDE_DIR") {
        return dir;
    }
    let out = Command::new("pkg-config")
        .args(["--variable=includedir", "flatbuffers"])
        .output()
        .expect("pkg-config failed; set FLATBUFFERS_INCLUDE_DIR to the dir holding flatbuffers/flatbuffers.h");
    let dir = String::from_utf8(out.stdout).unwrap().trim().to_string();
    assert!(
        !dir.is_empty(),
        "could not locate the flatbuffers C++ headers; set FLATBUFFERS_INCLUDE_DIR"
    );
    dir
}

/// Run `flatc` to generate the Rust reader and the C++ object-API writer for
/// `schema/plan.fbs` into `OUT_DIR`. Returns that directory so the C++ build can
/// add it to its include path.
fn generate_flatbuffers() -> PathBuf {
    check_flatc_version();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let schema = "schema/plan.fbs";

    let rust = Command::new("flatc")
        .args(["--rust", "-o"])
        .arg(&out_dir)
        .arg(schema)
        .status()
        .expect("failed to run flatc (is it installed and on PATH?)");
    assert!(rust.success(), "flatc --rust failed");

    let cpp = Command::new("flatc")
        .args(["--cpp", "--gen-object-api", "--gen-name-strings", "-o"])
        .arg(&out_dir)
        .arg(schema)
        .status()
        .expect("failed to run flatc (is it installed and on PATH?)");
    assert!(cpp.success(), "flatc --cpp failed");

    out_dir
}

fn main() {
    let num_jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());

    let generated_dir = generate_flatbuffers();
    let flatbuffers_inc = flatbuffers_include_dir();

    let duckdb = cmake::Config::new("duckdb-sources")
        .profile("Release")
        .build_target("duckdb_static")
        .build();

    // Build parquet and core_functions extensions
    let build_dir = duckdb.join("build");
    let status = std::process::Command::new("cmake")
        .args([
            "--build",
            build_dir.to_str().unwrap(),
            "--target",
            "parquet_extension",
            "--target",
            "core_functions_extension",
            "--config",
            "Release",
            "--parallel",
            &num_jobs,
        ])
        .status()
        .expect("Failed to build extensions");
    assert!(status.success(), "Failed to build extensions");

    println!(
        "cargo:rustc-link-search=native={}/build/src",
        duckdb.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/extension/parquet",
        duckdb.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/extension/core_functions",
        duckdb.display()
    );
    println!("cargo:rustc-link-lib=static=duckdb_static");
    println!("cargo:rustc-link-lib=static=parquet_extension");
    println!("cargo:rustc-link-lib=static=core_functions_extension");

    let include_path = "duckdb-sources/src/include";

    // autocxx build for auto-generated enum bindings
    let mut autocxx_b =
        autocxx_build::Builder::new("src/duckdb_bridge/duckdb_types.rs", [include_path])
            .build()
            .expect("autocxx build failed");
    autocxx_b
        .std("c++17")
        .warnings(false)
        .compile("duck_planner_autocxx");

    // cxx build for manual bridge functions
    cxx_build::bridge("src/duckdb_bridge/mod.rs")
        .file("src/duckdb_bridge/cpp/bridge.cpp")
        .file("src/duckdb_bridge/cpp/extension_loader.cpp")
        .file("src/duckdb_bridge/cpp/extension.cpp")
        .file("src/duckdb_bridge/cpp/storage_info.cpp")
        .file("src/duckdb_bridge/cpp/transaction_manager.cpp")
        .file("src/duckdb_bridge/cpp/catalog/catalog.cpp")
        .file("src/duckdb_bridge/cpp/catalog/schema_entry.cpp")
        .file("src/duckdb_bridge/cpp/catalog/table_entry.cpp")
        .include(include_path)
        .include("duckdb-sources/extension/core_functions/include")
        .include(&generated_dir)
        .include(&flatbuffers_inc)
        .std("c++17")
        .warnings(false)
        .compile("duck_planner_bridge");

    println!("cargo:rerun-if-changed=src/duckdb_bridge/mod.rs");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/duckdb_types.rs");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/bridge.h");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/bridge.cpp");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/common.h");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/extension_loader.cpp");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/extension.h");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/extension.cpp");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/storage_info.h");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/storage_info.cpp");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/transaction_manager.h");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/transaction_manager.cpp");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/catalog/catalog.h");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/catalog/catalog.cpp");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/catalog/schema_entry.h");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/catalog/schema_entry.cpp");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/catalog/table_entry.h");
    println!("cargo:rerun-if-changed=src/duckdb_bridge/cpp/catalog/table_entry.cpp");
    println!("cargo:rerun-if-changed=schema/plan.fbs");
    println!("cargo:rerun-if-env-changed=FLATBUFFERS_INCLUDE_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=duckdb-sources");
}
