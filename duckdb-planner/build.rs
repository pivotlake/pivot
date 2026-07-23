fn main() {
    let num_jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());

    let mut duckdb_config = cmake::Config::new("duckdb-sources");
    duckdb_config
        .profile("Release")
        .build_target("duckdb_static");

    // Route DuckDB's C/C++ compiles through the same content-addressed build
    // cache that wraps rustc, so a fresh checkout restores the object files
    // instead of recompiling all of DuckDB from scratch. cmake spawns the
    // compiler directly, so without a launcher the cache never sees these.
    // Builds that want fresh, uncached output clear RUSTC_WRAPPER and so skip
    // the launcher automatically.
    if let Ok(wrapper) = std::env::var("RUSTC_WRAPPER") {
        if !wrapper.is_empty() {
            duckdb_config
                .define("CMAKE_C_COMPILER_LAUNCHER", &wrapper)
                .define("CMAKE_CXX_COMPILER_LAUNCHER", &wrapper);
        }
    }

    let duckdb = duckdb_config.build();

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
        .include("third_party")
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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=duckdb-sources");
}
