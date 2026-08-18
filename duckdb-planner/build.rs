fn main() {
    let num_jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());

    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let ccache_base_dir = find_ccache_base_dir(std::path::Path::new(&crate_dir));

    let mut duckdb_config = cmake::Config::new("duckdb-sources");
    duckdb_config
        .profile("Release")
        .define("BUILD_EXTENSIONS", "icu")
        .build_target("duckdb_static");

    // Route DuckDB's C/C++ compiles through ccache so a fresh checkout restores
    // the object files instead of recompiling all of DuckDB from scratch. cmake
    // spawns the compiler directly, so without a launcher the cache never sees
    // these compiles.
    if let Some(ccache) = find_ccache() {
        // DuckDB is around 2400 translation units pinned to a submodule commit,
        // so every checkout sitting on that commit compiles byte-identical
        // objects. ccache turns all but the first of those into cache hits.
        //
        // Those compiles carry the checkout's absolute path in two places: the
        // include flags cmake passes, and the bodies of the unity files it
        // generates, which include each source by absolute path. Both would
        // otherwise land in the hash and miss on every sibling checkout.
        // CCACHE_BASEDIR makes ccache rewrite paths beneath it, in the command
        // line and in preprocessed output alike, as relative to the directory
        // being compiled in. Since that directory sits under the same root, the
        // part of the path naming the checkout appears on both sides of the
        // relative path and cancels out.
        duckdb_config
            .define("CMAKE_C_COMPILER_LAUNCHER", &ccache)
            .define("CMAKE_CXX_COMPILER_LAUNCHER", &ccache)
            .env("CCACHE_BASEDIR", &ccache_base_dir);
    }

    let duckdb = duckdb_config.build();

    // Build the static extensions loaded by extension_loader.cpp.
    let build_dir = duckdb.join("build");
    let status = std::process::Command::new("cmake")
        .args([
            "--build",
            build_dir.to_str().unwrap(),
            "--target",
            "parquet_extension",
            "--target",
            "core_functions_extension",
            "--target",
            "icu_extension",
            "--config",
            "Release",
            "--parallel",
            &num_jobs,
        ])
        // The launcher cmake recorded while configuring runs for this build too,
        // so it needs the same view of where paths are rooted.
        .env("CCACHE_BASEDIR", &ccache_base_dir)
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
    println!(
        "cargo:rustc-link-search=native={}/build/extension/icu",
        duckdb.display()
    );
    println!("cargo:rustc-link-lib=static=duckdb_static");
    println!("cargo:rustc-link-lib=static=parquet_extension");
    println!("cargo:rustc-link-lib=static=core_functions_extension");
    println!("cargo:rustc-link-lib=static=icu_extension");

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
        .include("duckdb-sources/extension/icu/include")
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

/// The directory ccache expresses compile paths relative to.
///
/// This has to be an ancestor of every checkout meant to share cached objects,
/// so it is the repository root rather than the checkout's own root: worktrees
/// live inside it, and picking the checkout root instead leaves each one with a
/// cache only it can use. A checkout placed outside the repository still caches
/// against itself, just not against its siblings.
fn find_ccache_base_dir(crate_dir: &std::path::Path) -> std::path::PathBuf {
    let repository_root = std::process::Command::new("git")
        .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .current_dir(crate_dir)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            let git_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            std::path::PathBuf::from(git_dir)
        })
        .and_then(|git_dir| git_dir.parent().map(std::path::Path::to_path_buf));

    repository_root.unwrap_or_else(|| {
        crate_dir
            .parent()
            .expect("the crate directory always has a parent")
            .to_path_buf()
    })
}

/// The ccache to hand cmake, or `None` when it is not installed and DuckDB has
/// to be compiled the slow way.
fn find_ccache() -> Option<String> {
    let installed = std::process::Command::new("ccache")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());

    installed.then(|| "ccache".to_string())
}
