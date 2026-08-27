//! The machine running the tests must satisfy the instruction-set floor the
//! binary was compiled against, so the check has to come back clean here.

#[test]
fn build_machine_supports_compiled_features() {
    let missing = dispatch::missing_cpu_features();

    assert!(
        missing.is_empty(),
        "CPU lacks compiled-in features: {missing:?}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_floor_reached_codegen() {
    let floor_compiled = cfg!(target_feature = "sse4.2")
        && cfg!(target_feature = "avx2")
        && cfg!(target_feature = "bmi2");

    assert!(
        floor_compiled,
        "x86-64-v3 floor from .cargo/config.toml (or the ci.sh RUSTFLAGS default) did not reach the compiler"
    );
}
