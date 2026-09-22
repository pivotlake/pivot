//! The buffer pool is faulted in whole while the workers start, so a budget the
//! machine cannot back gets the process OOM-killed mid-boot. The server checks
//! the budget against free memory first and exits with a message instead.

use std::process::Command;

#[test]
fn refuses_to_boot_when_the_buffer_pool_does_not_fit_in_free_memory() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("pivot.yaml");
    std::fs::write(&config, "memory: 1000000g\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .arg("server")
        .arg("--config")
        .arg(&config)
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "server started anyway: {stderr}");
    assert!(stderr.contains("the buffer pool needs"), "{stderr}");
}
