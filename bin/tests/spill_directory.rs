//! A spill directory that spill files do not work in would fail the first
//! query that runs short of memory. The server checks the directory when it
//! starts and exits with a message instead.

use std::process::Command;

#[test]
fn refuses_to_start_when_spill_files_cannot_be_created_in_the_spill_directory() {
    let dir = tempfile::tempdir().unwrap();
    let occupied = dir.path().join("occupied");
    std::fs::write(&occupied, "a file, where a directory is wanted").unwrap();
    let spill_directory = occupied.join("spill");
    let config = dir.path().join("pivot.yaml");
    std::fs::write(
        &config,
        format!(
            "memory: 64m\nworkers: 1\nspill_directory: {}\n",
            spill_directory.display()
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .arg("server")
        .arg("--config")
        .arg(&config)
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    let refusal = format!(
        "spill directory `{}` cannot be used: creating the directory: ",
        spill_directory.display()
    );
    assert!(!output.status.success(), "server started anyway: {stderr}");
    assert!(stderr.contains(&refusal), "{stderr}");
}
