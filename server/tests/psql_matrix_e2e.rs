//! Meta-command coverage across psql client versions.
//!
//! The catalog queries psql generates depend on the client's version (the
//! advertised server version is fixed), so a meta-command can work with one
//! client and fail to bind with another. Each version's psql runs from the
//! official postgres docker image against the shared test server. Without
//! docker, or when an image cannot be pulled, that lane skips with a note,
//! like the plain-psql tests do when psql is not installed.

mod common;

use std::process::{Command, Output};

use common::{Conn, conn, server_port};
use rstest::rstest;

const PSQL_IMAGES: &[&str] = &[
    "postgres:13-alpine",
    "postgres:14-alpine",
    "postgres:15-alpine",
    "postgres:16-alpine",
    "postgres:17-alpine",
];

/// The meta-commands every supported psql client must answer, each with a
/// string its rendered output must contain.
const SUPPORTED_COMMANDS: &[(&str, &str)] = &[
    ("\\dn", "matrix_schema"),
    ("\\dt matrix_schema.*", "matrix_people"),
    ("\\dt", "matrix_visible"),
    ("\\d", "matrix_visible"),
    ("\\d matrix_visible", "id"),
    ("\\d matrix_schema.matrix_people", "display_name"),
    ("\\du", "List of roles"),
    ("\\du+", "pivot"),
    ("\\dv", ""),
    ("\\ds", ""),
    ("\\dT", ""),
];

/// Whether the docker CLI exists at all; without it every lane skips.
fn docker_available() -> bool {
    match Command::new("docker").arg("--version").output() {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Pull `image`, returning whether it is now locally available. A failed pull
/// (offline or rate-limited environment) skips the lane rather than failing
/// the test: the lane tests psql compatibility, not registry reachability.
fn pull_image(image: &str) -> bool {
    let output = Command::new("docker")
        .args(["pull", "--quiet", image])
        .output()
        .expect("docker pull spawns after docker_available succeeded");
    output.status.success()
}

fn run_containerized_psql(image: &str, port: u16, command: &str) -> Output {
    Command::new("docker")
        .args([
            "run",
            "--rm",
            "--network=host",
            image,
            "psql",
            "-X",
            "--no-password",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--username",
            "pivot",
            "--dbname",
            "test",
            "--set",
            "ON_ERROR_STOP=1",
            "--command",
            command,
        ])
        .output()
        .expect("docker run spawns after docker_available succeeded")
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn meta_commands_work_across_psql_client_versions(#[future] conn: Conn) {
    if !docker_available() {
        eprintln!("skipping the psql client matrix because docker is not installed");
        return;
    }
    conn.simple_query("CREATE SCHEMA IF NOT EXISTS matrix_schema")
        .await
        .unwrap();
    conn.simple_query(
        "CREATE TABLE IF NOT EXISTS matrix_schema.matrix_people \
         (id BIGINT, display_name VARCHAR)",
    )
    .await
    .unwrap();
    conn.simple_query("CREATE TABLE IF NOT EXISTS matrix_visible (id BIGINT)")
        .await
        .unwrap();

    let mut failures = Vec::new();
    for image in PSQL_IMAGES {
        if !pull_image(image) {
            eprintln!("skipping {image} because the image could not be pulled");
            continue;
        }
        for (command, expected) in SUPPORTED_COMMANDS {
            let output = run_containerized_psql(image, server_port(), command);
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if !output.status.success() {
                failures.push(format!("{image} `{command}` failed:\n{stderr}"));
            } else if !stdout.contains(expected) {
                failures.push(format!(
                    "{image} `{command}` output lacks {expected:?}:\n{stdout}"
                ));
            }
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
