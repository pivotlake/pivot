#![cfg(unix)]

use std::fs;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn open_help_lists_resource_limits() {
    let output = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .args(["open", "--help"])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("--memory <SIZE>"), "{stdout}");
    assert!(stdout.contains("--workers <COUNT>"), "{stdout}");
    assert!(stdout.contains("--spill-directory <DIRECTORY>"), "{stdout}");
}

#[test]
fn open_help_lists_the_datastore_kinds_and_iceberg_credentials() {
    let output = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .args(["open", "--help"])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("--kind <KIND>"), "{stdout}");
    assert!(stdout.contains("[default: pivotlake]"), "{stdout}");
    assert!(stdout.contains("--warehouse <WAREHOUSE>"), "{stdout}");
    for variable in [
        datastore_iceberg::env::TOKEN_VAR,
        datastore_iceberg::env::CREDENTIAL_VAR,
        datastore_iceberg::env::OAUTH2_SERVER_URI_VAR,
        datastore_iceberg::env::OAUTH2_SCOPE_VAR,
    ] {
        assert!(
            stdout.contains(variable),
            "{variable} missing from:\n{stdout}"
        );
    }
}

#[test]
fn open_rejects_a_warehouse_for_a_pivot_datastore() {
    let output = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .args(["open", "/tmp/pivot", "--warehouse", "s3://lake"])
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "pivot accepted --warehouse for a pivotlake datastore"
    );
    assert!(
        stderr.contains("--warehouse applies only to --kind iceberg"),
        "{stderr}"
    );
}

#[test]
fn open_rejects_conflicting_iceberg_credentials() {
    let output = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .args(["open", "--kind", "iceberg", "http://127.0.0.1:1"])
        .env(datastore_iceberg::env::TOKEN_VAR, "t0k3n")
        .env(datastore_iceberg::env::CREDENTIAL_VAR, "id:secret")
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "pivot accepted a token and a credential together"
    );
    assert!(stderr.contains("set exactly one of"), "{stderr}");
}

#[test]
fn open_rejects_zero_workers() {
    let output = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .args(["open", "/tmp/pivot", "--workers", "0"])
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "pivot accepted zero workers");
    assert!(stderr.contains("--workers"), "{stderr}");
}

/// `pivot open` on a small instance over `datastore`, with `arguments` after it.
fn open_command(datastore: &std::path::Path, arguments: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pivot"));
    command
        .arg("open")
        .arg(datastore)
        .args(["--memory", "64m", "--workers", "1"])
        .args(arguments);
    command
}

#[test]
fn open_runs_commands_in_order() {
    let directory = tempfile::tempdir().unwrap();

    let output = open_command(
        directory.path(),
        &[
            "-c",
            "CREATE TABLE t (a BIGINT)",
            "-c",
            "INSERT INTO t VALUES (7)",
            "-c",
            "SELECT a FROM t",
        ],
    )
    .output()
    .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.ends_with(" a \n---\n 7\n(1 row)\n\n"), "{stdout}");
}

#[test]
fn open_runs_a_file_and_its_unterminated_last_statement() {
    let directory = tempfile::tempdir().unwrap();
    let script = directory.path().join("script.sql");
    fs::write(
        &script,
        "SELECT\n  41 AS first;\n\\timing on\nSELECT 42 AS last",
    )
    .unwrap();

    let output = open_command(
        &directory.path().join("db"),
        &["-f", script.to_str().unwrap()],
    )
    .output()
    .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.contains("    41\n"), "{stdout}");
    assert!(stdout.contains("Timing is on."), "{stdout}");
    assert!(stdout.contains("   42\n"), "{stdout}");
}

#[test]
fn open_reads_a_script_from_stdin_with_dash() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = open_command(directory.path(), &["-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    std::io::Write::write_all(&mut child.stdin.take().unwrap(), b"SELECT 42 AS answer;").unwrap();
    let output = child.wait_with_output().unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.contains("     42\n"), "{stdout}");
}

#[test]
fn open_interleaves_commands_and_files() {
    let directory = tempfile::tempdir().unwrap();
    let insert = directory.path().join("insert.sql");
    fs::write(&insert, "INSERT INTO t VALUES (7);").unwrap();

    let output = open_command(
        &directory.path().join("db"),
        &[
            "-c",
            "CREATE TABLE t (a BIGINT)",
            "-f",
            insert.to_str().unwrap(),
            "-c",
            "SELECT a FROM t",
        ],
    )
    .output()
    .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.ends_with(" 7\n(1 row)\n\n"), "{stdout}");
}

#[test]
fn open_stops_at_the_first_failed_statement() {
    let directory = tempfile::tempdir().unwrap();

    let failed = open_command(
        directory.path(),
        &[
            "-c",
            "SELECT * FROM missing",
            "-c",
            "CREATE TABLE t (a BIGINT)",
        ],
    )
    .output()
    .unwrap();
    let probe = open_command(directory.path(), &["-c", "SELECT * FROM t"])
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&failed.stderr);
    assert!(!failed.status.success(), "{failed:?}");
    assert!(stderr.contains("ERROR:"), "{stderr}");
    assert!(!probe.status.success(), "the command after the failure ran");
}

#[test]
fn open_rejects_a_missing_script_file() {
    let directory = tempfile::tempdir().unwrap();

    let output = open_command(directory.path(), &["-f", "/nonexistent/script.sql"])
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{output:?}");
    assert!(
        stderr.contains("pivot: /nonexistent/script.sql: No such file"),
        "{stderr}"
    );
}

#[test]
fn open_takes_a_spill_directory_and_creates_it() {
    let directory = tempfile::tempdir().unwrap();
    let spill_directory = directory.path().join("spill");

    let output = open_command(
        &directory.path().join("data"),
        &[
            "--spill-directory",
            spill_directory.to_str().unwrap(),
            "-c",
            "SELECT 1",
        ],
    )
    .output()
    .unwrap();

    assert!(output.status.success(), "{output:?}");
    assert!(spill_directory.is_dir());
}

#[test]
fn open_refuses_a_spill_directory_that_cannot_be_used() {
    let directory = tempfile::tempdir().unwrap();
    let occupied = directory.path().join("occupied");
    fs::write(&occupied, "a file, where a directory is wanted").unwrap();
    let spill_directory = occupied.join("spill");

    let output = open_command(
        &directory.path().join("data"),
        &[
            "--spill-directory",
            spill_directory.to_str().unwrap(),
            "-c",
            "SELECT 1",
        ],
    )
    .output()
    .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    let refusal = format!(
        "spill directory `{}` cannot be used: creating the directory: ",
        spill_directory.display()
    );
    assert!(!output.status.success(), "{output:?}");
    assert!(stderr.contains(&refusal), "{stderr}");
}

/// The index of the database at `datastore`: what records its table format
/// and where each of its tables is kept.
fn read_database_index(datastore: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(datastore.join("_pivot_manifest.json")).unwrap()).unwrap()
}

/// Open a database that does not exist yet with `arguments`, and hand back
/// the table format it was created in.
fn create_database_with(arguments: &[&str]) -> String {
    let directory = tempfile::tempdir().unwrap();
    let datastore = directory.path().join("db");
    let output = open_command(&datastore, &[arguments, &["-c", "SELECT 1"]].concat())
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let index = read_database_index(&datastore);
    index["table_format"].as_str().unwrap().to_string()
}

#[test]
fn open_takes_the_table_format_of_a_new_database_with_its_alias() {
    let directory = tempfile::tempdir().unwrap();

    let unnamed = create_database_with(&[]);
    let delta = create_database_with(&["--table-format", "delta"]);
    let aliased = create_database_with(&["--table-format", "deltalake"]);
    let iceberg = create_database_with(&["--table-format", "iceberg"]);
    let unknown = open_command(
        &directory.path().join("db"),
        &["--table-format", "hudi", "-c", "SELECT 1"],
    )
    .output()
    .unwrap();

    assert_eq!(unnamed, "iceberg");
    assert_eq!(delta, "delta");
    assert_eq!(aliased, "delta");
    assert_eq!(iceberg, "iceberg");
    let stderr = String::from_utf8_lossy(&unknown.stderr);
    assert!(!unknown.status.success(), "{unknown:?}");
    assert!(stderr.contains("--table-format"), "{stderr}");
}

#[test]
fn a_format_reaches_the_pivotlake_target_and_is_refused_for_a_catalog() {
    let directory = tempfile::tempdir().unwrap();
    let datastore = directory.path().join("lake");

    let created = open_command(
        &datastore,
        &["--table-format", "delta", "-c", "CREATE TABLE t (a BIGINT)"],
    )
    .output()
    .unwrap();
    let refused = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .args(["open", "--kind", "iceberg", "--table-format", "iceberg"])
        .arg("https://catalog.example.com/api")
        .output()
        .unwrap();

    assert!(created.status.success(), "{created:?}");
    let index = read_database_index(&datastore);
    let table_directories = index["table_locations"].as_object().unwrap();
    let table_directory = table_directories.values().next().unwrap().as_str().unwrap();
    assert_eq!(index["table_format"], "delta");
    assert!(datastore.join(table_directory).join("_delta_log").is_dir());
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        !refused.status.success(),
        "pivot accepted --table-format for a catalog"
    );
    assert!(
        stderr.contains("--table-format applies only to --kind pivotlake"),
        "{stderr}"
    );
}

#[test]
fn server_command_runs_until_terminated() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);

    let directory = tempfile::tempdir().unwrap();
    let datastore = directory.path().join("datastore");
    let metastore = directory.path().join("metastore.yaml");
    fs::write(&metastore, "users: {}\n").unwrap();
    let config = directory.path().join("pivot.yaml");
    fs::write(
        &config,
        format!(
            "memory: 64m\nworkers: 1\nserver:\n  bind: {address}\n\
             datastores:\n  default:\n    kind: pivotlake\n    location: {}\n    default: true\n\
             users:\n  pivot:\n    auth:\n      method: trust\n\
             metastore:\n  kind: file\n  path: {}\n",
            datastore.display(),
            metastore.display()
        ),
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_pivot"))
        .args(["server", "--config"])
        .arg(&config)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    wait_until_listening(&mut child, address);
    let result = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    assert_eq!(result, 0, "failed to send SIGTERM");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "pivot server exited with {status}");
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("pivot server did not stop after SIGTERM");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_until_listening(child: &mut std::process::Child, address: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(address).is_ok() {
            return;
        }
        if let Some(status) = child.try_wait().unwrap() {
            panic!("pivot server exited with {status} before listening");
        }
        thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
    panic!("pivot server did not listen on {address}");
}
