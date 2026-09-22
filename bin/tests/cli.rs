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
    assert!(stdout.contains("[default: pivot]"), "{stdout}");
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
        "pivot accepted --warehouse for a pivot datastore"
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
             datastores:\n  default:\n    kind: pivot\n    location: {}\n    default: true\n\
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
