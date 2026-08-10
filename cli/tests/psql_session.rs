//! End-to-end tests of the `pivot` binary: a real engine behind the built-in
//! shell, driven through a piped stdin the way `pivot < script.sql` is.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_script(script: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_pivot-cli"))
        .args(["--memory", "1g", "--workers", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn a_piped_session_creates_inserts_and_reads_back_in_psql_format() {
    let output = run_script(
        "CREATE TABLE events (id INT, name TEXT);\n\
         INSERT INTO events VALUES (7, 'first');\n\
         SELECT id, name FROM events;\n",
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("CREATE TABLE\n"), "stdout: {stdout}");
    assert!(stdout.contains("INSERT 0 1\n"), "stdout: {stdout}");
    assert!(
        stdout.contains(" id | name\n----+-------\n  7 | first\n(1 row)\n"),
        "stdout: {stdout}"
    );
    assert!(stderr.contains("removed on exit"), "stderr: {stderr}");
}

#[test]
fn timing_toggles_and_reports_query_time() {
    let output = run_script("\\timing\nSELECT 1;\n");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(stdout.contains("Timing is on.\n"), "stdout: {stdout}");
    assert!(stdout.contains("Time: "), "stdout: {stdout}");
}

#[test]
fn help_lists_the_meta_commands() {
    let output = run_script("\\help\n");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    for command in ["\\h", "\\q", "\\timing"] {
        assert!(stdout.contains(command), "stdout: {stdout}");
    }
}

#[test]
fn a_bare_quit_with_or_without_semicolon_ends_the_session() {
    for spelling in ["quit", "exit;", "QUIT ;"] {
        let output = run_script(&format!("{spelling}\nSELECT 1;\n"));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{spelling}: stdout: {stdout}");
        assert!(!stdout.contains("(1 row)"), "{spelling}: stdout: {stdout}");
    }
}

#[test]
fn a_quit_mid_statement_hints_at_backslash_q_instead_of_quitting() {
    let output = run_script("SELECT 1\nexit\n+ 1;\n");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Use \\q to quit."), "stderr: {stderr}");
    assert!(stdout.contains(" 2"), "stdout: {stdout}");
}

#[test]
fn a_query_error_prints_like_psql_and_the_script_continues() {
    let output = run_script("SELECT * FROM missing;\nSELECT 2 + 2;\n");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(stderr.contains("ERROR:  "), "stderr: {stderr}");
    assert!(stdout.contains(" 4"), "stdout: {stdout}");
}
