//! Launch and supervise a `pivot server` process for benchmarking.
//!
//! pivot-bench measures the real server binary from the outside: it writes a
//! minimal config, spawns the given `pivot` binary as `pivot server`, waits for
//! its listener, and shuts the process down when the handle drops. Measuring
//! (and, for PGO, profiling) the same binary that ships is the point: an
//! in-process stand-in is a differently linked artifact, and its compiled code
//! can diverge from the server's even when every crate is identical.

use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// How long to wait for the spawned server to accept connections. Generous
/// because an instrumented (PGO generation) build boots slowly: it prefaults
/// the same buffer pool as a release build while running instrumented code.
const LISTEN_DEADLINE: Duration = Duration::from_secs(300);

/// How long a clean shutdown may take before the process is killed.
const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(60);

pub struct ServerHandle {
    port: u16,
    child: Child,
    /// Scratch directory holding the generated config, the empty datastore
    /// and the server's log; deleted when the handle drops.
    scratch: tempfile::TempDir,
}

impl ServerHandle {
    pub fn port(&self) -> u16 {
        self.port
    }

    fn server_log(&self) -> String {
        fs::read_to_string(self.scratch.path().join("server.log")).unwrap_or_default()
    }

    /// Wait until the server accepts TCP connections, failing early if the
    /// process exits first. Either failure carries the server's log, which is
    /// otherwise deleted with the scratch directory.
    fn wait_until_listening(&mut self) -> std::io::Result<()> {
        let addr: SocketAddr = format!("127.0.0.1:{}", self.port)
            .parse()
            .expect("valid socket addr");
        let deadline = Instant::now() + LISTEN_DEADLINE;
        while Instant::now() < deadline {
            if TcpStream::connect(addr).is_ok() {
                return Ok(());
            }
            if let Some(status) = self.child.try_wait()? {
                return Err(std::io::Error::other(format!(
                    "pivot server exited with {status} before listening:\n{}",
                    self.server_log()
                )));
            }
            thread::sleep(Duration::from_millis(20));
        }
        Err(std::io::Error::other(format!(
            "timed out waiting for pivot server to listen on {addr}:\n{}",
            self.server_log()
        )))
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        // SIGINT is the server's clean-shutdown signal. A clean exit matters
        // beyond politeness: an instrumented build writes its profile
        // counters only on a normal exit, so killing the process would
        // silently forfeit a PGO profiling run.
        unsafe {
            libc::kill(self.child.id() as libc::pid_t, libc::SIGINT);
        }
        let deadline = Instant::now() + SHUTDOWN_DEADLINE;
        while Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Pick a free local port by binding to `:0` and dropping the listener; the
/// kernel will not immediately reuse the port for the brief window before the
/// spawned server re-binds it.
fn pick_free_port() -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

/// Where the scratch datastore lives: beside `source`, so the table and the data
/// it was built from sit on the same filesystem.
///
/// This is not a detail. A benchmark's data goes on the machine's fast scratch
/// disk while the default temp directory is usually the root volume, so a table
/// placed there measures the root volume's bandwidth rather than the engine. On
/// one c8gd run that was the difference between a 450ms and a 13.7s cold query,
/// with nothing in the output to say which had been measured. The chosen path is
/// printed for the same reason.
///
/// A source tree that cannot be written to (a read-only dataset mount) falls
/// back to the default temp directory, which is correct but may not measure what
/// the caller intended, so say so.
fn scratch_dir(source: &Path) -> std::io::Result<tempfile::TempDir> {
    let beside = source.parent().unwrap_or(source);
    match tempfile::tempdir_in(beside) {
        Ok(dir) => {
            eprintln!("benchmark datastore: {}", dir.path().display());
            Ok(dir)
        }
        Err(_) => {
            let dir = tempfile::tempdir()?;
            eprintln!(
                "note: {} is not writable, so the benchmark datastore goes to {} \
                 instead of beside the source data; if that is a different device, \
                 cold timings measure it and not the source's",
                beside.display(),
                dir.path().display(),
            );
            Ok(dir)
        }
    }
}

/// Start `server_bin` on a free port over an empty scratch datastore,
/// returning once its listener accepts connections. The catalog starts empty;
/// the runner sends `CREATE TABLE` over the wire to populate it.
///
/// `memory` is the buffer-pool budget in the config file's `memory` syntax
/// (`30%`, `16g`); `None` leaves the server on its default share of the
/// machine. `source` is the benchmark's data directory; the scratch datastore
/// is placed beside it (see [`scratch_dir`]).
pub fn start(
    server_bin: &Path,
    workers: Option<usize>,
    memory: Option<&str>,
    source: &Path,
) -> std::io::Result<ServerHandle> {
    let port = pick_free_port()?;
    let scratch = scratch_dir(source)?;
    let data_dir = scratch.path().join("data");
    fs::create_dir(&data_dir)?;

    let workers_line = match workers {
        Some(count) => format!("workers: {count}\n"),
        None => String::new(),
    };
    let memory_line = match memory {
        Some(budget) => format!("memory: {budget}\n"),
        None => String::new(),
    };
    // Background maintenance is off: the benchmark datastore adopts
    // pre-existing parquet, and the compacter would otherwise wake mid-run
    // (its poll fires seconds after CREATE) and rewrite the tables while
    // queries are being timed, spending disk reads and cache space the
    // measurements then absorb. Vacuum only ever follows compaction, but its
    // poll loop is just as pointless during a measurement, so it is off too.
    let config = format!(
        "\
server:
  bind: 127.0.0.1:{port}
{workers_line}{memory_line}datastores:
  default:
    kind: pivot
    location: {data_dir}
    default: true
    compact: false
    vacuum: false
users:
  pivot:
    auth:
      method: trust
",
        data_dir = data_dir.display(),
    );
    let config_path = scratch.path().join("config.yaml");
    fs::write(&config_path, config)?;

    // The child's output goes to a log file, not this process's stdout: the
    // benchmark's stdout is parsed by scripts, and the server's tracing lines
    // would corrupt it. The log surfaces in errors while the handle is alive.
    let log = fs::File::create(scratch.path().join("server.log"))?;
    let child = Command::new(server_bin)
        .arg("server")
        .arg("--config")
        .arg(&config_path)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()
        .map_err(|e| {
            std::io::Error::other(format!("failed to spawn {}: {e}", server_bin.display()))
        })?;

    let mut handle = ServerHandle {
        port,
        child,
        scratch,
    };
    handle.wait_until_listening()?;
    Ok(handle)
}
