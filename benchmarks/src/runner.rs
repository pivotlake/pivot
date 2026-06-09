//! Generic per-query timed runner shared across suites.
//!
//! A [`Suite`] describes a benchmark: which directory holds its `.sql` /
//! `.tsv` pairs, the `setup.sql` template, and the queries it ships. The
//! runner connects to a pivotdb server, runs the setup once, then loops over
//! each query recording wall-clock per iteration. Per-query output is also
//! captured and compared against the expected `.tsv` (or written to it under
//! `--update-results`) so a faster regression that silently broke the result
//! is caught.

use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use thiserror::Error;
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

#[derive(Debug, Error)]
pub enum Error {
    #[error("io error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("postgres error: {0}")]
    Postgres(#[from] tokio_postgres::Error),
    #[error("query {id} returned a non-row message we couldn't decode: {message}")]
    NonRowMessage { id: String, message: String },
    #[error("query {id} produced incorrect output\n\nExpected:\n{expected}\n\nActual:\n{actual}\n")]
    ResultMismatch {
        id: String,
        expected: String,
        actual: String,
    },
    #[error("no expected result file at {path}; rerun with --update-results to create it")]
    MissingExpected { path: PathBuf },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A single benchmark query. `expected_path` is the file the runner reads (or
/// writes, with `--update-results`) to verify accuracy.
#[derive(Debug, Clone)]
pub struct Query {
    pub id: String,
    pub sql_path: PathBuf,
    pub expected_path: PathBuf,
}

/// Description of a benchmark suite.
#[derive(Debug)]
pub struct Suite {
    pub name: String,
    pub setup_sql_path: PathBuf,
    pub queries: Vec<Query>,
}

/// Discover queries by listing `q*.sql` files in `suite_dir`. Each one is
/// paired with `<stem>.tsv` for the expected output. Sort by ID so the run
/// order is deterministic.
///
/// This is what makes a suite "just a directory": adding a query is dropping
/// in `qNN.sql` + `qNN.tsv`, no code change needed.
pub fn discover_suite(name: &str, suite_dir: &Path) -> Result<Suite> {
    let setup_sql_path = suite_dir.join("setup.sql");
    let entries = std::fs::read_dir(suite_dir).map_err(|source| Error::Io {
        path: suite_dir.to_path_buf(),
        source,
    })?;

    let mut queries = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: suite_dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("sql") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) if s.starts_with('q') => s.to_string(),
            _ => continue,
        };
        queries.push(Query {
            id: stem.clone(),
            sql_path: path.clone(),
            expected_path: suite_dir.join(format!("{stem}.tsv")),
        });
    }
    queries.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(Suite {
        name: name.to_string(),
        setup_sql_path,
        queries,
    })
}

/// Per-iteration timing for a single query, in milliseconds.
#[derive(Debug, Clone)]
pub struct QueryRun {
    pub id: String,
    pub iterations_ms: Vec<u128>,
}

impl QueryRun {
    /// First-iteration wall-clock, in ms — the "cold" number: what a fresh
    /// process pays before any caches (page cache, planner thread-locals,
    /// allocator arenas) are warm. Always defined; the runner refuses to run
    /// with zero iterations.
    pub fn cold_ms(&self) -> u128 {
        self.iterations_ms[0]
    }

    /// Arithmetic mean of every iteration *after* the first, in ms — the
    /// "hot" number: steady-state cost with caches warm. `None` when the run
    /// did a single iteration (there's nothing after the first to average).
    pub fn hot_ms(&self) -> Option<f64> {
        let tail = &self.iterations_ms[1..];
        if tail.is_empty() {
            return None;
        }
        let sum: u128 = tail.iter().sum();
        Some(sum as f64 / tail.len() as f64)
    }
}

/// The result of running every query in the suite once.
#[derive(Debug)]
pub struct SuiteRun {
    pub suite: String,
    pub queries: Vec<QueryRun>,
}

/// Knobs that apply to every suite.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub source: PathBuf,
    pub iterations: u32,
    pub sleep_ms: Option<u64>,
    pub update_results: bool,
    /// Skip the output-vs-expected comparison entirely (still records timings).
    pub skip_check: bool,
    /// `None` → run every query in `suite.queries`.
    pub query_filter: Option<Vec<String>>,
    /// Before each query, evict pivot's file cache (`SELECT drop_cache()`) and
    /// flush the OS page cache, so each query's first iteration is a true cold
    /// read — without restarting the warm server. Needs passwordless sudo.
    pub drop_caches: bool,
}

/// Clear pivot's file cache *and* the OS page cache so the next query reads cold
/// from disk, without tearing down the server. Best-effort on the page cache
/// (needs root) — a failure warns and continues, like ClickBench's runner.
async fn cold_clear(client: &Client) -> Result<()> {
    client.simple_query("SELECT drop_cache()").await?;
    let dropped = std::process::Command::new("sh")
        .arg("-c")
        .arg("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
        .status();
    if !matches!(dropped, Ok(status) if status.success()) {
        eprintln!("  warning: OS page-cache drop failed (need sudo/Linux); continuing");
    }
    Ok(())
}

fn read_to_string(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_string(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    std::fs::write(path, contents).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Run a `simple_query` and concatenate every `Row` message as TSV. Tabs
/// separate columns; rows are newline-terminated. Mirrors the format the
/// existing dispatch bench uses, so eyeballing diffs is straightforward.
async fn collect_tsv(client: &Client, sql: &str, id: &str) -> Result<String> {
    let messages = client.simple_query(sql).await?;
    let mut out = String::new();
    for msg in messages {
        match msg {
            SimpleQueryMessage::Row(row) => {
                for col in 0..row.len() {
                    if col > 0 {
                        out.push('\t');
                    }
                    if let Some(s) = row.get(col) {
                        out.push_str(s);
                    }
                }
                out.push('\n');
            }
            SimpleQueryMessage::CommandComplete(_) | SimpleQueryMessage::RowDescription(_) => {}
            other => {
                return Err(Error::NonRowMessage {
                    id: id.to_string(),
                    message: format!("{other:?}"),
                });
            }
        }
    }
    Ok(out)
}

/// Time one query: run it `iterations` times back-to-back, capturing the
/// resulting TSV from the first iteration for comparison. Subsequent
/// iterations skip the (already-paid) materialisation of `Vec<Row>` for
/// timing-only purposes — but the simpler approach of decoding every time
/// keeps the wall-clock numbers reflective of what the wire path costs end
/// to end, so we do that.
async fn run_query(
    client: &Client,
    query: &Query,
    opts: &RunOptions,
) -> Result<(QueryRun, String)> {
    let sql = read_to_string(&query.sql_path)?;
    let mut iterations_ms = Vec::with_capacity(opts.iterations as usize);
    let mut last_output = String::new();

    println!("=== Query {} ===", query.id);
    for i in 0..opts.iterations {
        let start = Instant::now();
        let output = collect_tsv(client, &sql, &query.id).await?;
        let elapsed = start.elapsed().as_millis();
        iterations_ms.push(elapsed);
        println!(
            "[{}/{}] Query {} — {}ms",
            i + 1,
            opts.iterations,
            query.id,
            elapsed
        );
        last_output = output;
        if let Some(ms) = opts.sleep_ms
            && i + 1 < opts.iterations
        {
            sleep(Duration::from_millis(ms));
        }
    }

    Ok((
        QueryRun {
            id: query.id.clone(),
            iterations_ms,
        },
        last_output,
    ))
}

/// Either `--update-results`-write or read-and-compare the captured TSV
/// against the on-disk expected file.
fn check_or_update_expected(query: &Query, actual: &str, update: bool) -> Result<()> {
    if update {
        write_string(&query.expected_path, actual)?;
        println!(
            "  wrote expected result → {}",
            query.expected_path.display()
        );
        return Ok(());
    }
    let expected = match std::fs::read_to_string(&query.expected_path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::MissingExpected {
                path: query.expected_path.clone(),
            });
        }
        Err(source) => {
            return Err(Error::Io {
                path: query.expected_path.clone(),
                source,
            });
        }
    };
    if expected.trim_end() == actual.trim_end() {
        Ok(())
    } else {
        Err(Error::ResultMismatch {
            id: query.id.clone(),
            expected,
            actual: actual.to_string(),
        })
    }
}

/// Run every query in `suite` (filtered by `opts.query_filter`).
///
/// `setup_template` reads the suite's `setup.sql`, substitutes `{source}`
/// with the data path, and ships it as one `simple_query`. Tables persist on
/// the server's `ParquetCatalog` for the lifetime of the process — fine,
/// since we tear the server down at the end of `main`.
pub async fn run_suite(port: u16, suite: &Suite, opts: &RunOptions) -> Result<SuiteRun> {
    let (client, connection) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(port)
        .user("bench")
        .dbname("bench")
        .connect(NoTls)
        .await?;
    let conn_handle = tokio::spawn(connection);

    let setup_template = read_to_string(&suite.setup_sql_path)?;
    let setup_sql = setup_template.replace("{source}", &opts.source.display().to_string());
    client.simple_query(&setup_sql).await?;

    let mut runs = Vec::with_capacity(suite.queries.len());
    for query in &suite.queries {
        if let Some(filter) = &opts.query_filter
            && !filter.contains(&query.id)
        {
            continue;
        }
        if opts.drop_caches {
            cold_clear(&client).await?;
        }
        let (run, last_output) = run_query(&client, query, opts).await?;
        if !opts.skip_check {
            check_or_update_expected(query, &last_output, opts.update_results)?;
        }
        runs.push(run);
    }

    drop(client);
    let _ = conn_handle.await;

    Ok(SuiteRun {
        suite: suite.name.clone(),
        queries: runs,
    })
}
