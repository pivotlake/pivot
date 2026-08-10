//! Generic per-query timed runner shared across suites.
//!
//! A [`Suite`] describes a benchmark: which directory holds its `.sql` /
//! `.tsv` pairs, the `setup.sql` template, and the queries it ships. The
//! runner connects to a pivotdb server, runs the setup once, then loops over
//! each query recording wall-clock per iteration. Per-query output is also
//! captured and compared against the expected `.tsv` (or written to it under
//! `--update-results`) so a faster regression that silently broke the result
//! is caught.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use flate2::read::MultiGzDecoder;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

use crate::server_handle::ServerHandle;

/// How many `INSERT`s the load keeps in flight, each on its own connection.
fn in_flight_inserts() -> usize {
    std::env::var("PIVOT_BENCH_INSERTS_IN_FLIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(8)
}

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

/// How a suite loads its data by `INSERT` rather than reading pre-written
/// Parquet in place, declared in the suite's `load.conf` as `<table> <batch>`.
/// The documents are the newline-delimited JSON under `--source`, sent as
/// batched `INSERT ... VALUES`, then compacted into target-sized files.
#[derive(Debug, Clone)]
pub struct LoadSpec {
    /// Table the documents are inserted into.
    pub table: String,
    /// Documents per `INSERT` statement. Each statement is its own file, so
    /// this trades statement size against how many small files compaction then
    /// merges.
    pub batch_rows: usize,
}

/// Description of a benchmark suite.
#[derive(Debug)]
pub struct Suite {
    pub name: String,
    pub setup_sql_path: PathBuf,
    pub queries: Vec<Query>,
    /// Present when the suite loads its data through `INSERT` (see [`LoadSpec`]).
    pub load: Option<LoadSpec>,
}

/// Read a suite's optional `load.conf`: one line, `<table> <batch_rows>`.
fn read_load_spec(suite_dir: &Path) -> Result<Option<LoadSpec>> {
    let path = suite_dir.join("load.conf");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(Error::Io { path, source }),
    };
    let spec_line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .unwrap_or_else(|| panic!("load.conf has no `<table> <batch_rows>` line: {text:?}"));
    let mut fields = spec_line.split_whitespace();
    let table = fields.next().unwrap_or_default().to_string();
    let batch_rows = fields
        .next()
        .and_then(|n| n.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            panic!("load.conf must read `<table> <batch_rows>`, got: {spec_line:?}")
        });
    Ok(Some(LoadSpec { table, batch_rows }))
}

/// Discover queries by listing `.sql` files in `suite_dir`. Every `*.sql`
/// other than `setup.sql` is treated as a query whose ID is the file stem
/// (e.g. `q07`); each one is paired with `<stem>.tsv` for the expected
/// output. Sort by ID so the run order is deterministic.
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
        // Skip setup.sql and the `*-duckdb.sql` overrides — those carry
        // DuckDB-only query text (e.g. q42-duckdb.sql wraps EventTime in
        // ClickHouse's toDateTime); pivot runs the matching shared `qNN.sql`.
        // Without this, a no-`--query` run (e.g. `just pgo-gen`) would try to
        // plan them and abort.
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) if s != "setup" && !s.ends_with("-duckdb") => s.to_string(),
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
        load: read_load_spec(suite_dir)?,
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
    /// Statement run once, untimed, after setup and before the first query.
    pub warmup: Option<String>,
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

/// Split a setup script into the statements it declares. The server prepares one
/// statement per query, so a suite that declares more than one table, or loads
/// one from another, has to send them separately. Line comments come off first:
/// a comment's prose carries semicolons of its own, and they are not statement
/// ends. This assumes no statement holds `--` or `;` inside a string literal,
/// which is true of every suite here.
fn setup_statements(script: &str) -> Vec<String> {
    let statements = script
        .lines()
        .map(|line| line.split_once("--").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n");
    statements
        .split(';')
        .map(str::trim)
        .filter(|statement| !statement.is_empty())
        .map(str::to_string)
        .collect()
}

/// Load every newline-delimited JSON document under `source` into `spec.table`
/// as batched `INSERT ... VALUES`. A `.json.gz` file is decompressed on the fly;
/// a plain `.json` is read as-is. Files are loaded in name order so a run is
/// reproducible.
async fn load_documents(port: u16, source: &Path, spec: &LoadSpec) -> Result<()> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(source)
        .map_err(|source_err| Error::Io {
            path: source.to_path_buf(),
            source: source_err,
        })?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            let name = path.to_string_lossy();
            name.ends_with(".json") || name.ends_with(".json.gz")
        })
        .collect();
    files.sort();

    // One connection carries one statement at a time, so overlapping inserts
    // needs a connection each. Appends do not conflict: a commit takes the next
    // log version and, when a concurrent writer took it first, reloads and
    // retries on top. The only consequence of overlapping them is that rows land
    // in an unspecified order, which no query in a suite depends on.
    let in_flight = in_flight_inserts();
    let mut clients = Vec::with_capacity(in_flight);
    for _ in 0..in_flight {
        clients.push(Arc::new(connect(port).await?));
    }
    // The permits are what bound memory: each insert in flight holds its batch
    // as SQL text here and again on the server.
    let inflight = Arc::new(Semaphore::new(in_flight));
    let mut running: JoinSet<Result<usize>> = JoinSet::new();
    let mut batch: Vec<String> = Vec::with_capacity(spec.batch_rows);
    let mut sent = 0usize;

    for path in &files {
        let file = std::fs::File::open(path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(BufReader::new(MultiGzDecoder::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };
        for line in reader.lines() {
            let document = line.map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            if document.trim().is_empty() {
                continue;
            }
            batch.push(document);
            if batch.len() == spec.batch_rows {
                let documents = std::mem::replace(&mut batch, Vec::with_capacity(spec.batch_rows));
                spawn_insert(&mut running, &inflight, &clients, sent, spec, documents).await;
                sent += 1;
            }
        }
    }
    if !batch.is_empty() {
        spawn_insert(&mut running, &inflight, &clients, sent, spec, batch).await;
    }

    let mut inserted = 0usize;
    while let Some(finished) = running.join_next().await {
        inserted += finished.expect("an insert task is not cancelled or panicked")?;
    }
    println!("loaded {inserted} documents into {}", spec.table);
    Ok(())
}

/// Wait for a free slot, then start one batch's `INSERT` on the next connection
/// in the pool. Connections are taken round-robin, so a slow statement holds up
/// only its own.
async fn spawn_insert(
    running: &mut JoinSet<Result<usize>>,
    inflight: &Arc<Semaphore>,
    clients: &[Arc<Client>],
    sent: usize,
    spec: &LoadSpec,
    documents: Vec<String>,
) {
    let permit = inflight
        .clone()
        .acquire_owned()
        .await
        .expect("the semaphore outlives every insert");
    let client = clients[sent % clients.len()].clone();
    let table = spec.table.clone();
    running.spawn(async move {
        let inserted = insert_documents(&client, &table, &documents).await;
        drop(permit);
        inserted
    });
}

/// Open one connection to the benchmark server, driving its protocol task in the
/// background for as long as the returned client lives.
async fn connect(port: u16) -> Result<Client> {
    let (client, connection) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(port)
        .user("pivot")
        .dbname("bench")
        .connect(NoTls)
        .await?;
    tokio::spawn(connection);
    Ok(client)
}

/// Insert one batch of JSON documents as a single `INSERT ... VALUES`. Each
/// document is a string literal the column's `VARIANT` type parses as JSON; the
/// only thing to escape for a SQL string literal is the single quote.
async fn insert_documents(client: &Client, table: &str, documents: &[String]) -> Result<usize> {
    let mut sql = String::from("INSERT INTO ");
    sql.push_str(table);
    sql.push_str(" VALUES ");
    for (i, document) in documents.iter().enumerate() {
        if i > 0 {
            sql.push(',');
        }
        sql.push_str("('");
        sql.push_str(&document.replace('\'', "''"));
        sql.push_str("')");
    }
    client.simple_query(&sql).await?;
    Ok(documents.len())
}

/// Run every query in `suite` (filtered by `opts.query_filter`).
///
/// `setup_template` reads the suite's `setup.sql`, substitutes `{source}`
/// with the data path, and ships each statement as its own `simple_query` (the
/// server plans one statement at a time; a multi-table suite's setup holds one
/// CREATE TABLE per table). Tables persist on
/// the server's `DeltaDatastore` for the lifetime of the process, fine,
/// since we tear the server down at the end of `main`.
pub async fn run_suite(
    server: &ServerHandle,
    suite: &Suite,
    opts: &RunOptions,
) -> Result<SuiteRun> {
    let (client, connection) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(server.port())
        .user("pivot")
        .dbname("bench")
        .connect(NoTls)
        .await?;
    let conn_handle = tokio::spawn(connection);

    let setup_template = read_to_string(&suite.setup_sql_path)?;
    let setup_sql = setup_template.replace("{source}", &opts.source.display().to_string());
    for statement in setup_statements(&setup_sql) {
        client.simple_query(&statement).await?;
    }

    // A suite with a `load.conf` populates its table by INSERT instead of
    // reading Parquet in place: send the documents in batches, then compact the
    // many small files each batch left into target-sized ones before querying.
    if let Some(spec) = &suite.load {
        let load_start = Instant::now();
        load_documents(server.port(), &opts.source, spec).await?;
        client
            .simple_query(&format!("COMPACT {} FINAL", spec.table))
            .await?;
        // The ingest (INSERT + compaction) is itself a benchmarked cost for a
        // suite loaded this way, so report it the way a query iteration is
        // reported. `=== Load ... ===` keeps it out of the query lines.
        println!(
            "=== Load {} — {}ms ===",
            spec.table,
            load_start.elapsed().as_millis()
        );
    }

    if let Some(warmup) = &opts.warmup {
        client.simple_query(warmup).await?;
    }

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
        if let Some(ms) = opts.sleep_ms
            && !runs.is_empty()
        {
            sleep(Duration::from_millis(ms));
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
