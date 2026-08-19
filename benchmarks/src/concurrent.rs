//! Concurrency benchmark: N clients run the whole suite at the same time.
//!
//! One invocation measures one client count: after setup (and, only with
//! `--warmup-sweep`, one untimed warm-up pass), N clients start together
//! behind a barrier and each runs every query. With no warmup the concurrent
//! sweep is the first work the fresh server does, so separate invocations at
//! different client counts compare cold to cold; scaling across counts is
//! read across those invocations.
//!
//! Each client either runs the queries in the suite's canonical order or in
//! its own deterministic permutation (seeded by client index, so orderings
//! differ between clients but are identical between runs).
//!
//! Timings only: query output is drained (latency includes the full result
//! transfer) but not compared against expected results.
//!
//! A query the server aborts for lack of evictable memory is recorded as a
//! failed execution (reported per query and in the totals) and the client
//! moves on, so a run under overload still yields its scaling numbers plus an
//! error rate. Latency stats cover completed executions only.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Barrier;
use tokio::task::JoinSet;

use crate::runner::{self, Result, RunOptions, Suite};
use crate::server_handle::ServerHandle;

/// How each concurrent client orders the suite's queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OrderMode {
    /// Every client runs the queries in the suite's canonical order.
    Same,
    /// Each client runs its own permutation, seeded by client index: stable
    /// across runs, different across clients.
    Shuffled,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub clients: usize,
    pub order: OrderMode,
    /// Timed sweeps per client in each measured phase. Latencies are pooled
    /// across sweeps; wall times cover all of them.
    pub sweeps: u32,
    /// Run one untimed sweep before the measured phases, so both measure a
    /// warm server. Off by default: the server does direct I/O, so restarting
    /// it per run compares cold to cold.
    pub warmup_sweep: bool,
    /// Write the full per-client, per-query latencies as JSON here.
    pub json_out: Option<PathBuf>,
}

/// A query with its SQL text loaded up front, so the timed loop only sends.
struct PreparedQuery {
    id: String,
    sql: String,
}

/// One query execution by one client.
struct Execution {
    query_id: String,
    /// Time to the result for a completed execution; time to the server's
    /// abort for a failed one.
    latency_ms: u128,
    /// False when the server aborted the query for lack of evictable memory.
    completed: bool,
}

/// One client's timed pass over the suite, executions in the order they ran.
struct ClientRun {
    executions: Vec<Execution>,
    /// Wall time of the client's whole pass (all sweeps), in ms.
    wall_ms: u128,
}

fn is_memory_exhausted(error: &runner::Error) -> bool {
    if let runner::Error::Postgres(pg_error) = error
        && let Some(db_error) = pg_error.as_db_error()
    {
        return db_error.message().contains("no evictable memory");
    }
    false
}

fn xorshift_next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// The order (as indices into the query list) client `client_index` runs.
/// `Shuffled` is a Fisher-Yates permutation from a PRNG seeded only by the
/// client index, so it is reproducible run to run.
fn order_for_client(query_count: usize, client_index: usize, mode: OrderMode) -> Vec<usize> {
    let mut order: Vec<usize> = (0..query_count).collect();
    if mode == OrderMode::Same {
        return order;
    }
    let mut state = (client_index as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
    for i in (1..query_count).rev() {
        let j = (xorshift_next(&mut state) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// Run one query once. A server-side memory abort is recorded as a failed
/// execution; anything else is a hard error.
async fn run_one_query(
    client: &tokio_postgres::Client,
    query: &PreparedQuery,
    label: &str,
) -> Result<Execution> {
    let start = Instant::now();
    match runner::collect_tsv(client, &query.sql, &query.id).await {
        Ok(_) => Ok(Execution {
            query_id: query.id.clone(),
            latency_ms: start.elapsed().as_millis(),
            completed: true,
        }),
        Err(error) if is_memory_exhausted(&error) => {
            println!(
                "{label} {}: aborted by the server (no evictable memory)",
                query.id
            );
            Ok(Execution {
                query_id: query.id.clone(),
                latency_ms: start.elapsed().as_millis(),
                completed: false,
            })
        }
        Err(error) => Err(error),
    }
}

/// Run one client's pass: `sweeps` full passes over the queries in `order`,
/// draining every result row so latency covers the wire cost end to end.
async fn run_client_pass(
    client: &tokio_postgres::Client,
    queries: &[PreparedQuery],
    order: &[usize],
    sweeps: u32,
    label: &str,
) -> Result<ClientRun> {
    let mut executions = Vec::with_capacity(order.len() * sweeps as usize);
    let pass_start = Instant::now();
    for _ in 0..sweeps {
        for &idx in order {
            executions.push(run_one_query(client, &queries[idx], label).await?);
        }
    }
    let wall_ms = pass_start.elapsed().as_millis();
    println!("{label}: pass done in {:.1}s", wall_ms as f64 / 1000.0);
    Ok(ClientRun {
        executions,
        wall_ms,
    })
}

/// Every execution of one query across the given runs.
fn executions_for_query<'a>(runs: &'a [ClientRun], query_id: &str) -> Vec<&'a Execution> {
    runs.iter()
        .flat_map(|run| &run.executions)
        .filter(|execution| execution.query_id == query_id)
        .collect()
}

fn mean(samples: &[u128]) -> f64 {
    samples.iter().sum::<u128>() as f64 / samples.len() as f64
}

fn render_report(
    queries: &[PreparedQuery],
    concurrent_runs: &[ClientRun],
    concurrent_wall_ms: u128,
    opts: &Options,
) {
    let clients = concurrent_runs.len();
    println!(
        "\n=== Concurrency report: {clients} clients, {:?} order ===",
        opts.order
    );
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>6}",
        "query", "min", "mean", "max", "failed"
    );
    for query in queries {
        let executions = executions_for_query(concurrent_runs, &query.id);
        let latencies: Vec<u128> = executions
            .iter()
            .filter(|e| e.completed)
            .map(|e| e.latency_ms)
            .collect();
        let failed = executions.iter().filter(|e| !e.completed).count();
        if latencies.is_empty() {
            println!(
                "{:<8} {:>8} {:>8} {:>8} {:>6}",
                query.id, "-", "-", "-", failed
            );
            continue;
        }
        println!(
            "{:<8} {:>8} {:>8.1} {:>8} {:>6}",
            query.id,
            latencies.iter().min().expect("latencies is non-empty"),
            mean(&latencies),
            latencies.iter().max().expect("latencies is non-empty"),
            failed,
        );
    }

    let wall_s = concurrent_wall_ms as f64 / 1000.0;
    let client_walls: Vec<u128> = concurrent_runs.iter().map(|r| r.wall_ms).collect();
    println!(
        "\nwall ({clients} clients): {wall_s:.1}s (per-client min {:.1}s, max {:.1}s)",
        *client_walls.iter().min().expect("at least one client") as f64 / 1000.0,
        *client_walls.iter().max().expect("at least one client") as f64 / 1000.0,
    );
    let all_executions: Vec<&Execution> = concurrent_runs
        .iter()
        .flat_map(|run| &run.executions)
        .collect();
    let total_failed = all_executions.iter().filter(|e| !e.completed).count();
    if total_failed > 0 {
        println!(
            "under concurrent load: {total_failed} of {} executions were aborted \
             by the server for lack of evictable memory",
            all_executions.len()
        );
    }
}

fn write_json(
    path: &PathBuf,
    suite: &Suite,
    concurrent_runs: &[ClientRun],
    concurrent_wall_ms: u128,
    opts: &Options,
) -> Result<()> {
    let client_json = |run: &ClientRun| {
        serde_json::json!({
            "executions": run.executions.iter().map(|e| serde_json::json!({
                "query": e.query_id,
                "latency_ms": e.latency_ms as u64,
                "completed": e.completed,
            })).collect::<Vec<_>>(),
            "wall_ms": run.wall_ms as u64,
        })
    };
    let report = serde_json::json!({
        "suite": suite.name,
        "clients": opts.clients,
        "order_mode": format!("{:?}", opts.order),
        "sweeps": opts.sweeps,
        "concurrent_wall_ms": concurrent_wall_ms as u64,
        "concurrent": concurrent_runs.iter().map(client_json).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&report).expect("report serialises");
    std::fs::write(path, text).map_err(|source| runner::Error::Io {
        path: path.clone(),
        source,
    })?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Run the whole concurrency benchmark against `server`.
pub async fn run(
    server: &ServerHandle,
    suite: &Suite,
    run_opts: &RunOptions,
    opts: &Options,
) -> Result<()> {
    let setup_client = runner::connect(server.port()).await?;
    let setup_template = runner::read_to_string(&suite.setup_sql_path)?;
    let setup_sql = setup_template.replace("{source}", &run_opts.source.display().to_string());
    for statement in runner::setup_statements(&setup_sql) {
        setup_client.simple_query(&statement).await?;
    }

    let mut queries = Vec::new();
    for query in &suite.queries {
        if let Some(filter) = &run_opts.query_filter
            && !filter.contains(&query.id)
        {
            continue;
        }
        queries.push(PreparedQuery {
            id: query.id.clone(),
            sql: runner::read_to_string(&query.sql_path)?,
        });
    }
    let queries = Arc::new(queries);
    let canonical: Vec<usize> = (0..queries.len()).collect();

    if opts.warmup_sweep {
        println!("=== Warmup sweep ({} queries) ===", queries.len());
        let warmup = run_client_pass(&setup_client, &queries, &canonical, 1, "warmup").await?;
        println!("warmup done in {:.1}s", warmup.wall_ms as f64 / 1000.0);
    }

    drop(setup_client);

    println!("\n=== Concurrent sweep: {} clients ===", opts.clients);
    let mut connections = Vec::with_capacity(opts.clients);
    for _ in 0..opts.clients {
        connections.push(runner::connect(server.port()).await?);
    }
    let start_line = Arc::new(Barrier::new(opts.clients + 1));
    let mut tasks: JoinSet<Result<ClientRun>> = JoinSet::new();
    for (client_index, connection) in connections.into_iter().enumerate() {
        let order = order_for_client(queries.len(), client_index, opts.order);
        let queries = queries.clone();
        let start_line = start_line.clone();
        let sweeps = opts.sweeps;
        tasks.spawn(async move {
            start_line.wait().await;
            let label = format!("client {client_index}");
            run_client_pass(&connection, &queries, &order, sweeps, &label).await
        });
    }
    start_line.wait().await;
    let wall_start = Instant::now();
    let mut concurrent_runs = Vec::with_capacity(opts.clients);
    while let Some(finished) = tasks.join_next().await {
        concurrent_runs.push(finished.expect("a client task is not cancelled or panicked")?);
    }
    let concurrent_wall_ms = wall_start.elapsed().as_millis();
    println!(
        "concurrent sweep done in {:.1}s",
        concurrent_wall_ms as f64 / 1000.0
    );

    render_report(&queries, &concurrent_runs, concurrent_wall_ms, opts);
    if let Some(path) = &opts.json_out {
        write_json(path, suite, &concurrent_runs, concurrent_wall_ms, opts)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{OrderMode, order_for_client};

    #[test]
    fn same_mode_keeps_canonical_order() {
        let order = order_for_client(5, 3, OrderMode::Same);

        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn shuffled_orders_are_permutations() {
        for client in 0..8 {
            let mut order = order_for_client(43, client, OrderMode::Shuffled);

            order.sort_unstable();
            assert_eq!(order, (0..43).collect::<Vec<_>>());
        }
    }

    #[test]
    fn shuffled_orders_are_stable_and_differ_between_clients() {
        let first = order_for_client(43, 0, OrderMode::Shuffled);
        let again = order_for_client(43, 0, OrderMode::Shuffled);
        let other = order_for_client(43, 1, OrderMode::Shuffled);

        assert_eq!(first, again);
        assert_ne!(first, other);
    }
}
