//! `pivot-bench` — performance harness for pivotdb.
//!
//! Spins up the full pivotdb stack (`dispatch` workers + pgwire `Server` +
//! `DeltaDatastore`) in-process on a free local port, connects with
//! `tokio-postgres`, and runs a benchmark suite end-to-end through the wire
//! protocol so the numbers reflect what a real client would see.
//!
//! # Suites
//!
//! A suite is just a directory under `benchmarks/<name>/` containing:
//! - `setup.sql`: the schema + table-binding statements (`{source}` is
//!   substituted with `--source`).
//! - `qXX.sql`: one file per query. The stem (`q07`) is the query ID.
//! - `qXX.tsv`: the expected TSV output for the matching `qXX.sql`. Compared
//!   for accuracy on every run; populated/refreshed with `--update-results`.
//!
//! Today the only suite is `clickbench`; future suites (e.g. `tpch`) just need
//! to be a sibling directory — the harness discovers queries by listing
//! `qXX.sql` files, so adding a query is "drop in two files".
//!
//! # Baselines
//!
//! Run timings are written to a baseline JSON. Future runs load the prior
//! baseline (from disk or a `gs://` URL) and print a diff highlighting
//! regressions. Use `--save-if-better` to keep the new file only when it wins,
//! or `--force-save` to overwrite unconditionally.

use std::path::PathBuf;

use clap::Parser;

mod baseline;
mod runner;
mod server_handle;

use baseline::{Comparison, SaveVerdict};
use runner::RunOptions;

/// Default regression threshold (%). Runs slower than the baseline by more
/// than this are flagged as REGRESSIONs. Picked to be outside our typical
/// run-to-run noise on warm caches.
const DEFAULT_REGRESSION_PCT: f64 = 5.0;

#[derive(Parser, Debug)]
#[command(name = "pivot-bench", about, version)]
struct Cli {
    /// Suite to run. Resolved as a directory under `benchmarks/`.
    #[arg(long, default_value = "clickbench")]
    suite: String,

    /// Override the resolved suite directory. Defaults to
    /// `<crate>/<suite>` (e.g. `benchmarks/clickbench`).
    #[arg(long)]
    suite_dir: Option<PathBuf>,

    /// Source data directory passed to the suite's `setup.sql` as `{source}`.
    /// Required unless `--show` is given (which only reads the baseline).
    #[arg(long, env = "SOURCE_DIRECTORY", required_unless_present = "show")]
    source: Option<PathBuf>,

    /// Path to the pivotdb-server binary to launch and measure. The benchmark
    /// is a client of this process; the binary named here is the one whose
    /// performance every number describes.
    #[arg(long, env = "PIVOT_SERVER_BIN", required_unless_present = "show")]
    server_bin: Option<PathBuf>,

    /// Number of dispatch worker threads. Defaults to the number of cores.
    #[arg(long, env = "WORKER_COUNT")]
    workers: Option<usize>,

    /// Comma-separated query IDs to run (e.g. `q07,q20`). Bare numbers like
    /// `7,20` are accepted and canonicalised to `qNN`. Defaults to all
    /// queries the suite ships.
    #[arg(long, env = "QUERY", value_delimiter = ',')]
    query: Vec<String>,

    /// Iterations per query.
    #[arg(long, default_value_t = 1, env = "QUERY_TEST_COUNT")]
    iterations: u32,

    /// Milliseconds to sleep between iterations. Useful when chasing thermal /
    /// allocator effects on long suites.
    #[arg(long, env = "SLEEP")]
    sleep: Option<u64>,

    /// Where to read/write the baseline timings. Local path, `gs://bucket/key`,
    /// or `https://...`. Defaults to `<suite_dir>/baseline.json`.
    #[arg(long)]
    baseline: Option<String>,

    /// Save the new run as the baseline only if the suite total improved.
    #[arg(long, conflicts_with = "force_save")]
    save_if_better: bool,

    /// Save the new run as the baseline regardless of comparison outcome.
    #[arg(long)]
    force_save: bool,

    /// Regression threshold (percent). Iterations slower than the baseline by
    /// more than this are flagged as REGRESSIONs.
    #[arg(long, default_value_t = DEFAULT_REGRESSION_PCT)]
    regression_pct: f64,

    /// Refresh the on-disk expected `.tsv` files from this run's output
    /// instead of comparing against them.
    #[arg(long)]
    update_results: bool,

    /// Skip verifying each query's output against its expected `.tsv`. Timings
    /// are still recorded; only the correctness check is suppressed. Useful
    /// when running against a dataset whose results don't match the committed
    /// expectations.
    #[arg(long, conflicts_with = "update_results")]
    skip_check: bool,

    /// SQL statement to run once, untimed, after setup and before the first
    /// query. Warms the server (worker spin-up, page faults, plan paths) so
    /// the first timed query isn't charged for one-time session costs.
    #[arg(long)]
    warmup: Option<String>,

    /// Before each query, evict pivot's file cache (`SELECT drop_cache()`) and
    /// flush the OS page cache, so each query's first iteration is a true cold
    /// read without restarting the warm server. Needs passwordless sudo (Linux).
    #[arg(long)]
    drop_caches: bool,

    /// Print the recorded results (current cold/hot per suite & query, plus
    /// per-query run count and last-saved timestamp) from the baseline and
    /// exit — does not boot the server or run anything.
    #[arg(long)]
    show: bool,
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Normalise a user-supplied query selector to its canonical ID. `q07`, `Q7`,
/// and `7` all map to `q07`. Non-numeric input is passed through so the
/// runner reports an unknown selector by name rather than silently dropping it.
fn canonicalise_query(input: &str) -> String {
    let trimmed = input.trim().trim_start_matches(['q', 'Q']);
    if let Ok(n) = trimmed.parse::<u32>() {
        format!("q{n:02}")
    } else {
        input.trim().to_string()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let cli = Cli::parse();

    let suite_dir = cli
        .suite_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&cli.suite));
    let baseline_location = cli
        .baseline
        .clone()
        .unwrap_or_else(|| suite_dir.join("baseline.json").display().to_string());

    // `--show` is a read-only inspection of the baseline: print and exit
    // without discovering the suite, booting the server, or running anything.
    if cli.show {
        match baseline::load(&baseline_location)? {
            Some(b) => b.print_summary(&baseline_location),
            None => println!("no baseline found at {baseline_location}"),
        }
        return Ok(());
    }

    let source = cli
        .source
        .clone()
        .expect("clap enforces --source unless --show");
    let suite = runner::discover_suite(&cli.suite, &suite_dir)?;

    let server_bin = cli
        .server_bin
        .clone()
        .expect("clap enforces --server-bin unless --show");
    // No worker count here means none in the generated config, and the server
    // applies its own default (the machine's core count).
    let server = server_handle::start(&server_bin, cli.workers, &source)?;

    let query_filter = if cli.query.is_empty() {
        None
    } else {
        Some(cli.query.iter().map(|s| canonicalise_query(s)).collect())
    };

    let opts = RunOptions {
        source,
        // Zero iterations would leave a query with no cold sample; clamp so a
        // stray `--iterations 0` (or `QUERY_TEST_COUNT=0`) still does one run.
        iterations: cli.iterations.max(1),
        sleep_ms: cli.sleep,
        update_results: cli.update_results,
        skip_check: cli.skip_check,
        query_filter,
        drop_caches: cli.drop_caches,
        warmup: cli.warmup.clone(),
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let suite_run = rt.block_on(runner::run_suite(&server, &suite, &opts))?;

    if cli.update_results {
        println!(
            "\nupdated expected output for {} queries — no comparison performed",
            cli.suite
        );
        drop(server);
        return Ok(());
    }

    let prior = baseline::load(&baseline_location)?;
    let comparison = Comparison::build(prior.as_ref(), &suite_run, cli.regression_pct);
    comparison.render();

    let should_save = if cli.force_save {
        true
    } else if cli.save_if_better {
        match comparison.save_verdict() {
            SaveVerdict::Improved {
                cold_pct,
                hot_pct: Some(hot_pct),
            } => {
                println!("\ncold {cold_pct:+.1}%, hot {hot_pct:+.1}% — saving baseline");
                true
            }
            SaveVerdict::Improved {
                cold_pct,
                hot_pct: None,
            } => {
                println!("\ncold {cold_pct:+.1}% (no comparable hot timing) — saving baseline");
                true
            }
            SaveVerdict::NoBaseline => {
                println!("\nno overlapping queries in prior baseline — saving");
                true
            }
            SaveVerdict::NotImproved(reason) => {
                println!("\nnot saving: {reason} (use --force-save to override)");
                false
            }
        }
    } else {
        false
    };

    if should_save {
        let mut baseline = prior.unwrap_or_default();
        let timestamp = chrono::Utc::now().to_rfc3339();
        baseline.record_run(&suite_run, &timestamp);
        baseline::save(&baseline_location, &baseline)?;
        println!("baseline updated → {baseline_location}");
    }

    drop(server);
    Ok(())
}
