//! `perfy serve <perf.data>` — parse the file once at startup, then expose
//! the tracks/flamegraph/annotate API used by the React frontend.

mod annotate;
mod api;
mod flamegraph;
mod parser;
mod pipeline;
mod profile;
mod stat;
mod tracks;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Context;
use arc_swap::ArcSwap;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "perfy", about = "perf.data analysis backend")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Parse the perf.data + perf.stat.data pair inside DIR and serve
    /// the perfy HTTP API. Both files are required; the server hot-
    /// reloads whenever **both** files have been rewritten (waiting for
    /// the second one keeps a partially-rerecorded directory from
    /// flashing inconsistent state into the UI).
    Serve {
        /// Directory containing both `perf.data` and `perf.stat.data`.
        dir: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 5005)]
        port: u16,
        /// Baseline (idle) L3-read-miss latency in core clocks. The
        /// memory-latency track plots `latency − base` so peaks above
        /// the floor are visible. Hover tooltips show both delta and
        /// absolute. Defaults to 0 (graph = absolute, no shift).
        #[arg(long, default_value_t = 0.0)]
        memory_base_latency: f64,
    },
    /// Parse PERF_DATA and dump summary metadata to stdout.
    DumpMeta {
        perf_data: String,
    },
    /// Print symbol-resolution diagnostics: every binary the perf.data
    /// references, its mmap range, whether wholesym loaded it, and the
    /// resolve result for a handful of representative IPs.
    Diagnose {
        perf_data: String,
        /// How many distinct (binary, ip) traces to print.
        #[arg(long, default_value_t = 30)]
        sample_traces: usize,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve {
            dir,
            host,
            port,
            memory_base_latency,
        } => serve(&dir, &host, port, memory_base_latency),
        Cmd::DumpMeta { perf_data } => dump_meta(&perf_data),
        Cmd::Diagnose { perf_data, sample_traces } => diagnose(&perf_data, sample_traces),
    }
}

/// Names of the two files we always expect inside the recording
/// directory. The watcher tracks both and only triggers a reload once
/// **both** mtimes have advanced past the previously loaded pair.
const PERF_DATA: &str = "perf.data";
const PERF_STAT_DATA: &str = "perf.stat.data";

/// `mtime` of a file as a `SystemTime`, or `None` if the file is
/// missing or `stat()` failed. Two `None`s never compare equal in our
/// watcher logic — we treat a missing file as "no fresh data yet".
fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn load_profile(
    perf_data: &Path,
    memory_base_latency: f64,
) -> anyhow::Result<Arc<profile::Profile>> {
    let t0 = std::time::Instant::now();
    let mut profile = parser::parse_perf_data(
        perf_data.to_str().context("perf.data path is not utf-8")?,
    )?;
    profile.memory_base_latency = memory_base_latency;
    if memory_base_latency > 0.0 {
        eprintln!("Memory latency baseline: {memory_base_latency:.0} clk");
    }
    eprintln!(
        "Loaded {} samples across {} CPUs ({}) in {:.2}s.",
        profile.samples.len(),
        profile.cpus.len(),
        profile
            .categories
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        t0.elapsed().as_secs_f64(),
    );
    Ok(Arc::new(profile))
}

fn serve(
    dir: &str,
    host: &str,
    port: u16,
    memory_base_latency: f64,
) -> anyhow::Result<()> {
    let dir = PathBuf::from(dir);
    if !dir.is_dir() {
        anyhow::bail!("{} is not a directory", dir.display());
    }
    let perf_data = dir.join(PERF_DATA);
    let stat_data = dir.join(PERF_STAT_DATA);
    if !perf_data.is_file() {
        anyhow::bail!(
            "missing {} in {}: a perf-record recording is required",
            PERF_DATA,
            dir.display()
        );
    }
    if !stat_data.is_file() {
        anyhow::bail!(
            "missing {} in {}: a perf-stat-record recording is required \
             (run `perf stat record -o perf.stat.data -M PipelineL2 -M l3_read_miss_latency …`)",
            PERF_STAT_DATA,
            dir.display()
        );
    }
    eprintln!("Loading {}…", perf_data.display());

    // Capture mtimes BEFORE parsing so an in-flight rewrite between
    // load and watcher start doesn't cause an immediate spurious
    // reload. We snapshot first, then parse; even if both files are
    // rewritten during the parse we'll detect that as a "fresh" pair
    // on the next watcher tick and reload, which is correct.
    let initial_mtimes = (file_mtime(&perf_data), file_mtime(&stat_data));
    let initial = load_profile(&perf_data, memory_base_latency)?;
    let state: api::AppState = Arc::new(ArcSwap::from(initial));

    spawn_reload_watcher(
        dir.clone(),
        perf_data.clone(),
        stat_data.clone(),
        memory_base_latency,
        initial_mtimes,
        Arc::clone(&state),
    );

    let app = api::router(Arc::clone(&state));
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("parsing host:port {host}:{port}"))?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("starting tokio runtime")?;
    runtime.block_on(async move {
        eprintln!("Serving on http://{addr}");
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

/// Background thread that polls `perf.data` + `perf.stat.data` mtimes
/// and atomically swaps in a freshly parsed profile once **both** have
/// advanced past the last loaded pair.
///
/// "Wait for both" is intentional: when the user re-runs the recording,
/// `perf.stat.data` lands a moment before `perf.data` finishes flushing
/// (or vice versa), and reloading on the first change shows the new
/// timeline against the old summary (or vice versa). Holding off until
/// both files have updated keeps the UI consistent.
fn spawn_reload_watcher(
    dir: PathBuf,
    perf_data: PathBuf,
    stat_data: PathBuf,
    memory_base_latency: f64,
    initial: (Option<SystemTime>, Option<SystemTime>),
    state: api::AppState,
) {
    std::thread::Builder::new()
        .name("perfy-reload-watcher".into())
        .spawn(move || {
            let mut loaded = initial;
            let poll = Duration::from_millis(750);
            loop {
                std::thread::sleep(poll);
                let now_perf = file_mtime(&perf_data);
                let now_stat = file_mtime(&stat_data);
                // Both files must exist AND both must have advanced
                // strictly past the previously loaded mtimes. `>` (not
                // `!=`) avoids a degenerate filesystem that rolls mtime
                // back briefly from looking like a fresh recording.
                let both_advanced = match (loaded.0, loaded.1, now_perf, now_stat) {
                    (Some(p0), Some(s0), Some(p1), Some(s1)) => p1 > p0 && s1 > s0,
                    // No prior load on one side (rare — happens if a
                    // file briefly disappeared on disk). Treat as
                    // "advanced" so we retry the load.
                    (_, _, Some(_), Some(_)) => true,
                    _ => false,
                };
                if !both_advanced {
                    continue;
                }
                eprintln!(
                    "[hot reload] both files in {} have changed — reparsing",
                    dir.display()
                );
                match load_profile(&perf_data, memory_base_latency) {
                    Ok(new_profile) => {
                        let old = state.swap(new_profile);
                        // wholesym's SymbolMap drop wants a tokio
                        // runtime; tearing it down from an arbitrary
                        // thread panics. Leaking the previous profile
                        // is fine — these are dev-time reloads and the
                        // old data set is measured in hundreds of MB at
                        // worst.
                        std::mem::forget(old);
                        loaded = (now_perf, now_stat);
                        eprintln!("[hot reload] swapped in new profile");
                    }
                    Err(e) => {
                        // Don't update `loaded`: leave it pointing at
                        // the previous successful pair so the next
                        // poll will retry as long as both files keep
                        // an advancing mtime. Common cause: caught the
                        // files mid-write and the parser bailed.
                        eprintln!("[hot reload] parse failed: {e:#}");
                    }
                }
            }
        })
        .expect("spawning reload watcher thread");
}

fn diagnose(perf_data: &str, sample_traces: usize) -> anyhow::Result<()> {
    use ahash::AHashMap;
    use ibs_annotate::reader::read_perf_data;
    use std::path::Path;

    eprintln!("Loading {perf_data}…");
    let raw = read_perf_data(Path::new(perf_data))?;

    println!("# perf.data: {perf_data}");
    println!(
        "samples={}, events={}, time_range_ns=[{}..{}]",
        raw.samples.len(),
        raw.events.len(),
        raw.time_start_ns,
        raw.time_end_ns,
    );
    println!();
    println!("# events");
    for (i, e) in raw.events.iter().enumerate() {
        println!("  [{i}] {} ({:?})", e.name, e.class);
    }
    println!();

    println!("# mmap inventory");
    let mut by_binary: AHashMap<&str, Vec<&ibs_annotate::Mapping>> = AHashMap::new();
    for m in raw.address_spaces.all_mappings_iter() {
        by_binary.entry(m.binary.as_str()).or_default().push(m);
    }
    let mut keys: Vec<&str> = by_binary.keys().copied().collect();
    keys.sort();
    for binary in keys {
        let maps = &by_binary[binary];
        println!("  {binary}  ({} mmaps)", maps.len());
        for m in maps.iter().take(5) {
            println!(
                "      [0x{:x}..0x{:x})  page_offset=0x{:x}  size=0x{:x}",
                m.addr_lo,
                m.addr_hi,
                m.page_offset,
                m.addr_hi - m.addr_lo,
            );
        }
        if maps.len() > 5 {
            println!("      … {} more", maps.len() - 5);
        }
    }
    println!();

    println!("# {sample_traces} representative resolves");
    // Pick distinct (binary, ip) pairs from the sample stream so we see a
    // mix of binaries.
    let mut seen: ahash::AHashSet<(String, u64)> = ahash::AHashSet::new();
    let mut printed = 0usize;
    for s in &raw.samples {
        if printed >= sample_traces {
            break;
        }
        let leaf_ip = s.ip;
        if let Some(m) = raw.address_spaces.lookup(s.pid, leaf_ip) {
            let key = (m.binary.clone(), leaf_ip);
            if !seen.insert(key) {
                continue;
            }
            let base = raw
                .address_spaces
                .binary_base(s.pid, &m.binary)
                .unwrap_or(m.addr_lo.saturating_sub(m.page_offset));
            let trace = raw.symbol_cache.resolve_verbose(m, leaf_ip, base);
            println!(
                "  ip=0x{:x}  pid={}  binary={}  page_offset=0x{:x}  file_off=0x{:x}  loaded={}  syms={}  sym={:?}  sym_addr=0x{:x}  sym_size={:?}  off_in_sym={}",
                trace.ip,
                s.pid,
                trace.binary.split('/').next_back().unwrap_or(&trace.binary),
                trace.mapping_page_offset,
                trace.file_off,
                trace.map_loaded,
                trace.map_symbol_count,
                trace.sym,
                trace.sym_address,
                trace.sym_size,
                trace.offset_within_sym,
            );
            printed += 1;
        } else {
            // Track unmapped IPs separately
        }
    }
    println!();

    // Per-binary leaf-IP resolution stats across ALL samples.
    println!("# leaf-ip resolution by binary");
    let mut by_bin_hits: AHashMap<String, u64> = AHashMap::new();
    let mut by_bin_miss: AHashMap<String, u64> = AHashMap::new();
    let mut unmapped_user_ips: u64 = 0;
    let mut unmapped_kernel_ips: u64 = 0;
    for s in &raw.samples {
        match raw.address_spaces.lookup(s.pid, s.ip) {
            Some(m) => {
                let base = raw
                    .address_spaces
                    .binary_base(s.pid, &m.binary)
                    .unwrap_or(m.addr_lo.saturating_sub(m.page_offset));
                if raw.symbol_cache.resolve_with_base(m, s.ip, base).is_some() {
                    *by_bin_hits.entry(m.binary.clone()).or_default() += 1;
                } else {
                    *by_bin_miss.entry(m.binary.clone()).or_default() += 1;
                }
            }
            None => {
                if s.ip > 0xffff_0000_0000_0000 {
                    unmapped_kernel_ips += 1;
                } else {
                    unmapped_user_ips += 1;
                }
            }
        }
    }
    let mut bins: Vec<&String> = by_bin_hits.keys().chain(by_bin_miss.keys()).collect();
    bins.sort();
    bins.dedup();
    for b in bins {
        let h = by_bin_hits.get(b).copied().unwrap_or(0);
        let m = by_bin_miss.get(b).copied().unwrap_or(0);
        println!("  hits={h:>10}  miss={m:>10}  {b}");
    }
    println!(
        "  unmapped: kernel_ips={unmapped_kernel_ips}  user_ips={unmapped_user_ips}",
    );
    println!();

    // For each known binary, the top resolved symbols (gives us a quick
    // sanity check that the right functions are showing up).
    println!("# top resolved leaf symbols per binary");
    let mut by_bin_top: AHashMap<String, AHashMap<String, u64>> = AHashMap::new();
    for s in &raw.samples {
        if let Some(m) = raw.address_spaces.lookup(s.pid, s.ip) {
            let base = raw
                .address_spaces
                .binary_base(s.pid, &m.binary)
                .unwrap_or(m.addr_lo.saturating_sub(m.page_offset));
            if let Some((sym, _)) = raw.symbol_cache.resolve_with_base(m, s.ip, base) {
                *by_bin_top
                    .entry(m.binary.clone())
                    .or_default()
                    .entry(sym)
                    .or_default() += 1;
            }
        }
    }
    let mut binaries: Vec<&String> = by_bin_top.keys().collect();
    binaries.sort();
    for b in binaries {
        let mut entries: Vec<(String, u64)> = by_bin_top[b].clone().into_iter().collect();
        entries.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        println!("  {b}");
        for (sym, c) in entries.into_iter().take(10) {
            println!("    {c:>8}  {sym}");
        }
    }
    println!();

    let stats = raw.symbol_cache.stats();
    println!("# resolve stats");
    println!(
        "  binaries_loaded={}  binaries_failed={}  hits={}  misses={}",
        stats.binaries_loaded, stats.binaries_failed, stats.lookup_hits, stats.lookup_misses,
    );

    Ok(())
}

fn dump_meta(perf_data: &str) -> anyhow::Result<()> {
    let profile = parser::parse_perf_data(perf_data)?;
    let summary = serde_json::json!({
        "perf_data": profile.perf_data_path,
        "binary": profile.binary_path,
        "cpus": profile.cpus,
        "categories": profile.categories.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
        "time_start_ns": profile.time_start_ns,
        "time_end_ns": profile.time_end_ns,
        "duration_ns": profile.duration_ns(),
        "samples": profile.samples.len(),
        "frames": profile.frames.len(),
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
