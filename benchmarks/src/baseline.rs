//! Load, save, diff, and accumulate baseline timings.
//!
//! A baseline is one JSON document with two parts:
//!
//! - `current[suite][query]` — the latest **cold** and **hot** timing for each
//!   query. This is what the comparison table is drawn against and what the
//!   `--save-if-better` gate inspects. Keying by *suite first* means two
//!   suites that happen to share a query stem (`q01`) can never overwrite each
//!   other's numbers.
//! - `history` — an append-only log of every run ever saved, one row per
//!   `(timestamp, suite, query)`. Each row is self-describing (carries its
//!   suite), so slicing the log by suite or query after the fact is trivial.
//!
//! **Cold** is the first iteration of a query — what a fresh process pays
//! before page cache, planner thread-locals, and allocator arenas warm up.
//! **Hot** is the arithmetic mean of every iteration *after* the first —
//! steady state. A run that did a single iteration has no hot number.
//!
//! `--save-if-better` only overwrites the baseline when *both* the cold suite
//! total and the hot suite total improved (a faster cold start that regressed
//! steady-state, or vice versa, is not an improvement).
//!
//! Reading a baseline supports local paths, `https://` URLs, and
//! `gs://bucket/object` (resolved against the public GCS HTTP endpoint).
//! Writing supports local paths and `gs://` (the latter shells out to
//! `gsutil` / `gcloud storage`, so the caller must be authenticated).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::runner::SuiteRun;

/// Minimal ANSI colouring for the result tables. No dependency; a no-op unless
/// stdout is a terminal and `NO_COLOR` is unset (per <https://no-color.org/>).
mod color {
    use std::io::IsTerminal;
    use std::sync::OnceLock;

    fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
        })
    }

    fn wrap(code: &str, s: &str) -> String {
        if enabled() {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn green(s: &str) -> String {
        wrap("32", s)
    }
    pub fn red(s: &str) -> String {
        wrap("31", s)
    }

    /// Colour `s` green when `current < baseline` (faster — an improvement),
    /// red when slower, and leave it plain when equal (or either value is
    /// non-finite). `s` should already be padded to its column width: the
    /// escape codes wrap the *outside*, so alignment is unaffected.
    pub fn by_delta(s: &str, current: f64, baseline: f64) -> String {
        if !current.is_finite() || !baseline.is_finite() || current == baseline {
            s.to_string()
        } else if current < baseline {
            green(s)
        } else {
            red(s)
        }
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("json error on {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("baseline url fetch failed for {url}: {message}")]
    Fetch { url: String, message: String },
    #[error("baseline upload failed for {url}: {message}")]
    Upload { url: String, message: String },
    #[error("unsupported scheme in baseline url: {0}")]
    UnsupportedScheme(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Cold/hot timing for one query.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QueryStats {
    /// First-iteration wall-clock, ms.
    pub cold_ms: u128,
    /// Mean of iterations after the first, ms. Absent for single-iteration runs.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub hot_ms: Option<f64>,
}

/// One saved run of one query — the unit of [`Baseline::history`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// RFC3339 UTC timestamp of when the run was saved.
    pub timestamp: String,
    pub suite: String,
    pub query: String,
    /// Iterations the run did (context for `stats.hot_ms`).
    pub iterations: u32,
    pub stats: QueryStats,
}

/// On-disk baseline document. `BTreeMap`/`Vec` keep a stable order so the file
/// doesn't churn purely from hash iteration order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Baseline {
    /// `current[suite][query] -> latest stats`.
    #[serde(default)]
    pub current: BTreeMap<String, BTreeMap<String, QueryStats>>,
    /// Append-only; newest runs at the end.
    #[serde(default)]
    pub history: Vec<RunRecord>,
}

impl Baseline {
    /// Latest stats for a query within a suite, if any are recorded.
    pub fn current_for(&self, suite: &str, query: &str) -> Option<QueryStats> {
        self.current.get(suite).and_then(|m| m.get(query)).copied()
    }

    /// Print the recorded `current` stats — every suite/query with its latest
    /// cold/hot, plus a per-query run count and last-saved timestamp pulled
    /// from `history`. Used by `pivot-bench --show`.
    pub fn print_summary(&self, location: &str) {
        println!("--- baseline: {location} ---");
        if self.current.is_empty() {
            println!("(no recorded results)");
            return;
        }
        for (suite, queries) in &self.current {
            println!();
            println!("suite: {suite}");
            println!(
                "{:<8}  {:>10}  {:>10}  {:>5}  last_run",
                "query", "cold_ms", "hot_ms", "runs",
            );
            for (query, stats) in queries {
                let mut hist: Vec<&RunRecord> = self
                    .history
                    .iter()
                    .filter(|r| &r.suite == suite && &r.query == query)
                    .collect();
                // RFC3339-UTC timestamps sort lexically == chronologically.
                hist.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
                let runs = hist.len();
                let last = hist.last().map(|r| r.timestamp.as_str()).unwrap_or("—");
                // Colour the latest values against the previous recorded run
                // for this query (if any): green = got faster, red = slower.
                let prev = (hist.len() >= 2).then(|| hist[hist.len() - 2].stats);

                let cold_cell = {
                    let s = format!("{:>10}", stats.cold_ms);
                    match prev {
                        Some(p) => color::by_delta(&s, stats.cold_ms as f64, p.cold_ms as f64),
                        None => s,
                    }
                };
                let hot_cell = {
                    let raw = stats
                        .hot_ms
                        .map(|h| format!("{h:.1}"))
                        .unwrap_or_else(|| "—".to_string());
                    let s = format!("{:>10}", raw);
                    match (prev.and_then(|p| p.hot_ms), stats.hot_ms) {
                        (Some(po), Some(cu)) => color::by_delta(&s, cu, po),
                        _ => s,
                    }
                };
                println!("{:<8}  {cold_cell}  {hot_cell}  {runs:>5}  {last}", query);
            }
        }
        if !self.history.is_empty() {
            // Timestamps are RFC3339 in UTC (`+00:00`), so lexical order is
            // chronological order.
            let oldest = self
                .history
                .iter()
                .map(|r| r.timestamp.as_str())
                .min()
                .unwrap();
            let newest = self
                .history
                .iter()
                .map(|r| r.timestamp.as_str())
                .max()
                .unwrap();
            println!();
            println!("history: {} rows, {oldest} … {newest}", self.history.len());
        }
    }

    /// Fold a run in: overwrite `current[suite][*]` with this run's timings
    /// and append one `history` row per query.
    pub fn record_run(&mut self, run: &SuiteRun, timestamp: &str) {
        let suite_map = self.current.entry(run.suite.clone()).or_default();
        for q in &run.queries {
            suite_map.insert(
                q.id.clone(),
                QueryStats {
                    cold_ms: q.cold_ms(),
                    hot_ms: q.hot_ms(),
                },
            );
        }
        for q in &run.queries {
            self.history.push(RunRecord {
                timestamp: timestamp.to_string(),
                suite: run.suite.clone(),
                query: q.id.clone(),
                iterations: q.iterations_ms.len() as u32,
                stats: QueryStats {
                    cold_ms: q.cold_ms(),
                    hot_ms: q.hot_ms(),
                },
            });
        }
    }
}

// ── Locations & IO ─────────────────────────────────────────────────────────

/// Where a baseline reference points. We classify on the prefix because each
/// kind needs different IO: `gs://` reads route through the public GCS HTTP
/// endpoint and writes shell out to `gsutil`; plain HTTPS is read-only;
/// everything else is a local path.
pub enum Location<'a> {
    Local(&'a Path),
    Https(&'a str),
    Gs(&'a str),
}

impl<'a> Location<'a> {
    pub fn parse(s: &'a str) -> Self {
        if let Some(stripped) = s.strip_prefix("gs://") {
            Location::Gs(stripped)
        } else if s.starts_with("https://") || s.starts_with("http://") {
            Location::Https(s)
        } else {
            Location::Local(Path::new(s))
        }
    }
}

/// Try to load a baseline. `Ok(None)` means the location does not yet exist —
/// a normal first-run condition that callers handle by starting fresh.
/// `Err` is reserved for genuine failures (network errors, malformed JSON,
/// permission issues).
pub fn load(location_str: &str) -> Result<Option<Baseline>> {
    match Location::parse(location_str) {
        Location::Local(path) => match std::fs::read_to_string(path) {
            Ok(s) => Ok(Some(serde_json::from_str(&s).map_err(|source| {
                Error::Json {
                    path: path.display().to_string(),
                    source,
                }
            })?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(Error::Io {
                path: path.display().to_string(),
                source,
            }),
        },
        Location::Https(url) => fetch_https(url).map(Some),
        Location::Gs(rest) => {
            let url = format!("https://storage.googleapis.com/{rest}");
            fetch_https(&url).map(Some)
        }
    }
}

/// Persist a baseline. Local paths write the file directly (creating parent
/// directories as needed); `gs://` URLs shell out to `gsutil cp` (the caller
/// is expected to be authenticated via `gcloud auth`). HTTPS-only locations
/// are read-only and writing is rejected.
pub fn save(location_str: &str, baseline: &Baseline) -> Result<()> {
    let serialised = serde_json::to_string_pretty(baseline).map_err(|source| Error::Json {
        path: location_str.to_string(),
        source,
    })?;
    match Location::parse(location_str) {
        Location::Local(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                    path: parent.display().to_string(),
                    source,
                })?;
            }
            std::fs::write(path, &serialised).map_err(|source| Error::Io {
                path: path.display().to_string(),
                source,
            })
        }
        Location::Gs(rest) => upload_gs(rest, &serialised),
        Location::Https(_) => Err(Error::UnsupportedScheme(
            "writing to plain http(s) baselines is not supported; use a local path or gs:// URL"
                .into(),
        )),
    }
}

/// Fetch a baseline over HTTP(S) with `ureq` (blocking, follows redirects).
/// A non-2xx status surfaces as an [`Error::Fetch`] carrying ureq's message
/// (which includes the status code), so a missing object in a bucket reads as
/// a clear error rather than a parse failure on an HTML error page.
fn fetch_https(url: &str) -> Result<Baseline> {
    let body = ureq::get(url)
        .call()
        .map_err(|e| Error::Fetch {
            url: url.to_string(),
            message: e.to_string(),
        })?
        .into_string()
        .map_err(|e| Error::Fetch {
            url: url.to_string(),
            message: format!("read body: {e}"),
        })?;
    serde_json::from_str(&body).map_err(|source| Error::Json {
        path: url.to_string(),
        source,
    })
}

/// Upload to `gs://...` by piping the serialised JSON into `gsutil cp - <url>`.
/// Falls through to `gcloud storage cp -` if `gsutil` isn't on `$PATH`.
fn upload_gs(rest: &str, contents: &str) -> Result<()> {
    let url = format!("gs://{rest}");
    let mut child = match Command::new("gsutil")
        .args(["cp", "-", &url])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => Command::new("gcloud")
            .args(["storage", "cp", "-", &url])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::Upload {
                url: url.clone(),
                message: format!("neither gsutil nor gcloud available: {e}"),
            })?,
    };
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("piped stdin");
        stdin
            .write_all(contents.as_bytes())
            .map_err(|e| Error::Upload {
                url: url.clone(),
                message: format!("write stdin: {e}"),
            })?;
    }
    let output = child.wait_with_output().map_err(|e| Error::Upload {
        url: url.clone(),
        message: format!("wait: {e}"),
    })?;
    if !output.status.success() {
        return Err(Error::Upload {
            url,
            message: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}

// ── Comparison ─────────────────────────────────────────────────────────────

/// A baseline→current delta for one metric, as a percentage. Negative ⇒ faster.
#[derive(Debug, Clone, Copy)]
pub struct Delta {
    pub baseline: f64,
    pub current: f64,
    pub pct: f64,
}

impl Delta {
    pub fn compute(baseline: f64, current: f64) -> Self {
        // Floor the divisor at 1ms so sub-millisecond baselines don't blow the
        // percentage up; sub-ms timings aren't meaningful for tracking anyway.
        let safe = baseline.max(1.0);
        Self {
            baseline,
            current,
            pct: (current - safe) / safe * 100.0,
        }
    }
}

pub struct ComparisonRow {
    pub query: String,
    pub kind: ComparisonKind,
}

pub enum ComparisonKind {
    /// Query present in both the baseline and this run.
    Both {
        cold: Delta,
        /// Baseline hot, if the baseline recorded one.
        hot_baseline: Option<f64>,
        /// This run's hot, if it did >1 iteration.
        hot_current: Option<f64>,
    },
    /// In this run, not the baseline.
    New { stats: QueryStats },
    /// In the baseline (same suite), not this run — e.g. filtered out by `--query`.
    Missing { stats: QueryStats },
}

pub struct Comparison {
    pub suite: String,
    pub rows: Vec<ComparisonRow>,
    pub regression_pct: f64,
}

/// Verdict of the `--save-if-better` gate.
pub enum SaveVerdict {
    /// At least one of the cold / hot suite totals improved by more than the
    /// regression threshold. `hot_pct` is `None` when there was no comparable
    /// hot data (run did a single iteration, or the baseline predates hot
    /// tracking) — in which case the win was on cold.
    Improved { cold_pct: f64, hot_pct: Option<f64> },
    /// Nothing in the prior baseline overlapped this run — there's nothing to
    /// be slower than, so we save.
    NoBaseline,
    /// Neither metric cleared the bar; the string explains the numbers.
    NotImproved(String),
}

impl Comparison {
    pub fn build(prior: Option<&Baseline>, run: &SuiteRun, regression_pct: f64) -> Self {
        let mut rows = Vec::new();
        let mut seen = BTreeSet::new();
        for q in &run.queries {
            seen.insert(q.id.clone());
            let cur = QueryStats {
                cold_ms: q.cold_ms(),
                hot_ms: q.hot_ms(),
            };
            let kind = match prior.and_then(|b| b.current_for(&run.suite, &q.id)) {
                Some(base) => ComparisonKind::Both {
                    cold: Delta::compute(base.cold_ms as f64, cur.cold_ms as f64),
                    hot_baseline: base.hot_ms,
                    hot_current: cur.hot_ms,
                },
                None => ComparisonKind::New { stats: cur },
            };
            rows.push(ComparisonRow {
                query: q.id.clone(),
                kind,
            });
        }
        if let Some(suite_map) = prior.and_then(|b| b.current.get(&run.suite)) {
            for (q, stats) in suite_map {
                if !seen.contains(q) {
                    rows.push(ComparisonRow {
                        query: q.clone(),
                        kind: ComparisonKind::Missing { stats: *stats },
                    });
                }
            }
        }
        Self {
            suite: run.suite.clone(),
            rows,
            regression_pct,
        }
    }

    fn hot_delta(hot_baseline: Option<f64>, hot_current: Option<f64>) -> Option<Delta> {
        match (hot_baseline, hot_current) {
            (Some(b), Some(c)) => Some(Delta::compute(b, c)),
            _ => None,
        }
    }

    /// `REGRESSION` if either metric regressed past the threshold; `improved`
    /// if cold improved past it and hot didn't regress (or isn't comparable);
    /// `ok` otherwise.
    fn status_label(&self, cold: &Delta, hot: Option<&Delta>) -> &'static str {
        let t = self.regression_pct;
        if cold.pct >= t || hot.is_some_and(|d| d.pct >= t) {
            return "REGRESSION";
        }
        let cold_improved = cold.pct <= -t;
        let hot_not_worse = hot.is_none_or(|d| d.pct < t);
        if cold_improved && hot_not_worse {
            "improved"
        } else {
            "ok"
        }
    }

    /// Print the comparison as an aligned table. Both metrics get their own
    /// before/after/Δ columns so a cold-vs-hot split is visible at a glance.
    /// The Δ% cells and the `status` column are coloured by direction (green
    /// faster, red slower) when stdout is a terminal; the `status` text still
    /// reads on monochrome terminals.
    pub fn render(&self) {
        fn opt(v: Option<f64>) -> String {
            v.map(|x| format!("{x:.1}"))
                .unwrap_or_else(|| "—".to_string())
        }
        // A Δ% value, right-aligned to the column width and tinted by sign.
        fn delta_cell(pct: f64) -> String {
            let padded = format!("{:>9}", format!("{pct:+.1}%"));
            color::by_delta(&padded, pct, 0.0)
        }
        fn dash(width: usize) -> String {
            format!("{:>width$}", "—")
        }
        fn status_cell(label: &str) -> String {
            match label {
                "REGRESSION" => color::red(label),
                "improved" => color::green(label),
                _ => label.to_string(),
            }
        }

        println!();
        println!("--- {} comparison vs baseline ---", self.suite);
        println!(
            "{:<8}  {:>10}  {:>10}  {:>9}  {:>10}  {:>10}  {:>9}  status",
            "query", "cold_base", "cold_new", "cold_Δ%", "hot_base", "hot_new", "hot_Δ%",
        );
        for row in &self.rows {
            let (cb, cn, cd, hb, hn, hd, status) = match &row.kind {
                ComparisonKind::Both {
                    cold,
                    hot_baseline,
                    hot_current,
                } => {
                    let hot = Self::hot_delta(*hot_baseline, *hot_current);
                    (
                        format!("{:>10}", format!("{:.0}", cold.baseline)),
                        format!("{:>10}", format!("{:.0}", cold.current)),
                        delta_cell(cold.pct),
                        format!("{:>10}", opt(*hot_baseline)),
                        format!("{:>10}", opt(*hot_current)),
                        hot.map(|d| delta_cell(d.pct)).unwrap_or_else(|| dash(9)),
                        status_cell(self.status_label(cold, hot.as_ref())),
                    )
                }
                ComparisonKind::New { stats } => (
                    dash(10),
                    format!("{:>10}", stats.cold_ms),
                    dash(9),
                    dash(10),
                    format!("{:>10}", opt(stats.hot_ms)),
                    dash(9),
                    "new".to_string(),
                ),
                ComparisonKind::Missing { stats } => (
                    format!("{:>10}", stats.cold_ms),
                    dash(10),
                    dash(9),
                    format!("{:>10}", opt(stats.hot_ms)),
                    dash(10),
                    dash(9),
                    "missing".to_string(),
                ),
            };
            println!(
                "{:<8}  {cb}  {cn}  {cd}  {hb}  {hn}  {hd}  {status}",
                row.query
            );
        }
    }

    /// Decide whether `--save-if-better` should overwrite the baseline.
    ///
    /// Sums each metric (cold, hot) across queries present in both the
    /// baseline and this run, then saves if *either* the cold suite total or
    /// the hot suite total improved by more than the regression threshold —
    /// a solid win on one axis is enough, even if the other is flat or a touch
    /// worse. Run-to-run wobble below the threshold on both axes is not enough
    /// (use `--force-save` for that). If the prior baseline had no overlap with
    /// this run there's nothing to be slower than, so we save.
    pub fn save_verdict(&self) -> SaveVerdict {
        let mut cold_base = 0.0;
        let mut cold_new = 0.0;
        let mut overlap = 0usize;
        let mut hot_base = 0.0;
        let mut hot_new = 0.0;
        let mut hot_overlap = 0usize;

        for row in &self.rows {
            if let ComparisonKind::Both {
                cold,
                hot_baseline,
                hot_current,
            } = &row.kind
            {
                overlap += 1;
                cold_base += cold.baseline;
                cold_new += cold.current;
                if let (Some(b), Some(c)) = (hot_baseline, hot_current) {
                    hot_overlap += 1;
                    hot_base += b;
                    hot_new += c;
                }
            }
        }

        if overlap == 0 {
            return SaveVerdict::NoBaseline;
        }

        let threshold = self.regression_pct;
        let cold_pct = Delta::compute(cold_base, cold_new).pct;
        let hot_pct = (hot_overlap > 0).then(|| Delta::compute(hot_base, hot_new).pct);

        let cold_wins = cold_pct <= -threshold;
        let hot_wins = hot_pct.is_some_and(|p| p <= -threshold);

        if cold_wins || hot_wins {
            SaveVerdict::Improved { cold_pct, hot_pct }
        } else {
            let hot_str = match hot_pct {
                Some(p) => format!("hot {p:+.1}%"),
                None => "no hot data".to_string(),
            };
            SaveVerdict::NotImproved(format!(
                "neither cold ({cold_pct:+.1}%) nor {hot_str} improved by more than {threshold:.0}%"
            ))
        }
    }
}
