//! axum HTTP routes — mirror the original Python Flask blueprint.
use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::get;
use axum::Router;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

use crate::annotate;
use crate::flamegraph;
use crate::pipeline;
use crate::profile::{Category, Profile};
use crate::stat;
use crate::tracks;

use ibs_annotate::InsnStats;

/// Shared application state. We wrap the `Profile` in an `ArcSwap` so the
/// hot-reload watcher can atomically replace it from a background thread
/// without disturbing any in-flight request — each handler grabs its own
/// snapshot via `state.load_full()` (cheap: one atomic refcount bump).
pub type AppState = Arc<ArcSwap<Profile>>;

pub fn router(state: AppState) -> Router {
    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);
    Router::new()
        .route("/api/meta", get(meta))
        .route("/api/tracks", get(get_tracks))
        .route("/api/flamegraph", get(get_flamegraph))
        .route("/api/annotate", get(get_annotate))
        .route("/api/insn_detail", get(get_insn_detail))
        .route("/api/stat", get(get_stat))
        .route("/api/pipeline_summary", get(get_pipeline_summary))
        .route("/api/health", get(health))
        .layer(cors)
        .with_state(state)
}

// -- shared helpers ----------------------------------------------------------

fn parse_categories(raw: Option<&str>, profile: &Profile) -> Vec<Category> {
    let Some(raw) = raw else { return profile.categories.clone() };
    let mut out: Vec<Category> = Vec::new();
    for tok in raw.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(c) = Category::parse(t) {
            if !out.contains(&c) {
                out.push(c);
            }
        }
    }
    // Canonical order
    Category::ALL
        .iter()
        .copied()
        .filter(|c| out.contains(c))
        .collect()
}

fn parse_int_list(raw: Option<&str>) -> Option<Vec<u32>> {
    let raw = raw?;
    let mut out: Vec<u32> = Vec::new();
    for tok in raw.split(',') {
        if let Ok(v) = tok.trim().parse::<u32>() {
            out.push(v);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

// -- /api/meta ---------------------------------------------------------------

#[derive(Serialize)]
struct MetaResponse {
    perf_data: String,
    binary: Option<String>,
    cpus: Vec<u32>,
    categories: Vec<&'static str>,
    time_start_ns: u64,
    time_end_ns: u64,
    duration_ns: u64,
    stats: crate::profile::StatsSummary,
}

async fn meta(State(state): State<AppState>) -> Json<MetaResponse> {
    let profile = state.load_full();
    Json(MetaResponse {
        perf_data: profile.perf_data_path.clone(),
        binary: profile.binary_path.clone(),
        cpus: profile.cpus.clone(),
        categories: profile.categories.iter().map(|c| c.as_str()).collect(),
        time_start_ns: profile.time_start_ns,
        time_end_ns: profile.time_end_ns,
        duration_ns: profile.duration_ns(),
        stats: profile.stats_summary(),
    })
}

// -- /api/tracks -------------------------------------------------------------

#[derive(Deserialize)]
struct TracksQuery {
    categories: Option<String>,
    consolidated: Option<String>,
    buckets: Option<usize>,
}

#[derive(Serialize)]
struct TracksResponse {
    buckets: usize,
    consolidated: bool,
    time_start_ns: u64,
    time_end_ns: u64,
    tracks: Vec<tracks::Track>,
}

async fn get_tracks(
    State(state): State<AppState>,
    Query(q): Query<TracksQuery>,
) -> Json<TracksResponse> {
    let profile = state.load_full();
    let cats = parse_categories(q.categories.as_deref(), &profile);
    let consolidated = q
        .consolidated
        .as_deref()
        .map(|s| !s.eq_ignore_ascii_case("false"))
        .unwrap_or(true);
    let buckets = q.buckets.unwrap_or(tracks::DEFAULT_BUCKETS);
    let rows = tracks::build_tracks(&profile, &cats, consolidated, buckets);
    Json(TracksResponse {
        buckets,
        consolidated,
        time_start_ns: profile.time_start_ns,
        time_end_ns: profile.time_end_ns,
        tracks: rows,
    })
}

// -- /api/flamegraph ---------------------------------------------------------

#[derive(Deserialize)]
struct FlameQuery {
    track: Option<String>,
    categories: Option<String>,
    cpus: Option<String>,
    time_lo_ns: Option<u64>,
    time_hi_ns: Option<u64>,
}

async fn get_flamegraph(
    State(state): State<AppState>,
    Query(q): Query<FlameQuery>,
) -> Result<Json<flamegraph::FlameResponse>, ApiError> {
    let profile = state.load_full();
    let (cats, cpus): (Vec<Category>, Option<Vec<u32>>) = if let Some(tid) = q.track.as_deref() {
        let (cat, cpu) = tracks::parse_track_id(tid).map_err(|e| ApiError::bad(e.to_string()))?;
        (vec![cat], cpu.map(|c| vec![c]))
    } else {
        (
            parse_categories(q.categories.as_deref(), &profile),
            parse_int_list(q.cpus.as_deref()),
        )
    };

    let time_range = match (q.time_lo_ns, q.time_hi_ns) {
        (Some(lo), Some(hi)) => Some((lo, hi)),
        _ => None,
    };

    let resp = flamegraph::build_flamegraph(&profile, &cats, cpus.as_deref(), time_range);
    Ok(Json(resp))
}

// -- /api/annotate -----------------------------------------------------------

#[derive(Deserialize)]
struct AnnotateQuery {
    function: String,
}

async fn get_annotate(
    State(state): State<AppState>,
    Query(q): Query<AnnotateQuery>,
) -> Result<Json<annotate::AnnotateResponse>, ApiError> {
    let profile = state.load_full();
    let resp = annotate::annotate_function(&profile, &q.function)
        .map_err(|e| ApiError::not_found_or_internal(e.to_string()))?;
    Ok(Json(resp))
}

// -- /api/insn_detail --------------------------------------------------------

#[derive(Deserialize)]
struct DetailQuery {
    function: String,
    offset: u64,
}

async fn get_insn_detail(
    State(state): State<AppState>,
    Query(q): Query<DetailQuery>,
) -> Json<InsnStats> {
    // Most instructions in a function won't have any samples — that's
    // normal, not an error. Return a default (all-zeros) record so the
    // frontend can render "Total samples: 0" cleanly.
    let profile = state.load_full();
    let key = format!("{}\0{}", q.function, q.offset);
    Json(profile.insn_stats.get(&key).cloned().unwrap_or_default())
}

// -- /api/stat ---------------------------------------------------------------

/// Compute the offset (seconds) to subtract from each stat-time
/// timestamp so it lands on the profile's CLOCK_MONOTONIC axis.
///
/// Two anchors are needed:
///   * Profile side: `HEADER_CLOCK_DATA` from perf.data (requires
///     `perf record --clockid …`). Sub-ns precise.
///   * Stat side: `mtime(perf.stat.data) − last STAT_ROUND.time`,
///     because perf-stat doesn't write a wall-clock anchor itself.
///     Sub-ns precise on any Linux filesystem; mtime correctly
///     represents the last interval write because perf stat record
///     emits no records after its child exits.
///
/// Returns `0.0` (no correction) when any input is missing — the
/// frontend will then show stat tracks aligned to fraction-of-span,
/// which is visibly off but not catastrophic.
fn compute_stat_offset(profile: &Profile) -> f64 {
    let Some(anchor) = profile.clock_anchor else {
        return 0.0;
    };
    let Some(stat_path) = profile.stat_data_path.as_deref() else {
        return 0.0;
    };
    let path = std::path::Path::new(stat_path);
    let Some(mtime_ns) = stat::mtime_unix_ns(path) else {
        return 0.0;
    };
    let Some(last_round_ns) = stat::last_round_time_ns(path) else {
        return 0.0;
    };
    let stat_started_unix_ns = mtime_ns - last_round_ns as i128;
    stat::alignment_offset_s(
        stat_started_unix_ns,
        anchor.wall_clock_ns,
        anchor.clock_id_time_ns,
        profile.time_start_ns,
    )
}

#[derive(Deserialize)]
struct StatQuery {
    buckets: Option<usize>,
}

#[derive(Serialize)]
struct StatResponse {
    /// `true` if a perf.stat.data was found alongside the perf.data
    /// and at least two intervals were parsed; `false` means the
    /// recording wasn't taken with `perf stat record` and the frontend
    /// should hide the memory tracks.
    available: bool,
    buckets: usize,
    /// The configured `--memory-base-latency` value in core clocks.
    /// `latency_clocks[i]` in `tracks` already has this subtracted
    /// (clamped at 0) — the frontend reports `value + latency_base`
    /// in the hover tooltip as the "absolute" latency.
    latency_base: f64,
    #[serde(flatten)]
    tracks: stat::StatTracks,
}

async fn get_stat(
    State(state): State<AppState>,
    Query(q): Query<StatQuery>,
) -> Json<StatResponse> {
    let profile = state.load_full();
    let buckets = q.buckets.unwrap_or(tracks::DEFAULT_BUCKETS);
    let axis_s = profile.duration_ns() as f64 / 1e9;
    // Compute the precise stat→profile alignment offset using
    // perf.data's HEADER_CLOCK_DATA (perf >= 5.7) and stat.csv's
    // `# started on …` header. If either is missing we fall back to
    // offset=0 (best-effort; user will see the constant skew the
    // heuristic was guarding against).
    let time_offset_s = compute_stat_offset(&profile);
    let tracks_data = match profile.stat_data_path.as_deref() {
        Some(p) => stat::read_perf_stat_data(
            std::path::Path::new(p),
            buckets,
            axis_s,
            time_offset_s,
            profile.memory_base_latency,
        ),
        None => stat::StatTracks::default(),
    };
    let available = !tracks_data.throughput_bps.is_empty()
        && tracks_data.duration_s > 0.0;
    Json(StatResponse {
        available,
        buckets,
        latency_base: profile.memory_base_latency,
        tracks: tracks_data,
    })
}

// -- /api/pipeline_summary --------------------------------------------------

#[derive(Deserialize)]
struct PipelineQuery {
    /// CLOCK_MONOTONIC ns (absolute), as the flamegraph endpoint
    /// expects them. Both must be present to take effect; missing
    /// either falls back to the full-recording aggregate.
    time_lo_ns: Option<u64>,
    time_hi_ns: Option<u64>,
}

async fn get_pipeline_summary(
    State(state): State<AppState>,
    Query(q): Query<PipelineQuery>,
) -> Json<pipeline::PipelineSummary> {
    let profile = state.load_full();
    let summary = match profile.stat_data_path.as_deref() {
        Some(p) => {
            // Convert the profile-axis (CLOCK_MONOTONIC) range to the
            // stat axis (ns since perf-stat process start) using the
            // same alignment we use for the timeline tracks. Drops
            // out to the "full recording" path if either endpoint is
            // missing or the alignment can't be computed.
            let stat_range = match (q.time_lo_ns, q.time_hi_ns) {
                (Some(lo_mono), Some(hi_mono)) => {
                    let offset_s = compute_stat_offset(&profile);
                    let offset_ns = (offset_s * 1e9) as i128;
                    let first_mono = profile.time_start_ns as i128;
                    let lo = (lo_mono as i128 - first_mono + offset_ns).max(0) as u64;
                    let hi = (hi_mono as i128 - first_mono + offset_ns).max(0) as u64;
                    if hi > lo { Some((lo, hi)) } else { None }
                }
                _ => None,
            };
            pipeline::compute_pipeline_summary(std::path::Path::new(p), stat_range)
        }
        None => pipeline::PipelineSummary::default(),
    };
    Json(summary)
}

// -- /api/health -------------------------------------------------------------

#[derive(Serialize)]
struct Health {
    ok: bool,
}
async fn health() -> Json<Health> {
    Json(Health { ok: true })
}

// -- error wrapper ----------------------------------------------------------

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    fn bad(message: String) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message }
    }
    fn not_found_or_internal(message: String) -> Self {
        let status = if message.contains("has no observed samples") {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        Self { status, message }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        #[derive(Serialize)]
        struct Body { error: String }
        (self.status, Json(Body { error: self.message })).into_response()
    }
}
