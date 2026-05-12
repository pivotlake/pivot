//! `perf stat record -o perf.stat.data` parser.
//!
//! `perf stat record` writes the same on-disk format as `perf record`
//! (the perf.data binary format with magic `PERFILE2`), but its records
//! are stat-specific:
//!
//! * `PERF_RECORD_STAT` — one per event-id × cpu × interval, carrying
//!   the counter `val`, plus `ena`/`run` for multiplexing accounting.
//! * `PERF_RECORD_STAT_ROUND` — one per interval boundary, carrying the
//!   nanosecond-precise timestamp at which the counters were read.
//! * `PERF_EVENT_UPDATE` — runtime metadata updates (event names,
//!   units, cpu lists). We don't currently consume these because
//!   `event_attributes()` already gives us the `id → name` map for
//!   every event-id we'll encounter in `PERF_STAT` rows.
//!
//! We map each event-id (via `event_attributes()`) to one of:
//!   * UMC read CAS commands → throughput numerator (read).
//!   * UMC write CAS commands → throughput numerator (write).
//!   * L3 sampled-latency total → latency numerator.
//!   * L3 sampled-latency requests → latency denominator.
//!
//! Each interval's throughput is `(rd_sum + wr_sum) × 64 B / dt`
//! (with `dt` taken from the actual round-time delta — perf's `-I`
//! is best-effort and jitters by 1–2 ms). Each interval's latency is
//! `lat_sum / req_sum` core clocks per L3 miss.
//!
//! Time alignment with the sampled timeline relies on two anchors,
//! one per recording:
//!   * Perf side: `HEADER_CLOCK_DATA` (perf >= 5.7, requires
//!     `perf record --clockid …`), giving a precise
//!     `(wall_ns, clock_id_time_ns)` pair.
//!   * Stat side: filesystem `mtime(perf.stat.data) − last_round_time`,
//!     because perf-stat doesn't write a wall-clock anchor anywhere
//!     itself. mtime is updated on the last `STAT_ROUND` write (modes
//!     using `-o file` don't emit any rows after the child exits), so
//!     this gives a sub-second wall-clock for perf-stat's process
//!     start.

use std::path::{Path, PathBuf};

use ahash::AHashMap;
use linux_perf_data::{PerfFileReader, PerfFileRecord};
use serde::Serialize;

/// AMD64 cache-line size; each `umc_cas_cmd.{rd,wr}` corresponds to
/// one transferred line.
const CACHE_LINE_BYTES: f64 = 64.0;

/// `l3_read_miss_latency` perf metric is defined as
/// `10 × l3_xi_sampled_latency.all / l3_xi_sampled_latency_requests.all`
/// — verified empirically against the reference values perf writes
/// into the `stat.csv` 7th column. The factor of 10 accounts for
/// AMD's quantisation of `l3_xi_sampled_latency.all` (counts in 0.1
/// clock units per request, so the ratio multiplied by 10 gives
/// average clocks per L3 read miss).
const L3_LATENCY_METRIC_SCALE: f64 = 10.0;

/// Per-bucket time-series we emit to the frontend. Same shape as
/// before — only the parser implementation changed.
#[derive(Debug, Clone, Serialize, Default)]
pub struct StatTracks {
    /// Length = `buckets`. Memory throughput in bytes/second.
    pub throughput_bps: Vec<f64>,
    /// Length = `buckets`. L3 read-miss latency in core clocks per miss.
    pub latency_clocks: Vec<f64>,
    pub throughput_peak: f64,
    pub latency_peak: f64,
    /// Total span covered by the bucket grid, in seconds. Frontend
    /// uses this only for diagnostic display.
    pub duration_s: f64,
}

/// Locate the `perf.stat.data` produced by `perf stat record -o
/// perf.stat.data` next to the perf.data file. Returns `None` when
/// absent — the frontend then hides the memory tracks for that
/// recording (correct behaviour for runs without perf-stat).
pub fn locate_stat_data(perf_data: &Path) -> Option<PathBuf> {
    let dir = perf_data.parent()?;
    let candidate = dir.join("perf.stat.data");
    if candidate.is_file() { Some(candidate) } else { None }
}

/// Compute the offset (in seconds) to subtract from each stat-time
/// timestamp to put it on the profile's CLOCK_MONOTONIC axis.
///
/// * `stat_started_unix_ns` — wall-clock (CLOCK_REALTIME) ns of
///   perf-stat's process start. Best estimate is
///   `mtime(perf.stat.data) − last_round_time_ns` (sub-ns precise on
///   any Linux filesystem). See the note at the top of this module
///   for why mtime works here.
/// * `clock_anchor_wall_ns` / `clock_anchor_mono_ns` — the
///   simultaneous (wall, CLOCK_MONOTONIC) pair perf-record wrote
///   into HEADER_CLOCK_DATA when it opened its session.
/// * `profile_first_sample_mono_ns` — CLOCK_MONOTONIC of the first
///   sample in perf.data (perfy treats this as profile-axis t=0).
pub fn alignment_offset_s(
    stat_started_unix_ns: i128,
    clock_anchor_wall_ns: u64,
    clock_anchor_mono_ns: u64,
    profile_first_sample_mono_ns: u64,
) -> f64 {
    // Δ = wall − mono is constant for the system (CLOCK_MONOTONIC and
    // CLOCK_REALTIME differ by a fixed value modulo NTP slew). i128
    // because the anchor instant can sit before the first sample, so
    // the unsigned subtraction would underflow.
    let delta_ns = clock_anchor_wall_ns as i128 - clock_anchor_mono_ns as i128;
    let first_sample_wall_ns = profile_first_sample_mono_ns as i128 + delta_ns;
    (first_sample_wall_ns - stat_started_unix_ns) as f64 / 1e9
}

/// Internal: which workload-level role each event-id plays. Anything
/// else is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    UmcRead,
    UmcWrite,
    L3LatencyRequests, // metric denominator
    L3LatencyTotal,    // metric numerator
}

fn classify_event(name: &str) -> Option<Role> {
    match name {
        "amd_umc/umc_cas_cmd.rd/" => Some(Role::UmcRead),
        "amd_umc/umc_cas_cmd.wr/" => Some(Role::UmcWrite),
        "l3_xi_sampled_latency_requests.all" => Some(Role::L3LatencyRequests),
        "l3_xi_sampled_latency.all" => Some(Role::L3LatencyTotal),
        _ => None,
    }
}

/// One complete interval after aggregating across PMU instances. Used
/// by both the live parser and the unit tests, which is why this is
/// public — the unit tests build a synthetic `Vec<Interval>` and feed
/// it through [`bucket_intervals`] without touching disk.
#[derive(Debug, Clone, Copy)]
pub struct Interval {
    /// Profile-axis time (seconds), i.e. stat-time minus alignment offset.
    /// Always non-negative; pre-profile intervals are dropped before
    /// this struct is constructed.
    pub time_s: f64,
    pub throughput_bps: f64,
    pub latency_clocks: f64,
}

/// Bucket a sequence of (time, throughput, latency) intervals into
/// `buckets` slots covering `[0, axis_duration_s]`. Step-function: each
/// bucket gets the value of the most recent interval whose time is
/// `<= bucket right edge`.
pub fn bucket_intervals(
    intervals: &[Interval],
    axis_duration_s: f64,
    buckets: usize,
) -> StatTracks {
    if buckets == 0 || axis_duration_s <= 0.0 || intervals.is_empty() {
        return StatTracks::default();
    }
    let eps = axis_duration_s * 1e-9;
    let mut throughput_bps = vec![0.0f64; buckets];
    let mut latency_clocks = vec![0.0f64; buckets];
    // NEG_INFINITY init so peak tracking handles negative latency
    // deltas correctly (when the user passes --memory-base-latency
    // higher than every observed value). We coerce back to 0 below
    // if no buckets received a value.
    let mut peak_t = f64::NEG_INFINITY;
    let mut peak_l = f64::NEG_INFINITY;
    let mut idx = 0usize;
    for k in 0..buckets {
        let t_right = ((k + 1) as f64 / buckets as f64) * axis_duration_s;
        while idx + 1 < intervals.len() && intervals[idx + 1].time_s <= t_right + eps {
            idx += 1;
        }
        let iv = intervals[idx.min(intervals.len() - 1)];
        // If the current interval is still in the future (no event
        // has caught up to bucket `k` yet), leave the bucket at 0.
        if iv.time_s > t_right + eps {
            continue;
        }
        throughput_bps[k] = iv.throughput_bps;
        latency_clocks[k] = iv.latency_clocks;
        if iv.throughput_bps > peak_t {
            peak_t = iv.throughput_bps;
        }
        if iv.latency_clocks > peak_l {
            peak_l = iv.latency_clocks;
        }
    }
    StatTracks {
        throughput_bps,
        latency_clocks,
        throughput_peak: if peak_t.is_finite() { peak_t } else { 0.0 },
        latency_peak: if peak_l.is_finite() { peak_l } else { 0.0 },
        duration_s: axis_duration_s,
    }
}

/// Read the perf.stat.data binary file at `path`, decoding it into a
/// list of profile-axis intervals. Returns `None` when the file
/// can't be opened or has no STAT_ROUND records (no measurements).
///
/// `time_offset_s` is subtracted from every stat-time before the
/// interval is emitted. Intervals that fall before profile-time 0
/// (i.e. perf-stat samples taken before perf-record's first sample)
/// are dropped — they don't belong on the profile timeline.
fn read_intervals(path: &Path, time_offset_s: f64) -> Option<Vec<Interval>> {
    let f = std::fs::File::open(path).ok()?;
    let r = std::io::BufReader::new(f);
    let perf = PerfFileReader::parse_file(r).ok()?;
    let PerfFileReader { mut perf_file, mut record_iter } = perf;

    // Build id → role from event_attributes(). Empirically every
    // `PERF_STAT.id` we see is covered by the header (each PMU
    // instance gets its own attr index but still appears here),
    // so a single pass at startup is enough — no need to also
    // consume `PERF_EVENT_UPDATE` records.
    let mut id_to_role: AHashMap<u64, Role> = AHashMap::new();
    for attr in perf_file.event_attributes() {
        let Some(name) = attr.name() else { continue };
        let Some(role) = classify_event(name) else { continue };
        for &id in attr.ids() {
            id_to_role.insert(id, role);
        }
    }

    // Streaming aggregation: PERF_STAT records carry the *cumulative*
    // counter value across the entire recording, not the per-interval
    // delta. (Verified empirically — id=142454 reads 36550 in
    // interval 0, 91881 in interval 1, 94361 in interval 2, etc.) We
    // therefore track the previous cumulative value per event-id and
    // compute deltas at each STAT_ROUND boundary. Without this the
    // throughput peak comes out scaled by `total_run / interval_ms`
    // (≈ 1700× for a 17 s run with -I 10), giving e.g. 30 TB/s
    // instead of ~100 GB/s.
    let mut intervals = Vec::<Interval>::new();
    let mut prev_round_ns: u64 = 0;
    // Per-id last-seen cumulative value. Initially 0 for every id —
    // safe because PERF_STAT.val is also 0 before any sample arrives.
    let mut prev_val: AHashMap<u64, u64> = AHashMap::new();
    // Per-interval delta accumulators (reset at each STAT_ROUND).
    let mut delta_rd: u64 = 0;
    let mut delta_wr: u64 = 0;
    let mut delta_lat_req: u64 = 0;
    let mut delta_lat_tot: u64 = 0;

    while let Some(rec) = record_iter.next_record(&mut perf_file).ok().flatten() {
        let PerfFileRecord::UserRecord(u) = rec else { continue };
        // Pull the body bytes. RawData::Single is universal for the
        // small (16/40-byte) records we care about; if a record
        // happens to span buffers (Split), `as_slice` will copy and
        // hand us a Cow<[u8]>.
        let body_cow = u.data.as_slice();
        let body: &[u8] = &body_cow;
        let ty = u.record_type.record_type().0;
        match ty {
            // PERF_RECORD_STAT = 76: counter sample for one event-id.
            76 => {
                if body.len() < 40 {
                    continue;
                }
                let id = u64::from_le_bytes(body[0..8].try_into().unwrap());
                // body[8..12]  = cpu (u32) — unused, we sum across PMUs
                // body[12..16] = thread (u32) — unused, system-wide profile
                let val = u64::from_le_bytes(body[16..24].try_into().unwrap());
                // body[24..32] = ena (u64)
                // body[32..40] = run (u64) — counts at 100% in our usage
                if let Some(role) = id_to_role.get(&id) {
                    let prev = prev_val.insert(id, val).unwrap_or(0);
                    let delta = val.saturating_sub(prev);
                    match role {
                        Role::UmcRead => delta_rd = delta_rd.saturating_add(delta),
                        Role::UmcWrite => delta_wr = delta_wr.saturating_add(delta),
                        Role::L3LatencyRequests => delta_lat_req = delta_lat_req.saturating_add(delta),
                        Role::L3LatencyTotal => delta_lat_tot = delta_lat_tot.saturating_add(delta),
                    }
                }
            }
            // PERF_RECORD_STAT_ROUND = 77: interval boundary.
            // Layout (verified empirically — kernel struct order):
            //   u64 type;   // 0 = INTERVAL, 1 = FINAL
            //   u64 time;   // ns since perf-stat start
            77 => {
                if body.len() < 16 {
                    continue;
                }
                let now_ns = u64::from_le_bytes(body[8..16].try_into().unwrap());
                let dt_ns = now_ns.saturating_sub(prev_round_ns);
                if dt_ns > 0 {
                    let dt_s = dt_ns as f64 / 1e9;
                    let bytes = (delta_rd + delta_wr) as f64 * CACHE_LINE_BYTES;
                    let throughput_bps = bytes / dt_s;
                    let latency_clocks = if delta_lat_req > 0 {
                        L3_LATENCY_METRIC_SCALE * delta_lat_tot as f64
                            / delta_lat_req as f64
                    } else {
                        0.0
                    };
                    let time_s = (now_ns as f64 / 1e9) - time_offset_s;
                    if time_s >= 0.0 {
                        intervals.push(Interval { time_s, throughput_bps, latency_clocks });
                    }
                }
                prev_round_ns = now_ns;
                delta_rd = 0;
                delta_wr = 0;
                delta_lat_req = 0;
                delta_lat_tot = 0;
            }
            // Other user records (STAT_CONFIG, EVENT_UPDATE,
            // THREAD_MAP, CPU_MAP) — ignored. They carry metadata
            // we already get from `event_attributes()` or don't need.
            _ => {}
        }
    }

    if intervals.is_empty() { None } else { Some(intervals) }
}

/// Read perf.stat.data, align it to the profile's time axis, and
/// bucket into `buckets` slots covering `[0, axis_duration_s]`.
///
/// `latency_base_clocks` is subtracted from each interval's latency
/// value before bucketing, so the graph shows deviation from the
/// system's idle floor. Negative deltas (latency below baseline) are
/// preserved — the frontend renders them as zero-height but still
/// shows the signed value in the hover tooltip. Pass `0.0` to keep
/// absolute latencies. The base value is **not** subtracted from the
/// throughput series — bandwidth has no equivalent baseline.
///
/// Returns an empty [`StatTracks`] (rather than an error) on any
/// failure so the frontend can still render the timeline.
pub fn read_perf_stat_data(
    path: &Path,
    buckets: usize,
    axis_duration_s: f64,
    time_offset_s: f64,
    latency_base_clocks: f64,
) -> StatTracks {
    let Some(mut intervals) = read_intervals(path, time_offset_s) else {
        return StatTracks::default();
    };
    if latency_base_clocks != 0.0 {
        for iv in intervals.iter_mut() {
            iv.latency_clocks -= latency_base_clocks;
        }
    }
    bucket_intervals(&intervals, axis_duration_s, buckets)
}

/// One-shot read of the last `STAT_ROUND.time` value (in ns since
/// perf-stat start) from the file. We use this for the mtime-based
/// alignment: `stat_started_wall_ns ≈ mtime − last_round_time`.
///
/// Streams through every record because STAT_ROUND records appear
/// throughout the file and we want the last one. Cheap on a
/// stat-record file (a few MB, at most thousands of rounds).
pub fn last_round_time_ns(path: &Path) -> Option<u64> {
    let f = std::fs::File::open(path).ok()?;
    let r = std::io::BufReader::new(f);
    let perf = PerfFileReader::parse_file(r).ok()?;
    let PerfFileReader { mut perf_file, mut record_iter } = perf;
    let mut last: Option<u64> = None;
    while let Some(rec) = record_iter.next_record(&mut perf_file).ok().flatten() {
        let PerfFileRecord::UserRecord(u) = rec else { continue };
        if u.record_type.record_type().0 != 77 {
            continue;
        }
        let body_cow = u.data.as_slice();
        let body: &[u8] = &body_cow;
        if body.len() >= 16 {
            last = Some(u64::from_le_bytes(body[8..16].try_into().unwrap()));
        }
    }
    last
}

/// Return the file's `mtime` as nanoseconds since the Unix epoch.
/// Used as the stat-side wall-clock anchor (combined with
/// [`last_round_time_ns`]) when no other anchor is available.
pub fn mtime_unix_ns(path: &Path) -> Option<i128> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    let dur = mtime.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(dur.as_nanos() as i128)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(t: f64, bps: f64, lat: f64) -> Interval {
        Interval { time_s: t, throughput_bps: bps, latency_clocks: lat }
    }

    /// Two intervals with the bucket grid landing exactly on each
    /// interval's right edge — the step function should pick up both
    /// values and the peak is the max.
    #[test]
    fn bucketing_picks_up_each_interval() {
        let ivs = vec![iv(0.010, 1_920_000.0, 100.0), iv(0.020, 7_680_000.0, 200.0)];
        let st = bucket_intervals(&ivs, 0.020, 2);
        assert!((st.throughput_bps[0] - 1_920_000.0).abs() < 1e-6);
        assert!((st.throughput_bps[1] - 7_680_000.0).abs() < 1e-6);
        assert_eq!(st.throughput_peak, 7_680_000.0);
        assert_eq!(st.latency_peak, 200.0);
    }

    /// Buckets finer than the natural interval cadence should
    /// step-function: until the first interval lands, buckets stay
    /// at 0; thereafter they hold the most-recent interval's value.
    #[test]
    fn buckets_finer_than_intervals_step_correctly() {
        let ivs = vec![iv(0.010, 1.0, 0.0), iv(0.020, 2.0, 0.0)];
        let st = bucket_intervals(&ivs, 0.020, 4);
        // Buckets at right-edges 0.005, 0.010, 0.015, 0.020.
        // 0.005 → no interval has landed yet → 0.
        assert_eq!(st.throughput_bps[0], 0.0);
        // 0.010 → first interval just barely.
        assert!((st.throughput_bps[1] - 1.0).abs() < 1e-6);
        // 0.015 → still on first interval.
        assert!((st.throughput_bps[2] - 1.0).abs() < 1e-6);
        // 0.020 → second interval.
        assert!((st.throughput_bps[3] - 2.0).abs() < 1e-6);
    }

    /// Buckets past the last interval just keep the last value
    /// (latch-and-hold) — perf-stat may have ended before the
    /// profile did.
    #[test]
    fn buckets_past_last_interval_hold() {
        let ivs = vec![iv(0.010, 5.0, 50.0)];
        let st = bucket_intervals(&ivs, 0.030, 3);
        // bucket 0: t_right = 0.010 → first interval.
        assert_eq!(st.throughput_bps[0], 5.0);
        // buckets 1 and 2: hold.
        assert_eq!(st.throughput_bps[1], 5.0);
        assert_eq!(st.throughput_bps[2], 5.0);
    }

    /// Empty inputs return all zeros and don't panic.
    #[test]
    fn empty_inputs_yield_default() {
        let zero = bucket_intervals(&[], 1.0, 4);
        assert!(zero.throughput_bps.iter().all(|v| *v == 0.0));
        assert_eq!(zero.duration_s, 0.0);
    }

    /// `alignment_offset_s` is pure math we proved correct against
    /// the real recording earlier — keep regression coverage.
    #[test]
    fn alignment_offset_combines_anchors() {
        // CLOCK_DATA: at mono=100 s the wall was 5 s past Unix epoch.
        // First sample at mono=101.3 s → wall 6.3 s.
        // perf-stat's mtime-derived start is at 5 s exactly.
        // δ = 6.3 − 5.0 = 1.3 s.
        let off = alignment_offset_s(
            5_000_000_000,
            5_000_000_000,
            100_000_000_000,
            101_300_000_000,
        );
        assert!((off - 1.3).abs() < 1e-6, "got {off}");
    }

    /// `read_perf_stat_data`'s `latency_base_clocks` argument
    /// shifts the latency series by subtracting the baseline.
    /// Negative deltas (latency lower than baseline) are preserved
    /// — we expose the signed value so the tooltip can show "-12 clk
    /// below base" rather than mis-clamping to zero.
    #[test]
    fn latency_base_subtracts_from_series_signed() {
        // Three intervals straddling a 100-clk baseline.
        let mut ivs = vec![
            iv(0.010, 0.0, 80.0),  // 20 below base → expect -20
            iv(0.020, 0.0, 100.0), // exactly at base → expect 0
            iv(0.030, 0.0, 155.0), // 55 above base → expect +55
        ];
        for v in ivs.iter_mut() {
            v.latency_clocks -= 100.0;
        }
        let st = bucket_intervals(&ivs, 0.030, 3);
        assert!((st.latency_clocks[0] - -20.0).abs() < 1e-6);
        assert_eq!(st.latency_clocks[1], 0.0);
        assert!((st.latency_clocks[2] - 55.0).abs() < 1e-6);
        // Peak is the most-positive delta, not abs-max.
        assert!((st.latency_peak - 55.0).abs() < 1e-6);
    }

    /// When the entire latency series is below the configured base
    /// (e.g. a misconfigured `--memory-base-latency` that's higher
    /// than every observed value), peak tracking must surface the
    /// max negative value rather than clamping to 0.
    #[test]
    fn latency_peak_can_be_negative() {
        let mut ivs = vec![iv(0.010, 0.0, 80.0), iv(0.020, 0.0, 60.0)];
        for v in ivs.iter_mut() {
            v.latency_clocks -= 100.0;
        }
        let st = bucket_intervals(&ivs, 0.020, 2);
        // -20 is the larger of {-20, -40} → that's our peak.
        assert!((st.latency_peak - -20.0).abs() < 1e-6);
    }

    /// Negative offsets are possible if the system clock jumped or
    /// perf-record actually started before perf-stat (rare, but the
    /// math should not underflow).
    #[test]
    fn alignment_offset_handles_negative() {
        let off = alignment_offset_s(
            10_000_000_000,
            5_000_000_000,
            100_000_000_000,
            101_300_000_000,
        );
        assert!(off < 0.0, "got {off}");
    }
}
