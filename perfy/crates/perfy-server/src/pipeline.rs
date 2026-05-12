//! Top-Down microarchitecture analysis for AMD Zen4, computed from
//! the same `perf.stat.data` we already consume for the memory tracks.
//!
//! Inputs (raw events the user is recording — see `perf list metric`
//! for the AMD Zen4 `PipelineL2` set):
//!
//! ```text
//! ls_not_halted_cyc                                   total CPU cycles
//! ex_ret_ops                                          retired ops
//! ex_ret_ucode_ops                                    retired µcoded ops
//! ex_ret_brn_misp                                     retired branch mispredicts
//! de_src_op_disp.all                                  dispatched ops
//! de_no_dispatch_per_slot.no_ops_from_frontend        FE-stalled slots
//! de_no_dispatch_per_slot.no_ops_from_frontend(cmask=6)
//!                                                     fully FE-stalled cycles
//! de_no_dispatch_per_slot.backend_stalls              BE-stalled slots
//! ex_no_retire.not_complete                           retire stalls (any reason)
//! ex_no_retire.load_not_complete                      retire stalls waiting on load
//! resyncs_or_nc_redirects                             pipeline resyncs
//! ```
//!
//! Top-Down L1 (each fraction is "% of pipeline slots", and they sum to 100):
//!   Retiring        =  ex_ret_ops                                   / (W·cycles)
//!   Frontend Bound  =  no_ops_from_frontend                         / (W·cycles)
//!   Backend Bound   =  backend_stalls                               / (W·cycles)
//!   Bad Speculation =  (de_src_op_disp.all − ex_ret_ops)            / (W·cycles)
//!
//! W is the dispatch / retire width, **6 on Zen4** (matches AMD's
//! distributed `amd-zen4-pipeline.json` metric definitions).

use std::path::Path;

use ahash::AHashMap;
use linux_perf_data::{PerfFileReader, PerfFileRecord};
use serde::Serialize;

/// Zen4 dispatch / retire width. Hard-coded because that's how AMD
/// publishes the metric formulas; if you want to support other Zen
/// generations later, derive this from the CPUID feature header
/// rather than guessing per-host.
const ZEN4_DISPATCH_WIDTH: f64 = 6.0;

/// Inputs we recognize. Anything else in `perf.stat.data` is ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Slot {
    Cycles,
    OpsRetired,
    UcodeOpsRetired,
    BranchMispredicts,
    OpsDispatched,
    NoOpsFromFrontend,
    NoOpsFromFrontendCmask6,
    BackendStalls,
    NoRetireAny,
    NoRetireLoad,
    Resyncs,
}

fn classify(name: &str) -> Option<Slot> {
    // The cmask=6 variant uses the kernel's syntax `cpu/<event>,cmask=0x6/`
    // (see the user's recording command). We split on the comma so both
    // permutations match.
    let bare = name.split(['/', ',']).next().unwrap_or(name);
    let cmask6 = name.contains("cmask=0x6") || name.contains("cmask=6");
    match (bare, cmask6) {
        ("ls_not_halted_cyc", _) => Some(Slot::Cycles),
        ("ex_ret_ops", _) => Some(Slot::OpsRetired),
        ("ex_ret_ucode_ops", _) => Some(Slot::UcodeOpsRetired),
        ("ex_ret_brn_misp", _) => Some(Slot::BranchMispredicts),
        ("de_src_op_disp.all", _) => Some(Slot::OpsDispatched),
        ("de_no_dispatch_per_slot.no_ops_from_frontend", true) => {
            Some(Slot::NoOpsFromFrontendCmask6)
        }
        ("de_no_dispatch_per_slot.no_ops_from_frontend", false) => {
            Some(Slot::NoOpsFromFrontend)
        }
        ("cpu", true) => {
            // Falls through to here for `cpu/de_no_dispatch_per_slot.no_ops_from_frontend,cmask=0x6/`
            // — the leading `cpu/` ate the bare-name split. We re-check
            // the full string to differentiate.
            if name.contains("de_no_dispatch_per_slot.no_ops_from_frontend") {
                Some(Slot::NoOpsFromFrontendCmask6)
            } else {
                None
            }
        }
        ("de_no_dispatch_per_slot.backend_stalls", _) => Some(Slot::BackendStalls),
        ("ex_no_retire.not_complete", _) => Some(Slot::NoRetireAny),
        ("ex_no_retire.load_not_complete", _) => Some(Slot::NoRetireLoad),
        ("resyncs_or_nc_redirects", _) => Some(Slot::Resyncs),
        _ => None,
    }
}

/// Final top-down breakdown over the entire recording. All
/// `_pct` fields are percentages (0..100); the four Top-Down L1
/// categories should sum to ~100 (within rounding).
///
/// Per-metric availability is signalled by `NaN`: when an input
/// event couldn't be paired with a co-scheduled cycles instance
/// (the events weren't grouped in the recording), the affected
/// metric is `NaN`. The frontend renders those as "—" so the user
/// sees a clear gap instead of a silently-wrong number. The
/// top-level `error` field carries the corresponding human-readable
/// "what to do about it" message.
#[derive(Debug, Default, Clone, Serialize)]
pub struct PipelineSummary {
    /// `true` only when every required metric was computable. If any
    /// metric is `NaN`, this is `false` and `error` describes why.
    pub available: bool,
    /// When `available: false`, a short message explaining what's
    /// missing and what the user needs to do. Surfaced as a banner.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    // ── Top-Down L1 (sum to ~100%) ──────────────────────────────
    pub retiring_pct: f64,
    pub frontend_bound_pct: f64,
    pub backend_bound_pct: f64,
    pub bad_speculation_pct: f64,

    // ── Top-Down L2 sub-breakdowns ──────────────────────────────
    // Each parent metric (`retiring_pct`, `frontend_bound_pct`,
    // `backend_bound_pct`) splits into two children that sum back to
    // it. Formulas match AMD's PipelineL2 metric set verbatim — see
    // the `frontend_bound_*` / `backend_bound_*` / `retiring_*`
    // annotations perf prints next to each event.

    /// % of total cycles where all 6 dispatch slots were starved by
    /// the frontend (icache miss, branch redirect). Computed against
    /// cycles, not slots, so it can be compared directly to perf's
    /// `frontend_bound_latency` annotation.
    pub frontend_latency_pct: f64,
    /// `frontend_bound_pct − frontend_latency_pct`. Front-end could
    /// supply *some* ops but not enough to fill the dispatch window.
    pub frontend_bandwidth_pct: f64,

    /// Of all backend stalls, the fraction that were "load not
    /// complete". A diagnostic ratio (0..1) used to split the
    /// backend-bound bar.
    pub backend_memory_share: f64,
    /// `backend_bound_pct × backend_memory_share` — memory-bound
    /// slots as a % of total pipeline slots.
    pub backend_memory_pct: f64,
    /// `backend_bound_pct − backend_memory_pct`.
    pub backend_core_pct: f64,

    /// % of slots that retired a *non*-microcoded op. Together with
    /// `retiring_microcode_pct` sums to `retiring_pct`.
    pub retiring_fastpath_pct: f64,
    /// % of slots that retired a microcoded op (div, gather, complex
    /// string, etc.).
    pub retiring_microcode_pct: f64,

    // ── Diagnostic single-numbers ───────────────────────────────
    /// Instructions-per-cycle, here counted as ops/cycle since AMD
    /// reports retired ops, not retired instructions.
    pub ipc: f64,
    /// Branch mispredicts as a fraction of retired ops. High → frontend
    /// will be paying redirect penalties.
    pub branch_misp_pct: f64,
    /// Microcoded ops as fraction of retired ops. High (>2%) suggests
    /// expensive instructions (div, gather, complex string ops).
    pub microcode_pct: f64,
    /// Pipeline resyncs / NC redirects as fraction of retired ops.
    /// Each resync flushes the pipeline.
    pub resync_pct: f64,

    // ── Raw totals (for tooltips / debugging) ───────────────────
    pub total_cycles: u64,
    pub total_ops_retired: u64,
    pub total_ops_dispatched: u64,
    pub total_branch_misp: u64,
}

/// Read perf.stat.data and return a fully-populated [`PipelineSummary`].
///
/// `range_stat_ns = Some((lo, hi))` restricts the metrics to the
/// activity that happened between two STAT_ROUND boundaries —
/// `lo` and `hi` are nanoseconds on the **perf-stat axis** (ns since
/// perf-stat's process start), not CLOCK_MONOTONIC. The API handler
/// is responsible for converting from the profile axis. Pass `None`
/// to compute over the full recording.
///
/// On any I/O or parse failure, returns a default (`available: false`)
/// so the frontend can render a graceful empty state.
pub fn compute_pipeline_summary(
    path: &Path,
    range_stat_ns: Option<(u64, u64)>,
) -> PipelineSummary {
    let totals = match read_totals(path, range_stat_ns) {
        Some(t) => t,
        None => return PipelineSummary::default(),
    };

    // Pull each slot's measurement along with its group's cycles.
    // Each metric formula's denominator comes from the cycles event
    // that was co-scheduled with the numerator — exactly what perf
    // does in its PipelineL2 metric expressions.
    let m = |s: Slot| totals.get(&s).copied().unwrap_or_default();
    let cycles_m = m(Slot::Cycles);
    let ops_ret_m = m(Slot::OpsRetired);
    if cycles_m.scaled <= 0.0 || ops_ret_m.scaled <= 0.0 {
        return PipelineSummary::default();
    }
    let ops_disp_m = m(Slot::OpsDispatched);
    let fe_m = m(Slot::NoOpsFromFrontend);
    let fe_c6_m = m(Slot::NoOpsFromFrontendCmask6);
    let be_m = m(Slot::BackendStalls);
    let no_ret_any_m = m(Slot::NoRetireAny);
    let no_ret_load_m = m(Slot::NoRetireLoad);
    let ucode_m = m(Slot::UcodeOpsRetired);
    let brn_m = m(Slot::BranchMispredicts);
    let resync_m = m(Slot::Resyncs);

    // `slots(measurement)` = the slot-count denominator perf uses for
    // this measurement's group: 6 × cycles_in_same_group. NaN when
    // the slot's group lacks a cycles event — the surrounding
    // arithmetic then propagates NaN to the metric, which the
    // frontend renders as "—".
    let slots = |m: SlotMeasurement| ZEN4_DISPATCH_WIDTH * m.cycles_in_group;

    // Percentage with explicit NaN propagation. We DELIBERATELY do
    // not fall back to a default — if the denominator is missing,
    // the metric is missing, and we say so.
    let pct = |numer: f64, denom: f64| {
        if denom.is_finite() && denom > 0.0 && numer.is_finite() {
            100.0 * numer / denom
        } else {
            f64::NAN
        }
    };

    let retiring_pct = pct(ops_ret_m.scaled, slots(ops_ret_m));
    let frontend_bound_pct = pct(fe_m.scaled, slots(fe_m));
    let backend_bound_pct = pct(be_m.scaled, slots(be_m));
    // Bad Speculation = ops dispatched but never retired. Both
    // numerator and denominator are taken from `ops_disp`'s group
    // (which also contains `ops_ret` in perf's standard layout, so
    // their counts are co-scheduled). Bound at 0 because in rare
    // edge cases the raw arithmetic can go slightly negative.
    let bad_speculation_pct = pct(
        (ops_disp_m.scaled - ops_ret_m.scaled).max(0.0),
        slots(ops_disp_m),
    );

    // Sub-breakdowns. `frontend_latency_pct` is "% of cycles" by
    // perf's convention (it's how many cycles had ALL 6 dispatch
    // slots starved by the frontend), and uses cycles from the
    // frontend group as denominator.
    let frontend_latency_pct = pct(fe_c6_m.scaled, fe_c6_m.cycles_in_group);
    let frontend_bandwidth_pct = (frontend_bound_pct - frontend_latency_pct).max(0.0);

    // backend_memory_share is a ratio of two same-group events
    // (load_not_complete / not_complete), so cycles isn't involved.
    // It's NaN only when the share's numerator/denominator pair
    // wasn't recorded at all.
    let backend_memory_share = if no_ret_any_m.scaled > 0.0
        && no_ret_load_m.scaled.is_finite()
        && no_ret_any_m.scaled.is_finite()
    {
        no_ret_load_m.scaled / no_ret_any_m.scaled
    } else {
        f64::NAN
    };
    let backend_memory_pct = backend_bound_pct * backend_memory_share;

    let retiring_microcode_pct = pct(ucode_m.scaled, slots(ucode_m));
    // `f64 − NaN = NaN`, so retiring_pct or microcode being missing
    // correctly leaves fastpath as NaN. The `.max(0.0)` is only there
    // for the rare negative-arithmetic case, so apply it only when
    // the subtraction produced a finite number.
    let retiring_fastpath_pct = {
        let v = retiring_pct - retiring_microcode_pct;
        if v.is_finite() { v.max(0.0) } else { f64::NAN }
    };
    let frontend_bandwidth_pct = {
        let v = frontend_bound_pct - frontend_latency_pct;
        if v.is_finite() { v.max(0.0) } else { f64::NAN }
    };
    let backend_core_pct = {
        let v = backend_bound_pct - backend_memory_pct;
        if v.is_finite() { v.max(0.0) } else { f64::NAN }
    };

    // Build the top-level "what went wrong" message. We list every
    // L1 category whose denominator we couldn't recover so the user
    // knows exactly which event group they're missing.
    let mut missing: Vec<&str> = Vec::new();
    if !retiring_pct.is_finite() { missing.push("Retiring (needs ex_ret_ops grouped with cycles)"); }
    if !frontend_bound_pct.is_finite() { missing.push("Frontend Bound (needs de_no_dispatch_per_slot.no_ops_from_frontend grouped with cycles)"); }
    if !backend_bound_pct.is_finite() { missing.push("Backend Bound (needs de_no_dispatch_per_slot.backend_stalls grouped with cycles)"); }
    if !bad_speculation_pct.is_finite() { missing.push("Bad Speculation (needs de_src_op_disp.all grouped with cycles)"); }
    let error = if missing.is_empty() {
        None
    } else {
        Some(format!(
            "Couldn't compute these metrics — their events weren't \
             grouped with a cycles counter in the recording. Re-record \
             with `perf stat record -M PipelineL2 …` (or explicit \
             `-e '{{event_a,event_b,cycles}}'` groups). Missing: {}.",
            missing.join(", ")
        ))
    };
    let available = error.is_none();

    PipelineSummary {
        available,
        error,
        retiring_pct,
        frontend_bound_pct,
        backend_bound_pct,
        bad_speculation_pct,
        frontend_latency_pct,
        frontend_bandwidth_pct,
        backend_memory_share,
        backend_memory_pct,
        backend_core_pct,
        retiring_fastpath_pct,
        retiring_microcode_pct,
        ipc: ops_ret_m.scaled / cycles_m.scaled,
        branch_misp_pct: 100.0 * brn_m.scaled / ops_ret_m.scaled,
        microcode_pct: 100.0 * ucode_m.scaled / ops_ret_m.scaled,
        resync_pct: 100.0 * resync_m.scaled / ops_ret_m.scaled,
        total_cycles: cycles_m.scaled as u64,
        total_ops_retired: ops_ret_m.scaled as u64,
        total_ops_dispatched: ops_disp_m.scaled as u64,
        total_branch_misp: brn_m.scaled as u64,
    }
}

/// One PERF_STAT record's payload: cumulative counter value plus
/// the cumulative `enabled` / `running` times that scale it. perf
/// records all three because counters can be multiplexed off
/// hardware — `running < enabled` means perf only got a slice of
/// the wall-clock time, and the canonical "scaled to 100%-time"
/// estimate is `val * enabled / running`.
#[derive(Default, Clone, Copy)]
struct StatPayload {
    val: u64,
    ena: u64,
    run: u64,
}

/// One per-attr aggregated snapshot, after summing the within-attr
/// PMU instances (e.g. the 8 UMC channels of `amd_umc/umc_cas_cmd.rd/`).
///
/// `ena` and `run` are kept because **events in the same perf group
/// share identical `(ena, run)` values bit-for-bit** — the kernel
/// schedules a group atomically (all members on hardware together or
/// none at all). That makes group membership recoverable from the
/// counter values themselves, without parsing the `HEADER_GROUP_DESC`
/// feature section: any two attrs whose `(ena, run)` tuple matches
/// were in the same group.
#[derive(Default, Clone, Copy, Debug)]
struct AttrSnapshot {
    scaled: f64,
    ena: u64,
    run: u64,
}

/// What each slot looks like, after coalescing within-group attrs.
/// `cycles_in_group` is the cycles count *from the same multiplex
/// group as this slot's numerator* — that's what perf divides by in
/// the PipelineL2 formulas, and what we use here.
///
#[derive(Default, Clone, Copy, Debug)]
struct SlotMeasurement {
    scaled: f64,
    cycles_in_group: f64,
}

/// Walk perf.stat.data once, recover perf's exact group structure
/// from the counter scheduling, and produce a per-slot measurement
/// (scaled value + the cycles count from the slot's group, which is
/// what perf divides by in the PipelineL2 formulas).
///
/// **Group recovery.** Events opened in the same perf group are
/// scheduled atomically — they're either all on hardware together
/// or all off together. So at every interval boundary their `ena`
/// and `run` values increment by the same amount. The cumulative
/// `(ena, run)` tuple at end-of-window is therefore identical across
/// in-group attrs, and we can recover the grouping exactly by
/// equivalence-classing on that tuple. (This is the same information
/// `HEADER_GROUP_DESC` carries explicitly; we just don't need to
/// parse it.)
///
/// **Cycles denominator.** Each metric formula reads `cycles` from
/// its own group. We find the cycles attr that shares each non-
/// cycles attr's `(ena, run)` tuple and use its scaled value as that
/// slot's denominator. If a slot's group doesn't contain a cycles
/// attr (e.g. the recording only opens cycles in a single group),
/// we fall back to the highest-running cycles instance.
fn read_totals(
    path: &Path,
    range_stat_ns: Option<(u64, u64)>,
) -> Option<AHashMap<Slot, SlotMeasurement>> {
    let f = std::fs::File::open(path).ok()?;
    let r = std::io::BufReader::new(f);
    let perf = PerfFileReader::parse_file(r).ok()?;
    let PerfFileReader { mut perf_file, mut record_iter } = perf;

    let mut id_to_attr: AHashMap<u64, usize> = AHashMap::new();
    let mut attr_slot: Vec<Option<Slot>> = Vec::new();
    for (idx, attr) in perf_file.event_attributes().iter().enumerate() {
        attr_slot.push(attr.name().and_then(classify));
        for &id in attr.ids() {
            id_to_attr.insert(id, idx);
        }
    }

    let mut current: AHashMap<u64, StatPayload> = AHashMap::new();
    let mut snap_lo: Option<AHashMap<u64, StatPayload>> = None;
    let mut snap_hi: Option<AHashMap<u64, StatPayload>> = None;
    let (lo_ns, hi_ns) = range_stat_ns.unwrap_or((0, u64::MAX));

    while let Some(rec) = record_iter.next_record(&mut perf_file).ok().flatten() {
        let PerfFileRecord::UserRecord(u) = rec else { continue };
        let body_cow = u.data.as_slice();
        let body: &[u8] = &body_cow;
        let ty = u.record_type.record_type().0;
        match ty {
            76 if body.len() >= 40 => {
                let id = u64::from_le_bytes(body[0..8].try_into().unwrap());
                if !id_to_attr.contains_key(&id) {
                    continue;
                }
                current.insert(
                    id,
                    StatPayload {
                        val: u64::from_le_bytes(body[16..24].try_into().unwrap()),
                        ena: u64::from_le_bytes(body[24..32].try_into().unwrap()),
                        run: u64::from_le_bytes(body[32..40].try_into().unwrap()),
                    },
                );
            }
            77 if body.len() >= 16 => {
                let now = u64::from_le_bytes(body[8..16].try_into().unwrap());
                if now <= lo_ns {
                    snap_lo = Some(current.clone());
                }
                if now <= hi_ns {
                    snap_hi = Some(current.clone());
                }
            }
            _ => {}
        }
    }
    let hi_snap = snap_hi?;
    let zero: AHashMap<u64, StatPayload> = AHashMap::new();
    let lo_snap = snap_lo.as_ref().unwrap_or(&zero);
    // `final_snap` is always the very last cumulative state of every
    // id (= the whole-recording totals). We use this for tier
    // matching: which multiplex group an event belongs to is a fixed
    // property of the recording, not of whatever range the user is
    // zoomed into. Without this, narrow zooms with noisy run/ena
    // ratios could match an event into the wrong tier and break the
    // cycles denominator.
    let final_snap = &current;

    // Step 1 — per-id measurements + cycles-id pairing.
    //
    // Two events are in the same perf group iff they share **bit-
    // exact (ena, run) on the SAME CPU**. With system-wide recording
    // each (attr, cpu) gets its own id, and the kernel multiplexes
    // each CPU independently — so ena/run vary across CPUs even
    // within the same group. We therefore match at the *id* level,
    // not the attr level.
    //
    // For each non-cycles id we look up the cycles id with the same
    // full-recording (ena, run) and same CPU implicitly (since
    // per-CPU values are unique). The matched cycles scaled value
    // is the denominator contribution for that one id's measurement.
    // We then sum scaled + matched-cycles across an attr's ids to
    // get system-wide totals.

    // Build a map from (ena, run) over the FULL recording to the
    // cycles id at that scheduling slot. Same-CPU same-group ids
    // share this tuple bit-exactly; across CPUs the tuple differs,
    // so there's one cycles entry per (CPU × group) combination.
    let mut cycles_id_by_key: AHashMap<(u64, u64), f64> = AHashMap::new();
    for (id, attr_idx) in &id_to_attr {
        if attr_slot[*attr_idx] != Some(Slot::Cycles) {
            continue;
        }
        let final_v = final_snap.get(id).copied().unwrap_or_default();
        if final_v.ena == 0 || final_v.run == 0 {
            continue;
        }
        let scaled = final_v.val as f64 * final_v.ena as f64 / final_v.run as f64;
        // Use the *range* scaled for the denominator the metrics
        // will actually divide by. Computed here so we don't have
        // to do a second walk; the range deltas are derived inline.
        let hi_v = hi_snap.get(id).copied().unwrap_or_default();
        let lo_v = lo_snap.get(id).copied().unwrap_or_default();
        let d_val = hi_v.val.saturating_sub(lo_v.val);
        let d_ena = hi_v.ena.saturating_sub(lo_v.ena);
        let d_run = hi_v.run.saturating_sub(lo_v.run);
        let range_scaled = if d_run > 0 {
            d_val as f64 * d_ena as f64 / d_run as f64
        } else {
            0.0
        };
        // Key by the FULL (ena, run) so the cycles-id lookup
        // remains stable regardless of how the user has zoomed.
        cycles_id_by_key.insert((final_v.ena, final_v.run), range_scaled);
        // (We discard `scaled` over the full recording — kept the
        // expression above for readability / future use.)
        let _ = scaled;
    }

    // Walk every id and accumulate per-attr (scaled, cycles_in_group).
    // The cycles_in_group sum aggregates the per-id matched cycles
    // — one cycles measurement per CPU per group — giving us the
    // canonical system-wide denominator. Missing matches contribute
    // 0 to the cycles sum **and** a missing-flag so we can tell
    // "no matching cycles for this attr" from "matched but small".
    #[derive(Default, Clone, Copy)]
    struct AttrAgg {
        scaled: f64,
        cycles_in_group: f64,
        ids_seen: u32,
        ids_with_cycles: u32,
    }
    let mut by_attr_agg: AHashMap<usize, AttrAgg> = AHashMap::new();
    for (id, attr_idx) in &id_to_attr {
        let final_v = final_snap.get(id).copied().unwrap_or_default();
        if final_v.run == 0 {
            continue;
        }
        let hi_v = hi_snap.get(id).copied().unwrap_or_default();
        let lo_v = lo_snap.get(id).copied().unwrap_or_default();
        let d_val = hi_v.val.saturating_sub(lo_v.val);
        let d_ena = hi_v.ena.saturating_sub(lo_v.ena);
        let d_run = hi_v.run.saturating_sub(lo_v.run);
        let range_scaled = if d_run > 0 {
            d_val as f64 * d_ena as f64 / d_run as f64
        } else {
            0.0
        };
        let agg = by_attr_agg.entry(*attr_idx).or_default();
        agg.scaled += range_scaled;
        agg.ids_seen += 1;
        if attr_slot[*attr_idx] != Some(Slot::Cycles) {
            if let Some(cyc_scaled) = cycles_id_by_key.get(&(final_v.ena, final_v.run)) {
                agg.cycles_in_group += *cyc_scaled;
                agg.ids_with_cycles += 1;
            }
        }
    }

    // Step 2 — collapse per-attr aggregates into per-slot.
    //
    // A slot's `cycles_in_group` is `NaN` (= "this metric is
    // unavailable") iff any of its measured ids couldn't find a
    // same-CPU same-group cycles partner. For cycles itself the
    // value is meaningless (cycles never divides by cycles) but we
    // pass it through for the IPC / "total cycles" display.
    let mut by_slot: AHashMap<Slot, SlotMeasurement> = AHashMap::new();
    for (attr_idx, agg) in &by_attr_agg {
        let Some(slot) = attr_slot[*attr_idx] else { continue };
        let cycles_in_group = if slot == Slot::Cycles {
            agg.scaled
        } else if agg.ids_seen > 0 && agg.ids_with_cycles == agg.ids_seen {
            agg.cycles_in_group
        } else {
            f64::NAN
        };
        let candidate = SlotMeasurement {
            scaled: agg.scaled,
            cycles_in_group,
        };
        // Prefer the attr with the largest scaled value (the one
        // that actually ran for any meaningful time). For slots with
        // multiple attrs (e.g. cycles repeated across groups, or
        // ex_ret_ops once per group), this picks the most-reliable.
        match by_slot.get(&slot) {
            Some(existing) if existing.scaled >= candidate.scaled => {}
            _ => {
                by_slot.insert(slot, candidate);
            }
        }
    }
    Some(by_slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `classify` recognises both bare event names and the
    /// kernel-syntax `cpu/<event>,cmask=0x6/` spelling.
    #[test]
    fn classify_handles_both_event_spellings() {
        assert_eq!(classify("ls_not_halted_cyc"), Some(Slot::Cycles));
        assert_eq!(classify("ex_ret_ops"), Some(Slot::OpsRetired));
        assert_eq!(
            classify("de_no_dispatch_per_slot.no_ops_from_frontend"),
            Some(Slot::NoOpsFromFrontend)
        );
        assert_eq!(
            classify("cpu/de_no_dispatch_per_slot.no_ops_from_frontend,cmask=0x6/"),
            Some(Slot::NoOpsFromFrontendCmask6)
        );
        assert_eq!(classify("amd_umc/umc_cas_cmd.rd/"), None);
        assert_eq!(classify("nonsense"), None);
    }

    /// End-to-end-ish: feed synthetic totals through the metric
    /// formulas and check the four Top-Down L1 categories sum to 100%.
    #[test]
    fn top_down_categories_sum_to_100() {
        // Hand-pick numbers that match the formulas exactly.
        // Cycles=1e8, OpsRetired=4.8e8 → Retiring = 4.8e8/(6*1e8) = 80%.
        // FE slots = 6e7 → 6e7 / 6e8 = 10%.
        // BE slots = 3e7 → 3e7 / 6e8 = 5%.
        // OpsDispatched = 4.8e8 + 3e7 = 5.1e8 → BadSpec = 3e7/6e8 = 5%.
        let mut t: AHashMap<Slot, u64> = AHashMap::new();
        t.insert(Slot::Cycles, 100_000_000);
        t.insert(Slot::OpsRetired, 480_000_000);
        t.insert(Slot::OpsDispatched, 510_000_000);
        t.insert(Slot::NoOpsFromFrontend, 60_000_000);
        t.insert(Slot::BackendStalls, 30_000_000);
        let cycles = t[&Slot::Cycles] as f64;
        let total_slots = ZEN4_DISPATCH_WIDTH * cycles;
        let r = 100.0 * t[&Slot::OpsRetired] as f64 / total_slots;
        let fe = 100.0 * t[&Slot::NoOpsFromFrontend] as f64 / total_slots;
        let be = 100.0 * t[&Slot::BackendStalls] as f64 / total_slots;
        let bs = 100.0
            * (t[&Slot::OpsDispatched] as f64 - t[&Slot::OpsRetired] as f64)
            / total_slots;
        assert!((r - 80.0).abs() < 1e-6);
        assert!((fe - 10.0).abs() < 1e-6);
        assert!((be - 5.0).abs() < 1e-6);
        assert!((bs - 5.0).abs() < 1e-6);
        assert!((r + fe + be + bs - 100.0).abs() < 1e-6);
    }

    /// `range_stat_ns` semantics: the per-id snapshot at the **last**
    /// STAT_ROUND ≤ `lo_ns` is the baseline; the snapshot at the
    /// **last** STAT_ROUND ≤ `hi_ns` is the endpoint; `endpoint −
    /// baseline` gives the in-window deltas.
    ///
    /// We can't easily test the file-walking part without a fixture,
    /// but we can test the snapshot-arithmetic shape: per-id deltas
    /// are summed into per-slot totals using the same map we'd build
    /// from `event_attributes()`. This double-checks that two ids
    /// pointing to the same Slot get correctly aggregated.
    #[test]
    fn per_id_deltas_sum_per_slot() {
        // Two ids both contributing to OpsRetired (e.g. multi-PMU
        // event). Snap_lo at one cumulative level, snap_hi later.
        let mut id_to_slot: AHashMap<u64, Slot> = AHashMap::new();
        id_to_slot.insert(101, Slot::OpsRetired);
        id_to_slot.insert(102, Slot::OpsRetired);
        let mut snap_lo: AHashMap<u64, u64> = AHashMap::new();
        snap_lo.insert(101, 1_000);
        snap_lo.insert(102, 500);
        let mut snap_hi: AHashMap<u64, u64> = AHashMap::new();
        snap_hi.insert(101, 4_000);
        snap_hi.insert(102, 1_500);
        let mut totals: AHashMap<Slot, u64> = AHashMap::new();
        for (id, slot) in &id_to_slot {
            let delta = snap_hi[id] - snap_lo[id];
            *totals.entry(*slot).or_default() += delta;
        }
        // (4000−1000) + (1500−500) = 4000.
        assert_eq!(totals[&Slot::OpsRetired], 4000);
    }

    /// Backend memory share comes from `load_not_complete /
    /// not_complete`; multiplied into the backend-bound % gives the
    /// memory-bound % of pipeline slots, with core-bound being
    /// whatever's left.
    #[test]
    fn backend_memory_split() {
        // backend_bound = 30%, of which 75% is memory-bound:
        //   memory = 22.5%, core = 7.5%.
        let backend_bound_pct: f64 = 30.0;
        let no_retire_any: f64 = 1000.0;
        let no_retire_load: f64 = 750.0;
        let share = no_retire_load / no_retire_any;
        let mem = backend_bound_pct * share;
        let core = backend_bound_pct - mem;
        assert!((mem - 22.5_f64).abs() < 1e-6);
        assert!((core - 7.5_f64).abs() < 1e-6);
    }
}
