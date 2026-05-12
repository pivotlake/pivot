//! Build a server [`Profile`] from a `perf.data` file by driving
//! `ibs_annotate::reader`.
//!
//! All sample-stream parsing now happens in the `ibs-annotate` crate (which
//! reads the binary `perf.data` directly via `linux-perf-data`). This file
//! is responsible for projecting that into the perfy-server-specific shape:
//!   - per-sample timeline records keyed by Category (Cycles/L1/L2/L3/Dram),
//!   - an interned [`FrameTable`] of demangled function names,
//!   - the canonical (cpus, categories) axes for the UI.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ahash::{AHashMap, AHashSet};
use anyhow::Context;
use rayon::prelude::*;

use ibs_annotate::reader::{read_perf_data, EventClass, ParsedSample, Profile as RawProfile};
use ibs_annotate::{AddressSpaces, CacheLevel, SymbolCache};

use crate::profile::{collect_axes, Category, FrameTable, Profile, Sample};

const UNKNOWN_FRAME: &str = "[unknown]";

fn category_for(sample: &ParsedSample, class: EventClass) -> Option<Category> {
    match class {
        EventClass::Cycles => Some(Category::Cycles),
        EventClass::IbsOp => match sample.cache? {
            CacheLevel::L1 | CacheLevel::LFB => Some(Category::L1),
            CacheLevel::L2 => Some(Category::L2),
            CacheLevel::L3 => Some(Category::L3),
            CacheLevel::DRAM | CacheLevel::REMOTE => Some(Category::Dram),
            CacheLevel::NonMemory => None,
        },
        EventClass::Other => None,
    }
}

/// Convert a `ParsedSample`'s perf-event period into the `weight` that
/// the timeline bucketer expects. Period == 0 happens when perf didn't
/// emit one (unusual, but we've seen it on synthetic events); we fall
/// back to 1 so the sample still shows up rather than vanishing.
fn weight_for(period: u64) -> u64 {
    period.max(1)
}

pub fn parse_perf_data(perf_data: &str) -> anyhow::Result<Profile> {
    let t_read = Instant::now();
    let raw: RawProfile = read_perf_data(Path::new(perf_data))
        .with_context(|| format!("reading perf.data at {perf_data}"))?;
    eprintln!(
        "  [timing] read_perf_data() total {:.2}s",
        t_read.elapsed().as_secs_f64()
    );

    if raw.samples.is_empty() {
        return Err(anyhow::anyhow!("No samples parsed from perf data."));
    }

    let mut frames = FrameTable::default();
    let mut samples: Vec<Sample> = Vec::with_capacity(raw.samples.len());
    let mut skipped_no_cat: u64 = 0;

    // Phase A: walk every sample and collect the set of distinct
    // `(pid, ip)` pairs we'll need a symbol for. The actual sample
    // pile has tens of millions of stack frames, but distinct IPs are
    // typically a few thousand — interning lookups dominate everything
    // else.
    let t_collect = Instant::now();
    let mut needed: AHashSet<(i32, u64)> = AHashSet::new();
    for s in &raw.samples {
        needed.insert((s.pid, s.ip));
        for &ip in &s.callchain {
            if is_perf_context(ip) || ip == s.ip {
                continue;
            }
            needed.insert((s.pid, ip));
        }
    }
    let needed_vec: Vec<(i32, u64)> = needed.into_iter().collect();
    eprintln!(
        "  [timing] collect unique (pid,ip) pairs {:.2}s ({} pairs)",
        t_collect.elapsed().as_secs_f64(),
        needed_vec.len(),
    );

    // Phase B: resolve every unique pair to a symbol name in parallel.
    // `SymbolCache` is `Sync` (internal Mutex around the binary→map
    // table, plus lock-free `lookup_sync` on the loaded SymbolMap);
    // `AddressSpaces` is read-only here.
    let t_resolve = Instant::now();
    let resolved: Vec<String> = needed_vec
        .par_iter()
        .map(|&(pid, ip)| resolve_to_name(&raw.address_spaces, &raw.symbol_cache, pid, ip))
        .collect();
    eprintln!(
        "  [timing] parallel symbol resolve {:.2}s",
        t_resolve.elapsed().as_secs_f64()
    );

    // Phase C: sequentially intern names into the FrameTable, building
    // the `(pid, ip) → frame_id` lookup we'll use to assemble stacks.
    let t_intern = Instant::now();
    let mut ip_to_frame: AHashMap<(i32, u64), u32> =
        AHashMap::with_capacity(needed_vec.len());
    for (pair, name) in needed_vec.iter().zip(resolved.iter()) {
        let id = frames.intern(name);
        ip_to_frame.insert(*pair, id);
    }
    let unknown_id = frames.intern(UNKNOWN_FRAME);
    eprintln!(
        "  [timing] intern frame names {:.2}s ({} unique frames)",
        t_intern.elapsed().as_secs_f64(),
        frames.len(),
    );

    // Phase D: build the final sample list using the lookup table. No
    // more per-frame symbol work — just hashmap lookups + stack
    // assembly, so this pass is memory-bound.
    let t_build = Instant::now();
    for s in &raw.samples {
        let class = raw
            .events
            .get(s.event_idx as usize)
            .map(|e| e.class)
            .unwrap_or(EventClass::Other);
        let Some(category) = category_for(s, class) else {
            skipped_no_cat += 1;
            continue;
        };

        let mut stack: Vec<u32> = Vec::with_capacity(1 + s.callchain.len());
        // Leaf first (matches perf's leaf-first callchain convention).
        stack.push(*ip_to_frame.get(&(s.pid, s.ip)).unwrap_or(&unknown_id));
        // Callchain entries after leaf. linux-perf-data's callchain begins
        // with PERF_CONTEXT markers (huge sentinel values like
        // 0xfffffffffffffffe = USER, 0xfffffffffffffff8 = KERNEL, …) — drop
        // those as they aren't real IPs.
        for &ip in &s.callchain {
            if is_perf_context(ip) || ip == s.ip {
                continue;
            }
            stack.push(*ip_to_frame.get(&(s.pid, ip)).unwrap_or(&unknown_id));
        }

        samples.push(Sample {
            time_ns: s.time_ns,
            cpu: s.cpu,
            category,
            weight: weight_for(s.period),
            stack,
        });
    }
    eprintln!(
        "  [timing] assemble sample stacks {:.2}s",
        t_build.elapsed().as_secs_f64()
    );

    if samples.is_empty() {
        return Err(anyhow::anyhow!(
            "Parsed {} blocks but none classified into a UI category. Re-record with `perf record -e cycles -e ibs_op//` for full coverage.",
            raw.samples.len(),
        ));
    }

    let t_sort = Instant::now();
    samples.sort_by_key(|s| s.time_ns);
    eprintln!(
        "  [timing] perfy sort {:.2}s",
        t_sort.elapsed().as_secs_f64()
    );
    let t_axes = Instant::now();
    let (cpus, categories) = collect_axes(&samples);
    eprintln!(
        "  [timing] collect_axes {:.2}s",
        t_axes.elapsed().as_secs_f64()
    );

    let binary_path = raw.primary_binary();
    let perf_data_path = std::path::Path::new(perf_data)
        .canonicalize()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| perf_data.to_string());

    let stat_data_path = crate::stat::locate_stat_data(Path::new(&perf_data_path))
        .map(|p| p.to_string_lossy().into_owned());
    if let Some(p) = stat_data_path.as_deref() {
        eprintln!("  stat: found {p}");
    } else {
        eprintln!(
            "  stat: no perf.stat.data sibling found — memory tracks will be empty. \
             Re-record with `perf stat record -o perf.stat.data … -- <perf record cmd>` \
             to populate them."
        );
    }
    if raw.clock_anchor.is_none() {
        eprintln!(
            "  WARN: perf.data has no HEADER_CLOCK_DATA — memory tracks will not be \
             aligned with the timeline. Re-record `perf record` with `--clockid mono` \
             so the alignment is wall-clock-precise."
        );
    }

    eprintln!(
        "  built {} timeline samples from {} raw samples (skipped {skipped_no_cat}); {} cpus; categories={:?}",
        samples.len(),
        raw.samples.len(),
        cpus.len(),
        categories.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
    );

    let time_start_ns = samples.first().unwrap().time_ns;
    let time_end_ns = samples.last().unwrap().time_ns;

    Ok(Profile {
        samples,
        frames,
        cpus,
        categories,
        time_start_ns,
        time_end_ns,
        clock_anchor: raw.clock_anchor,
        insn_stats: raw.insn_stats,
        ip_to_key: raw.ip_to_key,
        binary_path,
        perf_data_path,
        stat_data_path,
        memory_base_latency: 0.0,
        address_spaces: Arc::new(raw.address_spaces),
        symbol_cache: Arc::new(raw.symbol_cache),
    })
}

/// Resolve `(pid, ip)` to a symbol name string. Called once per
/// distinct pair (the parser collects the unique set first and runs
/// this in parallel via rayon); the actual `FrameTable::intern` step
/// happens sequentially afterwards so the table doesn't need internal
/// synchronisation.
fn resolve_to_name(
    spaces: &AddressSpaces,
    cache: &SymbolCache,
    pid: i32,
    ip: u64,
) -> String {
    spaces
        .lookup(pid, ip)
        .and_then(|m| {
            let base = spaces
                .binary_base(pid, &m.binary)
                .unwrap_or(m.addr_lo.saturating_sub(m.page_offset));
            cache.resolve_with_base(m, ip, base).map(|(s, _)| s)
        })
        .unwrap_or_else(|| UNKNOWN_FRAME.to_string())
}

#[inline]
fn is_perf_context(ip: u64) -> bool {
    // PERF_CONTEXT_* values from include/uapi/linux/perf_event.h —
    // they're all in the top of the u64 range. Conservatively treat
    // anything within the top 4096 of u64 as a sentinel.
    ip > !0u64 - 4096
}

// Used only by AHashSet imports above (avoid unused-imports warnings while
// the trait is used implicitly).
#[allow(dead_code)]
fn _silence_imports(_: AHashSet<u32>) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(period: u64, cache: Option<CacheLevel>) -> ParsedSample {
        ParsedSample {
            time_ns: 0,
            cpu: 0,
            pid: 0,
            tid: 0,
            ip: 0,
            callchain: vec![],
            event_idx: 0,
            cache,
            period,
        }
    }

    /// Cycles events always map to Category::Cycles regardless of cache
    /// (cache info is meaningless for them).
    #[test]
    fn cycles_event_is_always_cycles_category() {
        assert_eq!(
            category_for(&parsed(0, None), EventClass::Cycles),
            Some(Category::Cycles)
        );
        assert_eq!(
            category_for(&parsed(0, Some(CacheLevel::L3)), EventClass::Cycles),
            Some(Category::Cycles)
        );
    }

    /// IBS ops route into the cache-tier categories. LFB rolls up into
    /// L1; REMOTE rolls up into Dram. NonMemory and missing cache info
    /// yield None (the sample is dropped from the timeline).
    #[test]
    fn ibs_op_routes_by_cache_level() {
        let cases = [
            (CacheLevel::L1, Category::L1),
            (CacheLevel::LFB, Category::L1),
            (CacheLevel::L2, Category::L2),
            (CacheLevel::L3, Category::L3),
            (CacheLevel::DRAM, Category::Dram),
            (CacheLevel::REMOTE, Category::Dram),
        ];
        for (cl, want) in cases {
            assert_eq!(
                category_for(&parsed(0, Some(cl)), EventClass::IbsOp),
                Some(want),
                "ibs_op with cache={cl:?}",
            );
        }
        assert_eq!(
            category_for(&parsed(0, Some(CacheLevel::NonMemory)), EventClass::IbsOp),
            None
        );
        assert_eq!(category_for(&parsed(0, None), EventClass::IbsOp), None);
    }

    /// The sample's perf-event period becomes its bucketing weight,
    /// with a defensive floor of 1 so a period=0 sample still shows up.
    #[test]
    fn weight_for_uses_period_with_floor_one() {
        assert_eq!(weight_for(0), 1, "missing period defaults to 1");
        assert_eq!(weight_for(1), 1);
        assert_eq!(weight_for(4_000_000), 4_000_000);
        // u64-sized periods are preserved (cycles can be huge over long runs).
        assert_eq!(weight_for(u64::MAX), u64::MAX);
    }

    /// Other event classes (timer dummies, software events we don't
    /// model) are filtered out — they shouldn't show up on the
    /// timeline.
    #[test]
    fn other_event_class_is_dropped() {
        assert_eq!(category_for(&parsed(0, None), EventClass::Other), None);
        assert_eq!(
            category_for(&parsed(1234, Some(CacheLevel::L1)), EventClass::Other),
            None
        );
    }
}
