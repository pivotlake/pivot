//! Time-bucketed sample density per (category, [cpu]) track.
use rayon::prelude::*;
use serde::Serialize;

use crate::profile::{Category, Profile, Sample};

pub const DEFAULT_BUCKETS: usize = 1500;

#[derive(Debug, Clone, Serialize)]
pub struct Track {
    pub id: String,
    pub label: String,
    pub category: Category,
    pub cpu: Option<u32>,
    /// Per-bucket weight sum. Each sample contributes its perf-event
    /// `period` (cycles between samples), so this is "cycles in bucket"
    /// rather than "samples in bucket". u64 because a busy CPU at 3 GHz
    /// over a few seconds easily exceeds u32 (~4.3 G).
    pub counts: Vec<u64>,
    pub peak: u64,
}

#[inline]
fn bucket_of(time_ns: u64, t0: u64, slot_ns: u64, buckets: usize) -> usize {
    if slot_ns == 0 {
        return 0;
    }
    let idx = (time_ns.saturating_sub(t0) / slot_ns) as usize;
    if idx >= buckets { buckets - 1 } else { idx }
}

pub fn build_tracks(
    profile: &Profile,
    categories: &[Category],
    consolidated: bool,
    buckets: usize,
) -> Vec<Track> {
    build_tracks_inner(
        &profile.samples,
        &profile.cpus,
        profile.time_start_ns,
        profile.duration_ns().max(1),
        categories,
        consolidated,
        buckets,
    )
}

/// Pure-data version of [`build_tracks`] — takes the raw inputs the
/// algorithm actually depends on, so it can be exercised by unit tests
/// without building a whole `Profile` (which owns a thread-spawning
/// `SymbolCache` and a parsed perf.data file).
pub fn build_tracks_inner(
    samples: &[Sample],
    cpus: &[u32],
    t0: u64,
    total_ns: u64,
    categories: &[Category],
    consolidated: bool,
    buckets: usize,
) -> Vec<Track> {
    let buckets = if buckets == 0 { DEFAULT_BUCKETS } else { buckets };
    let total = total_ns.max(1);
    let slot_ns = (total / buckets as u64).max(1);

    let cat_set: ahash::AHashSet<Category> = categories.iter().copied().collect();

    if consolidated {
        // One track per category. For large sample sets, building the
        // count vectors in parallel by partitioning samples is overkill —
        // the bucket update itself is O(1) per sample. We do the simple
        // sequential pass into per-category vectors. Each sample
        // contributes its period (= cycles), not 1.
        let mut per_cat: ahash::AHashMap<Category, Vec<u64>> = ahash::AHashMap::new();
        for &c in categories {
            per_cat.insert(c, vec![0; buckets]);
        }
        for s in samples {
            if !cat_set.contains(&s.category) {
                continue;
            }
            let i = bucket_of(s.time_ns, t0, slot_ns, buckets);
            if let Some(v) = per_cat.get_mut(&s.category) {
                v[i] = v[i].saturating_add(s.weight);
            }
        }
        return categories
            .iter()
            .map(|&c| {
                let counts = per_cat.remove(&c).unwrap_or_else(|| vec![0; buckets]);
                let peak = counts.par_iter().copied().max().unwrap_or(0);
                Track {
                    id: format!("cat:{}", c.as_str()),
                    label: c.label().to_string(),
                    category: c,
                    cpu: None,
                    counts,
                    peak,
                }
            })
            .collect();
    }

    // One track per (category, cpu). Allocate all rows up front, then fill.
    let mut rows: ahash::AHashMap<(Category, u32), Vec<u64>> = ahash::AHashMap::new();
    for &c in categories {
        for &cpu in cpus {
            rows.insert((c, cpu), vec![0; buckets]);
        }
    }
    for s in samples {
        if !cat_set.contains(&s.category) {
            continue;
        }
        if let Some(v) = rows.get_mut(&(s.category, s.cpu)) {
            let i = bucket_of(s.time_ns, t0, slot_ns, buckets);
            v[i] = v[i].saturating_add(s.weight);
        }
    }

    let mut out: Vec<Track> = Vec::with_capacity(categories.len() * cpus.len());
    for &cat in categories {
        for &cpu in cpus {
            let counts = rows.remove(&(cat, cpu)).unwrap_or_else(|| vec![0; buckets]);
            let peak = counts.par_iter().copied().max().unwrap_or(0);
            out.push(Track {
                id: format!("cat:{}:cpu:{}", cat.as_str(), cpu),
                label: format!("{} · CPU {}", cat.label(), cpu),
                category: cat,
                cpu: Some(cpu),
                counts,
                peak,
            });
        }
    }
    out
}

/// Reverse of [`Track::id`] — used by `/api/flamegraph?track=…`.
pub fn parse_track_id(track_id: &str) -> anyhow::Result<(Category, Option<u32>)> {
    let parts: Vec<&str> = track_id.split(':').collect();
    if parts.len() >= 2 && parts[0] == "cat" {
        let cat = Category::parse(parts[1])
            .ok_or_else(|| anyhow::anyhow!("unrecognised category in track id: {track_id}"))?;
        let cpu = if parts.len() >= 4 && parts[2] == "cpu" {
            Some(parts[3].parse::<u32>()?)
        } else {
            None
        };
        return Ok((cat, cpu));
    }
    Err(anyhow::anyhow!("unrecognised track id: {track_id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(time_ns: u64, cpu: u32, category: Category, weight: u64) -> Sample {
        Sample {
            time_ns,
            cpu,
            category,
            weight,
            stack: vec![],
        }
    }

    /// Sanity: bucket_of maps time_ns → [0, buckets).
    #[test]
    fn bucket_of_clamps_at_end() {
        let buckets = 10;
        // total span 1000ns; slot=100ns. time=999 lands in bucket 9.
        assert_eq!(bucket_of(0, 0, 100, buckets), 0);
        assert_eq!(bucket_of(99, 0, 100, buckets), 0);
        assert_eq!(bucket_of(100, 0, 100, buckets), 1);
        assert_eq!(bucket_of(999, 0, 100, buckets), 9);
        // Anything past the end clamps to last bucket (defensive).
        assert_eq!(bucket_of(99_999, 0, 100, buckets), 9);
    }

    /// Each sample contributes its weight (period), not 1. This is the
    /// behaviour change that turns the y-axis from "samples in this
    /// bin" into "cycles in this bin".
    #[test]
    fn consolidated_sums_period_not_count() {
        let cpus = vec![0];
        let samples = vec![
            s(50, 0, Category::Cycles, 1000),
            s(60, 0, Category::Cycles, 2000),
            s(550, 0, Category::Cycles, 4000),
        ];
        let tracks = build_tracks_inner(
            &samples,
            &cpus,
            0,
            1000,
            &[Category::Cycles],
            true,
            10,
        );
        assert_eq!(tracks.len(), 1);
        let t = &tracks[0];
        assert_eq!(t.id, "cat:cycles");
        assert_eq!(t.cpu, None);
        // Bin 0 (time 0-99): 1000+2000 = 3000.
        // Bin 5 (time 500-599): 4000.
        assert_eq!(t.counts[0], 3000);
        assert_eq!(t.counts[5], 4000);
        assert_eq!(t.peak, 4000);
        // All other bins zero.
        for (i, &c) in t.counts.iter().enumerate() {
            if i != 0 && i != 5 {
                assert_eq!(c, 0, "bin {i} should be empty");
            }
        }
    }

    /// Per-CPU mode emits one track per (category, cpu) pair, and
    /// samples on cpu0 don't bleed into cpu1's bucket.
    #[test]
    fn per_cpu_splits_by_cpu_and_keeps_period() {
        let cpus = vec![0, 1];
        let samples = vec![
            s(10, 0, Category::Cycles, 100),
            s(20, 0, Category::Cycles, 100),
            s(30, 1, Category::Cycles, 500),
        ];
        let tracks = build_tracks_inner(
            &samples,
            &cpus,
            0,
            100,
            &[Category::Cycles],
            false,
            10,
        );
        assert_eq!(tracks.len(), 2);
        let cpu0 = tracks.iter().find(|t| t.cpu == Some(0)).unwrap();
        let cpu1 = tracks.iter().find(|t| t.cpu == Some(1)).unwrap();
        // CPU0 bin 1 (10-19) gets one sample, bin 2 (20-29) gets one.
        assert_eq!(cpu0.counts[1], 100);
        assert_eq!(cpu0.counts[2], 100);
        assert_eq!(cpu0.peak, 100);
        // CPU1 bin 3 (30-39) gets one sample.
        assert_eq!(cpu1.counts[3], 500);
        assert_eq!(cpu1.peak, 500);
        // No leakage.
        assert_eq!(cpu0.counts[3], 0);
        assert_eq!(cpu1.counts[1], 0);
    }

    /// A category that the user didn't request must not appear and
    /// must not pollute the buckets of categories that were requested.
    #[test]
    fn unrequested_category_is_filtered() {
        let cpus = vec![0];
        let samples = vec![
            s(0, 0, Category::Cycles, 1),
            s(0, 0, Category::L1, 9999),
        ];
        let tracks = build_tracks_inner(
            &samples,
            &cpus,
            0,
            10,
            &[Category::Cycles],
            true,
            5,
        );
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].category, Category::Cycles);
        assert_eq!(tracks[0].counts.iter().sum::<u64>(), 1);
    }

    /// `parse_track_id` is the inverse of `Track::id` for both
    /// consolidated (`cat:<name>`) and per-cpu (`cat:<name>:cpu:<n>`).
    #[test]
    fn parse_track_id_round_trip() {
        assert_eq!(
            parse_track_id("cat:cycles").unwrap(),
            (Category::Cycles, None)
        );
        assert_eq!(
            parse_track_id("cat:dram:cpu:7").unwrap(),
            (Category::Dram, Some(7))
        );
        assert!(parse_track_id("garbage").is_err());
        assert!(parse_track_id("cat:nope").is_err());
    }

    /// Regression: a single sample whose period exceeds u32::MAX
    /// (possible on a long-running event with adaptive frequency)
    /// must not overflow the bucket.
    #[test]
    fn weight_does_not_overflow_u32_boundary() {
        let cpus = vec![0];
        let huge = (u32::MAX as u64) + 1234;
        let samples = vec![s(0, 0, Category::Cycles, huge)];
        let tracks = build_tracks_inner(
            &samples,
            &cpus,
            0,
            10,
            &[Category::Cycles],
            true,
            5,
        );
        assert_eq!(tracks[0].counts[0], huge);
        assert_eq!(tracks[0].peak, huge);
    }
}
