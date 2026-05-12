//! In-memory profile model: per-sample records with interned call stacks.
use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use serde::Serialize;

use ibs_annotate::reader::ClockAnchor;
use ibs_annotate::{AddressSpaces, InsnStats, SymbolCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    Cycles,
    L1,
    L2,
    L3,
    Dram,
}

impl Category {
    pub const ALL: [Category; 5] = [
        Category::Cycles,
        Category::L1,
        Category::L2,
        Category::L3,
        Category::Dram,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Category::Cycles => "cycles",
            Category::L1 => "l1",
            Category::L2 => "l2",
            Category::L3 => "l3",
            Category::Dram => "dram",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Category::Cycles => "Cycles",
            Category::L1 => "L1",
            Category::L2 => "L2",
            Category::L3 => "L3",
            Category::Dram => "DRAM",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cycles" => Some(Category::Cycles),
            "l1" => Some(Category::L1),
            "l2" => Some(Category::L2),
            "l3" => Some(Category::L3),
            "dram" => Some(Category::Dram),
            _ => None,
        }
    }
}

/// One sample point on the timeline, with its full call stack as interned ids.
#[derive(Debug, Clone)]
pub struct Sample {
    pub time_ns: u64,
    pub cpu: u32,
    pub category: Category,
    /// Cycles represented by this sample (the perf event's `period`).
    /// Used to weight timeline intensity so the y-axis means "cycles
    /// in this bin" rather than "samples in this bin" — they only
    /// agree when the sample period is constant (e.g. `perf record -c
    /// <N>`); under `-F freq` the period adapts and counts no longer
    /// match cycle density. Falls back to 1 if perf didn't emit a
    /// period.
    pub weight: u64,
    /// Leaf-first stack: index 0 is the executing frame.
    pub stack: Vec<u32>,
}

/// Intern call-stack frame names → small u32 ids.
#[derive(Default, Debug)]
pub struct FrameTable {
    names: Vec<String>,
    index: AHashMap<String, u32>,
}

impl FrameTable {
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.index.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.index.insert(name.to_string(), id);
        id
    }

    pub fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[derive(Debug)]
pub struct Profile {
    pub samples: Vec<Sample>,
    pub frames: FrameTable,
    pub cpus: Vec<u32>,
    pub categories: Vec<Category>,
    pub time_start_ns: u64,
    pub time_end_ns: u64,
    /// Wall-clock anchor recorded by perf at session-open time (the
    /// HEADER_CLOCK_DATA feature, perf >= 5.7). When present, lets us
    /// convert any sample's CLOCK_MONOTONIC timestamp to a real wall
    /// clock — and so align the perf-stat CSV (whose timestamps are
    /// seconds-since-process-start) to the same axis.
    pub clock_anchor: Option<ClockAnchor>,
    /// Per-instruction stats for the source/asm view (compatible with
    /// `ibs_annotate`).
    pub insn_stats: AHashMap<String, InsnStats>,
    pub ip_to_key: AHashMap<u64, String>,
    pub binary_path: Option<String>,
    pub perf_data_path: String,
    /// Path to a `perf.stat.data` produced by `perf stat record -o
    /// perf.stat.data` alongside the perf.data file. `None` when the
    /// recording was taken without `perf stat record`. Used by the
    /// `/api/stat` endpoint to build memory-throughput /
    /// memory-latency tracks.
    pub stat_data_path: Option<String>,
    /// Idle/baseline memory latency in core clocks. Subtracted from
    /// the latency series before bucketing so the graph shows
    /// deviation from the floor rather than absolute values. 0 means
    /// no baseline (graph = absolute). Set via `--memory-base-latency`
    /// on the `serve` command.
    pub memory_base_latency: f64,
    /// Per-pid mmap2 layout, shared with the annotation path so it can map
    /// any IP back to the binary that owns it.
    pub address_spaces: Arc<AddressSpaces>,
    /// ELF symbol-table cache, also shared with the annotation path.
    #[allow(dead_code)]
    pub symbol_cache: Arc<SymbolCache>,
}

impl Profile {
    pub fn duration_ns(&self) -> u64 {
        self.time_end_ns.saturating_sub(self.time_start_ns)
    }

    pub fn stats_summary(&self) -> StatsSummary {
        let mut by_cat: AHashMap<Category, u64> = AHashMap::new();
        let mut by_cpu_cat: AHashMap<u32, AHashMap<Category, u64>> = AHashMap::new();
        for s in &self.samples {
            *by_cat.entry(s.category).or_default() += 1;
            *by_cpu_cat.entry(s.cpu).or_default().entry(s.category).or_default() += 1;
        }
        StatsSummary {
            total_samples: self.samples.len() as u64,
            by_category: by_cat
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect(),
            by_cpu: by_cpu_cat
                .into_iter()
                .map(|(cpu, m)| {
                    (
                        cpu,
                        m.into_iter()
                            .map(|(k, v)| (k.as_str().to_string(), v))
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct StatsSummary {
    pub total_samples: u64,
    pub by_category: AHashMap<String, u64>,
    pub by_cpu: AHashMap<u32, AHashMap<String, u64>>,
}

/// Helper: collect the ordered set of CPUs/categories that actually have
/// samples (stable order).
pub fn collect_axes(samples: &[Sample]) -> (Vec<u32>, Vec<Category>) {
    let mut cpus: AHashSet<u32> = AHashSet::new();
    let mut cats: AHashSet<Category> = AHashSet::new();
    for s in samples {
        cpus.insert(s.cpu);
        cats.insert(s.category);
    }
    let mut cpu_vec: Vec<u32> = cpus.into_iter().collect();
    cpu_vec.sort_unstable();
    let cat_vec: Vec<Category> = Category::ALL
        .iter()
        .copied()
        .filter(|c| cats.contains(c))
        .collect();
    (cpu_vec, cat_vec)
}
