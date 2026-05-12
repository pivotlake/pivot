//! Data model — mirrors the Python `ibs_annotate.model` module.
use ahash::AHashMap;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CacheLevel {
    L1,
    LFB,
    L2,
    L3,
    DRAM,
    REMOTE,
    NonMemory,
}

impl CacheLevel {
    pub const ALL: [CacheLevel; 7] = [
        CacheLevel::L1,
        CacheLevel::LFB,
        CacheLevel::L2,
        CacheLevel::L3,
        CacheLevel::DRAM,
        CacheLevel::REMOTE,
        CacheLevel::NonMemory,
    ];

    pub fn label(self) -> &'static str {
        match self {
            CacheLevel::L1 => "L1",
            CacheLevel::LFB => "LFB",
            CacheLevel::L2 => "L2",
            CacheLevel::L3 => "L3",
            CacheLevel::DRAM => "DRAM",
            CacheLevel::REMOTE => "REM",
            CacheLevel::NonMemory => "N-M",
        }
    }

    /// Approximate cost in cycles (Zen 4) — same numbers as the Python.
    pub fn weight(self) -> u64 {
        match self {
            CacheLevel::L1 => 4,
            CacheLevel::LFB => 9,
            CacheLevel::L2 => 14,
            CacheLevel::L3 => 50,
            CacheLevel::DRAM => 250,
            CacheLevel::REMOTE => 400,
            CacheLevel::NonMemory => 1,
        }
    }

    /// curses color pair index per cache level (kept for parity with the TUI)
    pub fn color_pair(self) -> u8 {
        match self {
            CacheLevel::L1 | CacheLevel::LFB => 1,        // green
            CacheLevel::L2 | CacheLevel::L3 => 2,         // yellow
            CacheLevel::DRAM => 3,                        // red
            CacheLevel::REMOTE => 4,                      // magenta
            CacheLevel::NonMemory => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum TlbLevel {
    L1Hit,
    L2Hit,
    Miss,
    NA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum OpType {
    Load,
    Store,
    NA,
}

impl OpType {
    pub fn label(self) -> &'static str {
        match self {
            OpType::Load => "LOAD",
            OpType::Store => "STORE",
            OpType::NA => "N/A",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum SnoopStatus {
    NA,
    None,
    Hit,
    HitM,
    Miss,
}

impl SnoopStatus {
    pub fn label(self) -> &'static str {
        match self {
            SnoopStatus::NA => "N/A",
            SnoopStatus::None => "None",
            SnoopStatus::Hit => "Hit",
            SnoopStatus::HitM => "HitM",
            SnoopStatus::Miss => "Miss",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DisplayMode {
    Weighted,
    Percent,
    Absolute,
}

impl DisplayMode {
    pub fn label(self) -> &'static str {
        match self {
            DisplayMode::Weighted => "weighted %",
            DisplayMode::Percent => "%",
            DisplayMode::Absolute => "count",
        }
    }
}

pub const COL_WIDTH: usize = 5;

/// Aggregated raw IBS register fields for a single instruction.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IBSRaw {
    pub dc_miss_lat_sum: u64,
    pub dc_miss_lat_count: u64,
    pub tlb_refill_lat_sum: u64,
    pub tlb_refill_lat_count: u64,
    pub comp_to_ret_sum: u64,
    pub comp_to_ret_count: u64,
    pub tag_to_ret_sum: u64,
    pub tag_to_ret_count: u64,
    pub mabs_sum: u64,
    pub mabs_count: u64,
    pub mabs_max: u64,
    pub mabs_max_count: u64,
    pub sw_pf_count: u64,
    pub misaligned_count: u64,
    pub dc_miss_no_mab_count: u64,
    pub dc_l1_tlb_miss_count: u64,
    pub dc_l2_tlb_miss_count: u64,
    pub dc_miss_count: u64,
    pub l2_miss_count: u64,
    pub mem_op_count: u64,
    pub mem_width_counts: AHashMap<u64, u64>,
    pub brn_ret_count: u64,
    pub brn_misp_count: u64,
    pub brn_taken_count: u64,
    pub brn_return_count: u64,
    pub brn_fuse_count: u64,
    pub sample_count: u64,
}

impl IBSRaw {
    pub fn avg(total: u64, count: u64) -> f64 {
        if count == 0 { 0.0 } else { total as f64 / count as f64 }
    }
    pub fn avg_dc_miss_lat(&self) -> f64 { Self::avg(self.dc_miss_lat_sum, self.dc_miss_lat_count) }
    pub fn avg_tlb_refill_lat(&self) -> f64 { Self::avg(self.tlb_refill_lat_sum, self.tlb_refill_lat_count) }
    pub fn avg_comp_to_ret(&self) -> f64 { Self::avg(self.comp_to_ret_sum, self.comp_to_ret_count) }
    pub fn avg_tag_to_ret(&self) -> f64 { Self::avg(self.tag_to_ret_sum, self.tag_to_ret_count) }
    pub fn avg_mabs(&self) -> f64 { Self::avg(self.mabs_sum, self.mabs_count) }

    /// Merge another IBSRaw into self (for parallel parsing reductions).
    pub fn merge(&mut self, other: &IBSRaw) {
        self.dc_miss_lat_sum += other.dc_miss_lat_sum;
        self.dc_miss_lat_count += other.dc_miss_lat_count;
        self.tlb_refill_lat_sum += other.tlb_refill_lat_sum;
        self.tlb_refill_lat_count += other.tlb_refill_lat_count;
        self.comp_to_ret_sum += other.comp_to_ret_sum;
        self.comp_to_ret_count += other.comp_to_ret_count;
        self.tag_to_ret_sum += other.tag_to_ret_sum;
        self.tag_to_ret_count += other.tag_to_ret_count;
        self.mabs_sum += other.mabs_sum;
        self.mabs_count += other.mabs_count;
        if other.mabs_max > self.mabs_max {
            self.mabs_max = other.mabs_max;
            self.mabs_max_count = other.mabs_max_count;
        } else if other.mabs_max == self.mabs_max {
            self.mabs_max_count += other.mabs_max_count;
        }
        self.sw_pf_count += other.sw_pf_count;
        self.misaligned_count += other.misaligned_count;
        self.dc_miss_no_mab_count += other.dc_miss_no_mab_count;
        self.dc_l1_tlb_miss_count += other.dc_l1_tlb_miss_count;
        self.dc_l2_tlb_miss_count += other.dc_l2_tlb_miss_count;
        self.dc_miss_count += other.dc_miss_count;
        self.l2_miss_count += other.l2_miss_count;
        self.mem_op_count += other.mem_op_count;
        for (k, v) in &other.mem_width_counts {
            *self.mem_width_counts.entry(*k).or_default() += v;
        }
        self.brn_ret_count += other.brn_ret_count;
        self.brn_misp_count += other.brn_misp_count;
        self.brn_taken_count += other.brn_taken_count;
        self.brn_return_count += other.brn_return_count;
        self.brn_fuse_count += other.brn_fuse_count;
        self.sample_count += other.sample_count;
    }
}

/// Per-instruction SW prefetch PMC counts (PMCx04B, PMCx052, PMCx059).
#[derive(Debug, Clone, Default, Serialize)]
pub struct PrefetchCounts {
    pub dispatched: u64,
    pub ineffective_dc_hit: u64,
    pub ineffective_mab_match: u64,
    pub fills: u64,
}

impl PrefetchCounts {
    pub fn total_ineffective(&self) -> u64 {
        self.ineffective_dc_hit + self.ineffective_mab_match
    }
    pub fn effective_rate(&self) -> f64 {
        if self.dispatched == 0 { 0.0 } else { self.fills as f64 / self.dispatched as f64 }
    }
    pub fn merge(&mut self, other: &PrefetchCounts) {
        self.dispatched += other.dispatched;
        self.ineffective_dc_hit += other.ineffective_dc_hit;
        self.ineffective_mab_match += other.ineffective_mab_match;
        self.fills += other.fills;
    }
}

/// Aggregated IBS statistics for a single instruction address.
#[derive(Debug, Clone, Default, Serialize)]
pub struct InsnStats {
    pub cache_counts: AHashMap<CacheLevel, u64>,
    pub tlb_counts: AHashMap<TlbLevel, u64>,
    pub op_counts: AHashMap<OpType, u64>,
    pub snoop_counts: AHashMap<SnoopStatus, u64>,
    pub locked_count: u64,
    pub total_weight: u64,
    pub weight_count: u64,
    pub total_samples: u64,
    pub cycles: u64,
    pub ibs: IBSRaw,
    pub prefetch: PrefetchCounts,
    /// Per-instruction PMC totals for `de_dis_dispatch_token_stalls{1,2}.*`.
    pub token_stalls: AHashMap<String, u64>,
}

impl InsnStats {
    pub fn add(
        &mut self,
        cache: CacheLevel,
        tlb: TlbLevel,
        op: OpType,
        snoop: SnoopStatus,
        locked: bool,
        weight: u64,
    ) {
        self.total_samples += 1;
        *self.cache_counts.entry(cache).or_default() += 1;
        *self.tlb_counts.entry(tlb).or_default() += 1;
        *self.op_counts.entry(op).or_default() += 1;
        *self.snoop_counts.entry(snoop).or_default() += 1;
        if locked {
            self.locked_count += 1;
        }
        if weight > 0 {
            self.total_weight += weight;
            self.weight_count += 1;
        }
    }

    pub fn avg_weight(&self) -> f64 {
        if self.weight_count == 0 { 0.0 } else { self.total_weight as f64 / self.weight_count as f64 }
    }

    /// Merge `other` into self (used to reduce parallel partial maps).
    pub fn merge(&mut self, other: &InsnStats) {
        for (k, v) in &other.cache_counts { *self.cache_counts.entry(*k).or_default() += v; }
        for (k, v) in &other.tlb_counts { *self.tlb_counts.entry(*k).or_default() += v; }
        for (k, v) in &other.op_counts { *self.op_counts.entry(*k).or_default() += v; }
        for (k, v) in &other.snoop_counts { *self.snoop_counts.entry(*k).or_default() += v; }
        self.locked_count += other.locked_count;
        self.total_weight += other.total_weight;
        self.weight_count += other.weight_count;
        self.total_samples += other.total_samples;
        self.cycles += other.cycles;
        self.ibs.merge(&other.ibs);
        self.prefetch.merge(&other.prefetch);
        for (k, v) in &other.token_stalls {
            *self.token_stalls.entry(k.clone()).or_default() += v;
        }
    }
}

/// Per-function aggregate for the function picker.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FunctionSummary {
    pub name: String,
    pub total_samples: u64,
    pub weighted_cost: u64,
    pub cycles: u64,
    pub cache_counts: AHashMap<CacheLevel, u64>,
}

// -- Annotated line types -----------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct SourceLine {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionHeader {
    pub name: String,
    pub base_addr: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstructionLine {
    pub addr: String,
    pub offset: u64,
    pub disasm: String,
    pub sym: Option<String>,
    pub stats: InsnStats,
    /// Source file path declared by the `/path/file.rs:LINE` marker that
    /// preceded this instruction in the objdump stream, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_line: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SeparatorLine;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AnnotatedLine {
    Source(SourceLine),
    Function(FunctionHeader),
    Instruction(InstructionLine),
    Separator(SeparatorLine),
}

// -- Jump graph (annotated-view gutter) --------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct JumpArrow {
    pub src_idx: usize,
    /// Target line index; -1 → outside this function.
    pub tgt_idx: i64,
    pub forward: bool,
    pub lane: u32,
    pub is_short: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct JumpGraph {
    pub arrows: Vec<JumpArrow>,
    pub targets: ahash::AHashSet<usize>,
    pub max_lanes: u32,
    pub addr_width: usize,
}

// -- Skipped samples bookkeeping ---------------------------------------------

#[derive(Debug, Clone, Default, Serialize)]
pub struct SkippedSamples {
    pub unknown_symbol: AHashMap<CacheLevel, u64>,
    pub unknown_level: u64,
}

impl SkippedSamples {
    pub fn merge(&mut self, other: &SkippedSamples) {
        for (k, v) in &other.unknown_symbol {
            *self.unknown_symbol.entry(*k).or_default() += v;
        }
        self.unknown_level += other.unknown_level;
    }

    pub fn summary_lines(&self) -> Vec<String> {
        let total_sym: u64 = self.unknown_symbol.values().sum();
        if total_sym == 0 && self.unknown_level == 0 {
            return vec![];
        }
        let mut out = vec!["Skipped samples:".to_string()];
        if total_sym > 0 {
            out.push(format!("  Unresolved symbol: {total_sym}"));
            for lvl in CacheLevel::ALL {
                if let Some(c) = self.unknown_symbol.get(&lvl) {
                    if *c > 0 {
                        out.push(format!("    {:>4}: {}", lvl.label(), c));
                    }
                }
            }
        }
        if self.unknown_level > 0 {
            out.push(format!("  Unknown cache level: {}", self.unknown_level));
        }
        out
    }
}
