//! Source + assembly annotation for a selected function.
//!
//! Disassembly comes from `objdump`; per-instruction stats come from the
//! `Profile`'s `insn_stats` map (populated by `ibs_annotate::reader` from the
//! `perf.data` binary, including IBS data_src + raw MSR fields). The
//! ASLR/PIE load-base is derived from the perf-data MMAP2 records (no
//! `perf script --show-mmap-events` invocation).

use std::sync::OnceLock;

use ahash::AHashMap;
use anyhow::{anyhow, Context};
use serde::Serialize;

use ibs_annotate::cmd;
use ibs_annotate::model::{AnnotatedLine, CacheLevel};
use ibs_annotate::parse as ibs_parse;

use crate::profile::Profile;

/// Computed once per (perf.data + binary): load_base + nm symbol size table.
struct AnnotatorCache {
    load_base: u64,
    func_bounds: AHashMap<u64, u64>,
    binary_path: String,
}

static CACHE: OnceLock<std::sync::Mutex<AHashMap<String, AnnotatorCache>>> = OnceLock::new();

fn cache() -> &'static std::sync::Mutex<AHashMap<String, AnnotatorCache>> {
    CACHE.get_or_init(|| std::sync::Mutex::new(AHashMap::new()))
}

fn ensure_cache(profile: &Profile, binary: &str) -> anyhow::Result<(u64, AHashMap<u64, u64>)> {
    {
        let guard = cache().lock().unwrap();
        if let Some(c) = guard.get(&profile.perf_data_path) {
            if c.binary_path == binary {
                return Ok((c.load_base, c.func_bounds.clone()));
            }
        }
    }
    // The "load base" we want is the runtime address where rva=0 of the
    // binary lives — derived from the first LOAD segment's mapping
    // (page_offset=0). `AddressSpaces::binary_base` does this across all
    // recorded pids and picks the consistent minimum, mirroring exactly the
    // value used by the live symbol resolver. No `readelf` or
    // executable-segment guessing required.
    let load_base = profile
        .address_spaces
        .all_mappings_iter()
        .filter(|m| m.binary == binary)
        .map(|m| m.addr_lo.saturating_sub(m.page_offset))
        .min()
        .ok_or_else(|| {
            anyhow!("no MMAP2 record observed for binary {binary:?}; can't compute load_base")
        })?;
    let func_bounds = ibs_parse::get_function_bounds(binary).context("get_function_bounds")?;
    let mut guard = cache().lock().unwrap();
    guard.insert(
        profile.perf_data_path.clone(),
        AnnotatorCache {
            load_base,
            func_bounds: func_bounds.clone(),
            binary_path: binary.to_string(),
        },
    );
    Ok((load_base, func_bounds))
}

/// Resolve a symbol name to a (file_addr, size) pair via the runtime ips
/// observed in the profile.
fn find_function_addr(
    profile: &Profile,
    sym: &str,
    load_base: u64,
    bounds: &AHashMap<u64, u64>,
) -> Option<(u64, u64)> {
    for (ip, key) in &profile.ip_to_key {
        let mut parts = key.splitn(2, '\0');
        let ksym = parts.next()?;
        let koff_s = parts.next()?;
        if ksym != sym {
            continue;
        }
        let koff: u64 = koff_s.parse().ok()?;
        let runtime_base = ip.saturating_sub(koff);
        let file_addr = runtime_base.wrapping_sub(load_base);
        let size = bounds.get(&file_addr).copied().unwrap_or(0x10000);
        return Some((file_addr, size));
    }
    None
}

#[derive(Debug, Serialize)]
pub struct AnnotateLine {
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub addr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disasm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycles: Option<u64>,
    /// Raw cache-level sample counts ("L1", "L2", "L3", "DRAM"). The
    /// frontend turns these into percentages (relative-to-itself,
    /// weighted, etc.) per the user's mode selection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_counts: Option<AHashMap<&'static str, u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples: Option<u64>,
    /// Source line this asm row maps to. Pair with `source_file` since a
    /// single function can reference multiple files (inlined helpers,
    /// headers, generics in a separate `mod`, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_line: Option<u32>,
    /// Absolute path of the source file this asm row maps to. Lines up
    /// 1:1 with one of the entries in `source_panes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    /// If this is a `j*` instruction with an intra-function target offset,
    /// the offset (in bytes from the function start) the jump points to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jump_target_offset: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct SourcePaneLine {
    pub line: u32,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct SourcePane {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub lines: Vec<SourcePaneLine>,
}

#[derive(Debug, Serialize, Default)]
pub struct FunctionTotals {
    pub cycles: u64,
    pub samples: u64,
    pub cache_counts: AHashMap<&'static str, u64>,
}

#[derive(Debug, Serialize)]
pub struct AnnotateResponse {
    pub symbol: String,
    pub binary: Option<String>,
    pub totals: Totals,
    /// Sums for this function only — the frontend uses these as the
    /// denominators for "Relative %" and "Weighted" display modes.
    pub function_totals: FunctionTotals,
    pub lines: Vec<AnnotateLine>,
    /// One pane per source file referenced by the function. Sorted with
    /// the heaviest (most cycles) first so the default tab is the
    /// "primary" file. Empty if no debuginfo or no file is on disk.
    pub source_panes: Vec<SourcePane>,
    /// Targets of intra-function jumps — used by the frontend to emit
    /// `addr:` labels next to the targets.
    pub jump_targets: Vec<u64>,
}

#[derive(Debug, Serialize)]
pub struct Totals {
    pub samples: u64,
    pub weight: u64,
    pub cycles: u64,
}

pub fn annotate_function(profile: &Profile, sym: &str) -> anyhow::Result<AnnotateResponse> {
    let binary = profile
        .binary_path
        .as_deref()
        .ok_or_else(|| anyhow!("No binary detected for this perf.data; can't annotate."))?;
    let (load_base, bounds) = ensure_cache(profile, binary)?;
    let (file_addr, size) = find_function_addr(profile, sym, load_base, &bounds)
        .ok_or_else(|| anyhow!("function {sym:?} has no observed samples; can't disassemble"))?;

    let raw = cmd::run_objdump_function(binary, file_addr, file_addr + size)?;
    let lines = ibs_parse::build_annotated_lines(&raw, &profile.insn_stats, &profile.ip_to_key, load_base);

    let (total_uw, _total_w) = ibs_parse::compute_totals(&profile.insn_stats);
    let total_cycles: u64 = profile.insn_stats.values().map(|s| s.cycles).sum();

    let mut function_name: Option<String> = None;
    let mut out_lines: Vec<AnnotateLine> = Vec::with_capacity(lines.len());
    let mut file_weight: AHashMap<String, u64> = AHashMap::new();
    let mut file_range: AHashMap<String, (u32, u32)> = AHashMap::new();

    let mut function_totals = FunctionTotals::default();
    let mut function_offsets: ahash::AHashSet<u64> = ahash::AHashSet::new();

    // First pass: collect the offsets of every instruction in the function so
    // we can recognise intra-function jump targets in pass two.
    for ln in &lines {
        if let AnnotatedLine::Instruction(insn) = ln {
            function_offsets.insert(insn.offset);
        }
    }

    let mut jump_targets: ahash::AHashSet<u64> = ahash::AHashSet::new();

    for ln in lines {
        match ln {
            AnnotatedLine::Function(h) => {
                function_name = Some(h.name.clone());
                out_lines.push(AnnotateLine {
                    kind: "function",
                    name: Some(h.name),
                    text: None, addr: None, offset: None, disasm: None,
                    cycles: None, cache_counts: None, samples: None,
                    source_line: None, source_file: None,
                    jump_target_offset: None,
                });
            }
            AnnotatedLine::Separator(_) => {
                out_lines.push(AnnotateLine {
                    kind: "sep",
                    name: None, text: None, addr: None, offset: None, disasm: None,
                    cycles: None, cache_counts: None, samples: None,
                    source_line: None, source_file: None,
                    jump_target_offset: None,
                });
            }
            AnnotatedLine::Source(s) => {
                out_lines.push(AnnotateLine {
                    kind: "source",
                    name: None, text: Some(s.text), addr: None, offset: None, disasm: None,
                    cycles: None, cache_counts: None, samples: None,
                    source_line: None, source_file: None,
                    jump_target_offset: None,
                });
            }
            AnnotatedLine::Instruction(insn) => {
                let mut cache_counts: AHashMap<&'static str, u64> = AHashMap::new();
                for lvl in [CacheLevel::L1, CacheLevel::L2, CacheLevel::L3, CacheLevel::DRAM] {
                    let count = insn
                        .stats
                        .cache_counts
                        .get(&lvl)
                        .copied()
                        .unwrap_or(0);
                    cache_counts.insert(lvl.label(), count);
                    *function_totals.cache_counts.entry(lvl.label()).or_default() += count;
                }
                function_totals.cycles += insn.stats.cycles;
                function_totals.samples += insn.stats.total_samples;

                if let (Some(file), Some(line)) = (
                    insn.source_file.as_ref(),
                    insn.source_line,
                ) {
                    *file_weight.entry(file.clone()).or_default() += insn.stats.cycles + 1;
                    file_range
                        .entry(file.clone())
                        .and_modify(|(lo, hi)| {
                            if line < *lo { *lo = line }
                            if line > *hi { *hi = line }
                        })
                        .or_insert((line, line));
                }

                let jump_tgt = parse_jump_target_offset(&insn.disasm)
                    .filter(|off| function_offsets.contains(off));
                if let Some(off) = jump_tgt {
                    jump_targets.insert(off);
                }

                out_lines.push(AnnotateLine {
                    kind: "insn",
                    name: None,
                    text: None,
                    addr: Some(insn.addr),
                    offset: Some(insn.offset),
                    disasm: Some(insn.disasm),
                    cycles: Some(insn.stats.cycles),
                    cache_counts: Some(cache_counts),
                    samples: Some(insn.stats.total_samples),
                    source_line: insn.source_line,
                    source_file: insn.source_file.clone(),
                    jump_target_offset: jump_tgt,
                });
            }
        }
    }

    let mut jump_targets: Vec<u64> = jump_targets.into_iter().collect();
    jump_targets.sort_unstable();

    // Build a source pane per file referenced; sort by descending weight so
    // the heaviest file is the default tab.
    let mut files: Vec<(String, u64)> = file_weight.iter().map(|(f, w)| (f.clone(), *w)).collect();
    files.sort_by(|a, b| b.1.cmp(&a.1));
    let source_panes: Vec<SourcePane> = files
        .into_iter()
        .filter_map(|(file, _)| {
            let range = file_range.get(&file).copied()?;
            build_source_pane_for(&file, range)
        })
        .collect();

    Ok(AnnotateResponse {
        symbol: function_name.unwrap_or_else(|| sym.to_string()),
        binary: profile.binary_path.clone(),
        totals: Totals {
            samples: total_uw,
            weight: 0,
            cycles: total_cycles,
        },
        function_totals,
        lines: out_lines,
        source_panes,
        jump_targets,
    })
}

/// Parse the operand of a jump instruction (already simplified by
/// `simplify_jump` to a function-relative hex offset) and return the
/// target offset. Returns `None` for non-jumps or indirect jumps.
fn parse_jump_target_offset(disasm: &str) -> Option<u64> {
    let trimmed = disasm.trim_start();
    // Match `j[a-z]+` mnemonics; skip `call` and indirect forms.
    let bytes = trimmed.as_bytes();
    if bytes.is_empty() || bytes[0] != b'j' {
        return None;
    }
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_lowercase() {
        i += 1;
    }
    if i == 0 || i >= bytes.len() || !bytes[i].is_ascii_whitespace() {
        return None;
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let rest = &trimmed[i..];
    let rest = rest.strip_prefix("0x").unwrap_or(rest);
    let mut j = 0;
    let rb = rest.as_bytes();
    while j < rb.len() && rb[j].is_ascii_hexdigit() {
        j += 1;
    }
    if j == 0 {
        return None;
    }
    u64::from_str_radix(&rest[..j], 16).ok()
}

fn build_source_pane_for(file: &str, range: (u32, u32)) -> Option<SourcePane> {
    let (lo, hi) = range;
    if lo == 0 || hi < lo {
        return None;
    }
    let body = std::fs::read_to_string(file).ok()?;
    // Pad the visible range slightly so the user sees the full function
    // without having to deduce the line bounds from line numbers alone.
    let pad = 2u32;
    let start = lo.saturating_sub(pad).max(1);
    let end = hi.saturating_add(pad);
    let mut out: Vec<SourcePaneLine> = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let n = (i + 1) as u32;
        if n < start { continue; }
        if n > end { break; }
        out.push(SourcePaneLine { line: n, text: line.to_string() });
    }
    if out.is_empty() {
        return None;
    }
    Some(SourcePane {
        file: file.to_string(),
        start_line: start,
        end_line: end,
        lines: out,
    })
}
