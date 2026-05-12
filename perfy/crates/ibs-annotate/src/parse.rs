//! `objdump` output → annotated lines, plus jump-graph / load-base
//! helpers used by the source+assembly view.
//!
//! All `perf.data` parsing now lives in [`crate::reader`]. This module is
//! deliberately scoped to the assembly side of the pipeline.

use ahash::{AHashMap, AHashSet};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;

use crate::cmd;
use crate::model::{
    AnnotatedLine, FunctionHeader, FunctionSummary, InsnStats, InstructionLine,
    JumpArrow, JumpGraph, SeparatorLine, SourceLine,
};
use crate::symbols::Mapping;
use crate::Result;

// -- Regexes -----------------------------------------------------------------

static FUNC_HEADER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"([0-9a-fA-F]+)\s+<(.+)>:\s*$").unwrap());

static INSN_LINE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*(?:[\d.]*\s*:\s+)?([0-9a-fA-F]+):\s*(.*)").unwrap());

static SOURCE_LINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*:\s*(.*)").unwrap());

/// Match the file:line marker objdump -S emits before each block of code.
/// Examples:
///   `/home/foo/bar.rs:42`
///   `/home/foo/bar.rs:42 (discriminator 1)`
static SOURCE_LOC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?P<path>[/A-Za-z][^:]*):(?P<line>\d+)(?:\s*\([^)]*\))?\s*$").unwrap()
});

static JUMP_OPERAND_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(j[a-z]+\s+)(?:0x)?([0-9a-fA-F]+)(\s+<[^>]*\+0x([0-9a-fA-F]+)>)?(.*)$").unwrap()
});

static JUMP_TARGET_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(j[a-z]+)\s+(?:0x)?([0-9a-f]+)\b").unwrap());

static RUST_HASH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"::h[0-9a-f]+$").unwrap());

const BRACKET_MAX_SPAN: usize = 30;

// -- Disassembly post-processing --------------------------------------------

fn strip_rust_hash(name: &str) -> String {
    RUST_HASH_RE.replace(name, "").into_owned()
}

/// Rewrite `jne 2306f0 <funcname+0x480>` → `jne 480` so the jump operand
/// matches the offset label rendered next to the target instruction.
pub fn simplify_jump(disasm: &str, current_base: u64) -> String {
    let trimmed = disasm.trim_start();
    let Some(cap) = JUMP_OPERAND_RE.captures(trimmed) else {
        return disasm.to_string();
    };
    let off_str: String;
    if let Some(annot) = cap.get(4) {
        off_str = annot.as_str().to_string();
    } else {
        let Ok(target) = u64::from_str_radix(&cap[2], 16) else {
            return disasm.to_string();
        };
        off_str = format!("{:x}", target.wrapping_sub(current_base));
    }
    format!(
        "{}{}{}",
        &cap[1],
        off_str,
        cap.get(5).map(|m| m.as_str()).unwrap_or("")
    )
}

/// Parse a fragment of `objdump -d -S` output.
///
/// `load_base` is added to each file address to recover the runtime IP, and
/// the resulting IP is looked up in `ip_to_key` to attach per-instruction
/// stats. If no IP match exists, we fall back to a `"<sym>\0<offset>"` key.
pub fn build_annotated_lines(
    disasm_output: &str,
    stats: &AHashMap<String, InsnStats>,
    ip_to_key: &AHashMap<u64, String>,
    load_base: u64,
) -> Vec<AnnotatedLine> {
    let mut lines: Vec<AnnotatedLine> = Vec::new();
    let mut current_sym: Option<String> = None;
    let mut current_base: u64 = 0;
    let mut current_source_file: Option<String> = None;
    let mut current_source_line: Option<u32> = None;

    for raw in disasm_output.lines() {
        if raw.trim().is_empty() {
            continue;
        }
        if raw.contains("file format") || raw.starts_with("Disassembly of section") {
            continue;
        }
        if raw.contains("Percent") && raw.contains('|') {
            continue;
        }
        if raw.trim().chars().all(|c| c == '-') {
            lines.push(AnnotatedLine::Separator(SeparatorLine));
            continue;
        }

        if let Some(cap) = FUNC_HEADER_RE.captures(raw) {
            let base = u64::from_str_radix(&cap[1], 16).unwrap_or(0);
            current_base = base;
            let stripped = strip_rust_hash(&cap[2]);
            current_sym = Some(stripped.clone());
            current_source_file = None;
            current_source_line = None;
            lines.push(AnnotatedLine::Separator(SeparatorLine));
            lines.push(AnnotatedLine::Function(FunctionHeader {
                name: stripped,
                base_addr: base,
            }));
            continue;
        }

        // `/path/file.rs:42` — source-location marker. Stash and don't emit
        // a SourceLine; the source text (if any) follows on the next lines.
        if let Some(cap) = SOURCE_LOC_RE.captures(raw.trim_start()) {
            let path = cap.name("path").unwrap().as_str().to_string();
            let line: u32 = cap.name("line").unwrap().as_str().parse().unwrap_or(0);
            // Filter out things like `1234.567:` that look like file:line but
            // aren't paths — heuristic: must contain a / or end with a known
            // source extension.
            if path.starts_with('/') || path.contains('.') {
                current_source_file = Some(path);
                current_source_line = if line > 0 { Some(line) } else { None };
                continue;
            }
        }

        if let Some(cap) = INSN_LINE_RE.captures(raw) {
            let addr_s = cap[1].to_string();
            let file_addr = u64::from_str_radix(&addr_s, 16).unwrap_or(0);
            let offset = file_addr.wrapping_sub(current_base);
            let runtime_ip = file_addr.wrapping_add(load_base);

            let key = if let Some(k) = ip_to_key.get(&runtime_ip) {
                Some(k.clone())
            } else if let Some(sym) = &current_sym {
                let candidate = format!("{sym}\0{offset}");
                if stats.contains_key(&candidate) {
                    Some(candidate)
                } else {
                    None
                }
            } else {
                None
            };

            let stats_for_line = key
                .as_ref()
                .and_then(|k| stats.get(k))
                .cloned()
                .unwrap_or_default();

            lines.push(AnnotatedLine::Instruction(InstructionLine {
                addr: addr_s,
                offset,
                disasm: simplify_jump(&cap[2], current_base),
                sym: current_sym.clone(),
                stats: stats_for_line,
                source_file: current_source_file.clone(),
                source_line: current_source_line,
            }));
            continue;
        }

        if let Some(cap) = SOURCE_LINE_RE.captures(raw) {
            lines.push(AnnotatedLine::Source(SourceLine {
                text: cap[1].trim_start().to_string(),
            }));
        } else if current_sym.is_some() {
            lines.push(AnnotatedLine::Source(SourceLine {
                text: raw.trim().to_string(),
            }));
        }
    }

    lines
}

// -- Jump graph --------------------------------------------------------------

pub fn compute_jump_graph(lines: &[AnnotatedLine]) -> JumpGraph {
    let mut offset_to_idx: AHashMap<u64, usize> = AHashMap::new();
    let mut max_offset: u64 = 0;
    for (i, ln) in lines.iter().enumerate() {
        if let AnnotatedLine::Instruction(insn) = ln {
            offset_to_idx.insert(insn.offset, i);
            if insn.offset > max_offset {
                max_offset = insn.offset;
            }
        }
    }

    let mut arrows: Vec<JumpArrow> = Vec::new();
    for (i, ln) in lines.iter().enumerate() {
        let AnnotatedLine::Instruction(insn) = ln else { continue };
        let Some(cap) = JUMP_TARGET_RE.captures(insn.disasm.trim_start()) else { continue };
        let Ok(target_off) = u64::from_str_radix(&cap[2], 16) else { continue };
        let src_off = insn.offset;
        match offset_to_idx.get(&target_off) {
            Some(&tgt_idx) => {
                let span = (i as i64 - tgt_idx as i64).unsigned_abs() as usize;
                arrows.push(JumpArrow {
                    src_idx: i,
                    tgt_idx: tgt_idx as i64,
                    forward: tgt_idx as i64 > i as i64,
                    lane: 0,
                    is_short: span <= BRACKET_MAX_SPAN,
                });
            }
            None => {
                arrows.push(JumpArrow {
                    src_idx: i,
                    tgt_idx: -1,
                    forward: target_off > src_off,
                    lane: 0,
                    is_short: false,
                });
            }
        }
    }

    // Greedy lane assignment for short arrows.
    let mut short_indices: Vec<usize> = arrows
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_short)
        .map(|(i, _)| i)
        .collect();
    short_indices.sort_by(|&a, &b| {
        let aa = &arrows[a];
        let bb = &arrows[b];
        let alo = aa.src_idx.min(aa.tgt_idx as usize);
        let blo = bb.src_idx.min(bb.tgt_idx as usize);
        let aspan = (aa.src_idx as i64 - aa.tgt_idx).unsigned_abs() as usize;
        let bspan = (bb.src_idx as i64 - bb.tgt_idx).unsigned_abs() as usize;
        alo.cmp(&blo).then(bspan.cmp(&aspan))
    });

    let mut placed_ranges: Vec<(u32, usize, usize)> = Vec::new();
    let mut max_lane: i32 = -1;
    for &idx in &short_indices {
        let (lo, hi) = {
            let a = &arrows[idx];
            let s = a.src_idx;
            let t = a.tgt_idx as usize;
            (s.min(t), s.max(t))
        };
        for lane in 0u32..16 {
            let conflict = placed_ranges
                .iter()
                .any(|&(l, plo, phi)| l == lane && !(hi < plo || lo > phi));
            if !conflict {
                arrows[idx].lane = lane;
                placed_ranges.push((lane, lo, hi));
                if lane as i32 > max_lane {
                    max_lane = lane as i32;
                }
                break;
            }
        }
    }

    let mut targets: AHashSet<usize> = AHashSet::new();
    for a in &arrows {
        if a.tgt_idx >= 0 {
            targets.insert(a.tgt_idx as usize);
        }
    }
    let max_lanes = if max_lane < 0 { 0 } else { (max_lane + 1) as u32 };
    let addr_width = std::cmp::max(4, format!("{max_offset:x}").len());

    JumpGraph {
        arrows,
        targets,
        max_lanes,
        addr_width,
    }
}

// -- nm: per-function size table --------------------------------------------

pub fn get_function_bounds(binary: &str) -> Result<AHashMap<u64, u64>> {
    let stdout = cmd::run_nm(binary)?;
    let mut bounds: AHashMap<u64, u64> = AHashMap::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }
        if parts[2] != "t" && parts[2] != "T" {
            continue;
        }
        let Ok(addr) = u64::from_str_radix(parts[0], 16) else { continue };
        let Ok(size) = u64::from_str_radix(parts[1], 16) else { continue };
        bounds.insert(addr, size);
    }
    Ok(bounds)
}

/// Compute the runtime ASLR offset given an mmap2 record for `binary` and
/// the binary's executable LOAD segment.
///
/// `load_base = mapping.addr_lo + p_offset - mapping.page_offset - p_vaddr`
///
/// — same formula as the Python `compute_load_base`, but the mmap fields
/// come from the perf.data MMAP2 record (we already read it natively) so
/// no `perf script --show-mmap-events` invocation is needed.
pub fn compute_load_base_from_mapping(mapping: &Mapping) -> Result<u64> {
    let elf_out = cmd::run_readelf_l(&mapping.binary)?;
    let mut p_offset: Option<u64> = None;
    let mut p_vaddr: Option<u64> = None;
    for line in elf_out.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.iter().any(|p| *p == "LOAD") && parts.iter().any(|p| *p == "E") {
            let idx = parts.iter().position(|p| *p == "LOAD").unwrap();
            if let (Some(off), Some(va)) = (parts.get(idx + 1), parts.get(idx + 2)) {
                p_offset = u64::from_str_radix(off.trim_start_matches("0x"), 16).ok();
                p_vaddr = u64::from_str_radix(va.trim_start_matches("0x"), 16).ok();
                break;
            }
        }
    }
    let (Some(p_offset), Some(p_vaddr)) = (p_offset, p_vaddr) else {
        return Ok(0);
    };

    let load_base = mapping
        .addr_lo
        .wrapping_add(p_offset)
        .wrapping_sub(mapping.page_offset)
        .wrapping_sub(p_vaddr);
    Ok(load_base)
}

// -- Function-level aggregation + totals ------------------------------------

pub fn compute_function_summaries(stats: &AHashMap<String, InsnStats>) -> Vec<FunctionSummary> {
    let mut by_func: AHashMap<String, FunctionSummary> = AHashMap::new();
    for (key, s) in stats {
        let func_name = key.split('\0').next().unwrap_or("").to_string();
        let f = by_func
            .entry(func_name.clone())
            .or_insert_with(|| FunctionSummary { name: func_name, ..Default::default() });
        f.total_samples += s.total_samples;
        f.cycles += s.cycles;
        for (lvl, c) in &s.cache_counts {
            f.weighted_cost += c * lvl.weight();
            *f.cache_counts.entry(*lvl).or_default() += c;
        }
    }
    let mut out: Vec<FunctionSummary> = by_func.into_values().collect();
    out.sort_by(|a, b| b.cycles.cmp(&a.cycles));
    out
}

pub fn compute_totals(stats: &AHashMap<String, InsnStats>) -> (u64, u64) {
    stats
        .par_iter()
        .map(|(_, s)| {
            let mut uw = 0u64;
            let mut w = 0u64;
            for (lvl, c) in &s.cache_counts {
                uw += c;
                w += c * lvl.weight();
            }
            (uw, w)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplify_jump_with_annotation() {
        assert_eq!(
            simplify_jump("jne 2306f0 <funcname+0x480>", 0x230000),
            "jne 480"
        );
    }

    #[test]
    fn simplify_jump_relative_only() {
        assert_eq!(simplify_jump("jne 2306f0", 0x230000), "jne 6f0");
    }

    #[test]
    fn simplify_jump_passthrough_for_non_jump() {
        assert_eq!(simplify_jump("mov rax, rbx", 0), "mov rax, rbx");
    }
}
