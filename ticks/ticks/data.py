"""Wraps ibs_annotate's parsing pipeline and serializes results to JSON."""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock

from ibs_annotate.model import (
    CACHE_LEVELS,
    CacheLevel,
    FunctionHeader,
    InstructionLine,
    JumpGraph,
    OpType,
    SeparatorLine,
    SnoopStatus,
    SourceLine,
    TlbLevel,
)
from ibs_annotate.parse import (
    build_annotated_lines,
    compute_function_summaries,
    compute_jump_graph,
    compute_load_base,
    compute_totals,
    find_binary,
    get_function_bounds,
    parse_cycles,
    parse_ibs_raw,
    parse_prefetch_pmc,
    parse_samples,
    run_objdump_function,
)


@dataclass
class LoadedPerfData:
    perf_data: str
    binary: str
    load_base: int
    func_bounds: dict
    stats: dict
    ip_to_key: dict
    total_uw: int
    total_w: int
    total_cycles: int
    summaries: list
    skipped_lines: list = field(default_factory=list)
    pf_summary: dict = field(default_factory=dict)
    cache_summary: dict = field(default_factory=dict)
    _line_cache: dict = field(default_factory=dict)
    _jg_cache: dict = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def annotated(self, func_name: str):
        with self._lock:
            if func_name in self._line_cache:
                return self._line_cache[func_name], self._jg_cache[func_name]
        for ip, key in self.ip_to_key.items():
            sym, off_s = key.split("\0", 1)
            if sym == func_name:
                runtime_base = ip - int(off_s)
                file_addr = runtime_base - self.load_base
                size = self.func_bounds.get(file_addr, 0x10000)
                out = run_objdump_function(self.binary, file_addr, file_addr + size)
                lines = build_annotated_lines(out, self.stats, self.ip_to_key, self.load_base)
                jg = compute_jump_graph(lines)
                with self._lock:
                    self._line_cache[func_name] = lines
                    self._jg_cache[func_name] = jg
                return lines, jg
        return [], JumpGraph()


def load_perf_data(perf_data: str, binary: str | None = None) -> LoadedPerfData:
    proc_samples = subprocess.Popen(
        ["perf", "script", "-F", "ip,sym,symoff,data_src", "-G", "-i", perf_data],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    proc_ibs = subprocess.Popen(
        ["perf", "script", "-D", "-i", perf_data],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    proc_pmc = subprocess.Popen(
        ["perf", "script", "-F", "event,period,ip,sym,symoff", "-G", "-i", perf_data],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    proc_cycles = subprocess.Popen(
        ["perf", "script", "-F", "comm,event,ip,sym,symoff", "-i", perf_data],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    drain_pool = ThreadPoolExecutor(max_workers=2)
    pmc_drain = drain_pool.submit(proc_pmc.communicate)
    cycles_drain = drain_pool.submit(proc_cycles.communicate)

    so, se = proc_samples.communicate()
    if proc_samples.returncode != 0:
        raise RuntimeError(f"perf script failed: {se.strip()}")
    stats, skipped, ip_to_key = parse_samples(so)
    total_uw, total_w = compute_totals(stats)

    pmc_output, _ = pmc_drain.result()
    cycles_output, _ = cycles_drain.result()

    def _do_ibs():
        parse_ibs_raw(proc_ibs.stdout, stats, ip_to_key)
        proc_ibs.wait()

    def _do_pmc():
        return parse_prefetch_pmc(pmc_output, stats, ip_to_key)

    def _do_cycles():
        return parse_cycles(cycles_output, stats, ip_to_key)

    with ThreadPoolExecutor(max_workers=3) as pool:
        ibs_f = pool.submit(_do_ibs)
        pmc_f = pool.submit(_do_pmc)
        cyc_f = pool.submit(_do_cycles)
        ibs_f.result()
        pf_disp, pf_dc, pf_mab, pf_fills = pmc_f.result()
        total_cycles = cyc_f.result()

    binary_path = binary or find_binary(perf_data)
    load_base = compute_load_base(perf_data, binary_path)
    func_bounds = get_function_bounds(binary_path)
    summaries = compute_function_summaries(stats)

    # Cache hierarchy summary (mirrors ibs_annotate cli)
    totals_per_level = {lvl.label: 0 for lvl in CACHE_LEVELS}
    for s in stats.values():
        for lvl, c in s.cache_counts.items():
            totals_per_level[lvl.label] += c
    cache_summary = {"per_level": totals_per_level}

    pf_summary = {
        "dispatched": pf_disp,
        "fills": pf_fills,
        "ineffective_dc": pf_dc,
        "ineffective_mab": pf_mab,
    }

    return LoadedPerfData(
        perf_data=perf_data,
        binary=binary_path,
        load_base=load_base,
        func_bounds=func_bounds,
        stats=stats,
        ip_to_key=ip_to_key,
        total_uw=total_uw,
        total_w=total_w,
        total_cycles=total_cycles,
        summaries=summaries,
        skipped_lines=skipped.summary_lines(),
        pf_summary=pf_summary,
        cache_summary=cache_summary,
    )


# -- Per-line gutter (mirrors ibs_annotate.tui._gutter_cells) -----------------


def _gutter_cells(idx: int, jg: JumpGraph):
    if not jg.arrows:
        return []
    width = jg.max_lanes + 1
    cells = [{"glyph": " ", "kind": "blank"} for _ in range(width)]

    for a in jg.arrows:
        if not a.is_short:
            continue
        lo, hi = sorted((a.src_idx, a.tgt_idx))
        if not (lo <= idx <= hi):
            continue
        if idx == lo:
            glyph = "┌"  # ┌
        elif idx == hi:
            glyph = "└"  # └
        else:
            glyph = "│"  # │
        if cells[a.lane]["kind"] == "blank":
            cells[a.lane] = {"glyph": glyph, "kind": "lane"}

    ending = [a for a in jg.arrows if a.is_short and idx in (a.src_idx, a.tgt_idx)]
    if ending:
        leftmost = min(a.lane for a in ending)
        for j in range(leftmost + 1, width):
            cells[j] = {"glyph": "─", "kind": "arm"}  # ─

    for a in jg.arrows:
        if a.is_short or a.src_idx != idx:
            continue
        cells[-1] = {"glyph": "↓" if a.forward else "↑", "kind": "longarrow"}
        break

    return cells


# -- JSON serializers ----------------------------------------------------------


def serialize_summaries(data: LoadedPerfData):
    out = []
    for f in data.summaries:
        cyc_pct = (100.0 * f.cycles / data.total_cycles) if data.total_cycles else 0.0
        out.append({
            "name": f.name,
            "cycles": f.cycles,
            "samples": f.total_samples,
            "weighted_cost": f.weighted_cost,
            "cycles_pct": cyc_pct,
        })
    return out


def _stats_to_json(stats):
    cache_counts = {lvl.label: stats.cache_counts.get(lvl, 0) for lvl in CACHE_LEVELS}
    tlb_counts = {t.name: stats.tlb_counts.get(t, 0) for t in TlbLevel}
    op_counts = {o.label: stats.op_counts.get(o, 0) for o in OpType}
    snoop_counts = {s.label: stats.snoop_counts.get(s, 0) for s in SnoopStatus}
    ibs = stats.ibs
    return {
        "cache_counts": cache_counts,
        "tlb_counts": tlb_counts,
        "op_counts": op_counts,
        "snoop_counts": snoop_counts,
        "locked_count": stats.locked_count,
        "total_samples": stats.total_samples,
        "cycles": stats.cycles,
        "ibs": {
            "sample_count": ibs.sample_count,
            "avg_dc_miss_lat": ibs.avg_dc_miss_lat,
            "avg_tlb_refill_lat": ibs.avg_tlb_refill_lat,
            "avg_comp_to_ret": ibs.avg_comp_to_ret,
            "avg_tag_to_ret": ibs.avg_tag_to_ret,
            "avg_mabs": ibs.avg_mabs,
            "mabs_max": ibs.mabs_max,
            "dc_miss_count": ibs.dc_miss_count,
            "l2_miss_count": ibs.l2_miss_count,
            "dc_l1_tlb_miss_count": ibs.dc_l1_tlb_miss_count,
            "dc_l2_tlb_miss_count": ibs.dc_l2_tlb_miss_count,
            "brn_ret_count": ibs.brn_ret_count,
            "brn_misp_count": ibs.brn_misp_count,
            "brn_taken_count": ibs.brn_taken_count,
            "brn_return_count": ibs.brn_return_count,
            "brn_fuse_count": ibs.brn_fuse_count,
        },
    }


def serialize_function(data: LoadedPerfData, func_name: str):
    lines, jg = data.annotated(func_name)
    out_lines = []
    for idx, line in enumerate(lines):
        if isinstance(line, InstructionLine):
            stats = line.stats
            mnem, _, operands = line.disasm.partition(" ")
            cyc_pct = (100.0 * stats.cycles / data.total_cycles) if data.total_cycles else 0.0
            cache_pct = {}
            for lvl in CACHE_LEVELS:
                c = stats.cache_counts.get(lvl, 0)
                if c and data.total_w:
                    cache_pct[lvl.label] = 100.0 * c * lvl.weight / data.total_w
                else:
                    cache_pct[lvl.label] = 0.0
            out_lines.append({
                "type": "instruction",
                "addr": line.addr,
                "offset": line.offset,
                "mnem": mnem.strip(),
                "operands": operands.strip(),
                "is_target": idx in jg.targets,
                "gutter": _gutter_cells(idx, jg),
                "cache_counts": {lvl.label: stats.cache_counts.get(lvl, 0) for lvl in CACHE_LEVELS},
                "cache_weighted_pct": cache_pct,
                "cycles": stats.cycles,
                "cycles_pct": cyc_pct,
                "samples": stats.total_samples,
                "stats": _stats_to_json(stats),
            })
        elif isinstance(line, SourceLine):
            out_lines.append({
                "type": "source",
                "text": line.text,
                "gutter": _gutter_cells(idx, jg),
            })
        elif isinstance(line, FunctionHeader):
            out_lines.append({"type": "function_header", "name": line.name})
        elif isinstance(line, SeparatorLine):
            out_lines.append({"type": "separator"})
    return {
        "name": func_name,
        "lines": out_lines,
        "max_lanes": jg.max_lanes,
        "addr_width": jg.addr_width,
        "totals": {
            "cycles": data.total_cycles,
            "unweighted": data.total_uw,
            "weighted": data.total_w,
        },
    }
