from __future__ import annotations

import re
import subprocess
from collections import defaultdict

from .model import (
    CACHE_LEVELS,
    AnnotatedLine,
    CacheLevel,
    FunctionHeader,
    FunctionSummary,
    IBSRaw,
    InsnStats,
    InstructionLine,
    OpType,
    PrefetchCounts,
    SeparatorLine,
    SkippedSamples,
    SnoopStatus,
    SourceLine,
    TlbLevel,
)

_SYM_OFFSET_RE = re.compile(r"(\S+)\+0x([0-9a-fA-F]+)")
_IP_SYM_RE = re.compile(r"([0-9a-fA-F]{8,})\s+(.+)\+0x([0-9a-fA-F]+)\s*$")
_FUNC_HEADER_RE = re.compile(r"([0-9a-fA-F]+)\s+<(.+)>:\s*$")
_INSN_LINE_RE = re.compile(r"^\s*(?:[\d.]*\s*:\s+)?([0-9a-fA-F]+):\s*(.*)")
_SOURCE_LINE_RE = re.compile(r"\s*:\s*(.*)")
_SAMPLE_IP_RE = re.compile(r"PERF_RECORD_SAMPLE.*:\s+0x([0-9a-fA-F]+)\s+period:")


# -- perf subprocess ----------------------------------------------------------


def run_perf(args: list[str], perf_data: str) -> str:
    result = subprocess.run(
        ["perf", *args, "-i", perf_data],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"perf {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def find_binary(perf_data: str) -> str:
    """Auto-detect the main user binary from perf data via buildid-list."""
    result = subprocess.run(
        ["perf", "buildid-list", "-i", perf_data],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"perf buildid-list failed: {result.stderr.strip()}")

    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        path = parts[1]
        if path.startswith("[") or not path.startswith("/"):
            continue
        if path.startswith(("/usr/", "/lib/", "/lib64/")):
            continue
        return path

    raise RuntimeError(
        "Could not auto-detect binary from perf buildid-list; pass --binary explicitly"
    )


def run_objdump(binary: str) -> str:
    """Disassemble the binary in one shot — much faster than perf annotate.

    -S interleaves source (requires DWARF; for Rust set [profile.release] debug = true).
    """
    result = subprocess.run(
        ["objdump", "-d", "-S", "--no-show-raw-insn", "--demangle", binary],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"objdump failed: {result.stderr.strip()}")
    return result.stdout


def run_perf_annotate_sym(sym: str, perf_data: str) -> str:
    """Run perf annotate for a single symbol — fast, handles PIE correctly."""
    result = subprocess.run(
        ["perf", "annotate", "--symbol", sym, "--stdio", "-i", perf_data],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"perf annotate failed: {result.stderr.strip()}")
    return result.stdout


# -- data_src field extraction -------------------------------------------------


def _extract_field(line: str, name: str) -> str:
    m = re.search(rf"\|{name}\s+([^|]+)", line)
    if not m:
        raise ValueError(f"Missing |{name}| in: {line!r}")
    return m.group(1).strip()


def _parse_cache_level(op: str, lvl: str) -> CacheLevel | None:
    if op == "N/A":
        return CacheLevel.NON_MEMORY
    if "L1" in lvl:
        return CacheLevel.L1
    if "LFB" in lvl or "MAB" in lvl:
        return CacheLevel.LFB
    if "L2" in lvl:
        return CacheLevel.L2
    if "L3" in lvl:
        return CacheLevel.L3
    if "RAM" in lvl and "Remote" not in lvl:
        return CacheLevel.DRAM
    if "Remote" in lvl:
        return CacheLevel.REMOTE
    if lvl == "N/A" or "Uncached" in lvl or "I/O" in lvl:
        return None
    raise ValueError(f"Unknown cache level: OP={op!r} LVL={lvl!r}")


def _parse_tlb(op: str, tlb: str) -> TlbLevel:
    if op == "N/A":
        return TlbLevel.NA
    if "miss" in tlb.lower():
        # maor: All misses are TLB l2 misses
        return TlbLevel.MISS
    if "L1" in tlb:
        return TlbLevel.L1_HIT
    if "L2" in tlb:
        return TlbLevel.L2_HIT
    if tlb == "N/A":
        return TlbLevel.NA
    raise ValueError(f"Unknown TLB level: OP={op!r} TLB={tlb!r}")


def _parse_op(op: str) -> OpType:
    if op == "LOAD":
        return OpType.LOAD
    if op == "STORE":
        return OpType.STORE
    if op == "N/A":
        return OpType.NA
    raise ValueError(f"Unknown op type: {op!r}")


def _parse_snoop(snp: str) -> SnoopStatus:
    if snp == "N/A":
        return SnoopStatus.NA
    if "HitM" in snp:
        return SnoopStatus.HITM
    if "Hit" in snp:
        return SnoopStatus.HIT
    if "Miss" in snp:
        return SnoopStatus.MISS
    if "None" in snp:
        return SnoopStatus.NONE
    raise ValueError(f"Unknown snoop status: {snp!r}")


# -- Phase 1: perf script -> per-instruction stats ----------------------------


def parse_samples(
    script_output: str,
) -> tuple[dict[str, InsnStats], SkippedSamples, dict[int, str]]:
    """Returns (stats_by_key, skipped, ip_to_key)."""
    stats: dict[str, InsnStats] = defaultdict(InsnStats)
    skipped = SkippedSamples()
    ip_to_key: dict[int, str] = {}

    for line in script_output.splitlines():
        if "|OP " not in line:
            continue

        op_s = _extract_field(line, "OP")
        lvl_s = _extract_field(line, "LVL")
        tlb_s = _extract_field(line, "TLB")
        lck_s = _extract_field(line, "LCK")
        snp_s = _extract_field(line, "SNP")

        cache = _parse_cache_level(op_s, lvl_s)
        tlb = _parse_tlb(op_s, tlb_s)
        op = _parse_op(op_s)
        snoop = _parse_snoop(snp_s)
        locked = lck_s != "N/A"

        weight = 0

        # Extract IP + sym + offset (search after last | to avoid matching data_src hex)
        after_pipe = line.rsplit("|", 1)[-1]
        ip_m = _IP_SYM_RE.search(after_pipe)
        if not ip_m:
            if "[unknown]" in line:
                if cache is not None:
                    skipped.unknown_symbol[cache] += 1
                else:
                    skipped.unknown_level += 1
                continue
            raise ValueError(f"No sym+offset found: {line!r}")

        if cache is None:
            skipped.unknown_level += 1
            continue

        ip_val = int(ip_m.group(1), 16)
        key = f"{ip_m.group(2)}\0{int(ip_m.group(3), 16)}"
        ip_to_key[ip_val] = key
        stats[key].add(cache, tlb, op, snoop, locked, weight)

    return stats, skipped, ip_to_key


# -- Cycles PMC samples -------------------------------------------------------


def parse_cycles(
    script_output: str,
    stats: dict[str, InsnStats],
    ip_to_key: dict[int, str],
) -> int:
    """Parse cycles PMC samples, attribute per-instruction. Returns total.

    Format: each cycles sample is a "cycles:" line followed by the IP on
    the next line (first call-graph entry).  We grab the first sym+offset
    line after each "cycles:" marker.
    """
    total = 0
    want_ip = False
    for line in script_output.splitlines():
        if "cycles:" in line:
            want_ip = True
            continue
        if not want_ip:
            continue
        want_ip = False
        if "[unknown]" in line:
            continue
        m = _IP_SYM_RE.search(line)
        if not m:
            continue
        ip_val = int(m.group(1), 16)
        key = ip_to_key.get(ip_val)
        if key is None:
            key = f"{m.group(2)}\0{int(m.group(3), 16)}"
            ip_to_key[ip_val] = key
        if key not in stats:
            stats[key] = InsnStats()
        stats[key].cycles += 1
        total += 1
    return total


# -- Phase 2: merge with perf annotate disassembly -----------------------------


def build_annotated_lines(
    disasm_output: str,
    stats: dict[str, InsnStats],
    ip_to_key: dict[int, str],
    load_base: int = 0,
) -> list[AnnotatedLine]:
    """Parse disassembly from either objdump or perf annotate --stdio.

    load_base is added to file addresses from objdump to convert them to
    the runtime IPs used as keys in ip_to_key (needed for PIE binaries).
    """
    lines: list[AnnotatedLine] = []
    current_sym: str | None = None
    current_base = 0
    _matched_ip = 0
    _matched_sym = 0
    _unmatched = 0

    for raw in disasm_output.splitlines():
        if not raw.strip():
            continue
        # Skip noise from objdump and perf annotate headers
        if "file format" in raw or raw.startswith("Disassembly of section"):
            continue
        if "Percent" in raw and "|" in raw:
            continue

        if re.match(r"^-+\s*$", raw):
            lines.append(SeparatorLine())
            continue

        func_m = _FUNC_HEADER_RE.search(raw)
        if func_m:
            current_base = int(func_m.group(1), 16)
            current_sym = func_m.group(2)
            # Strip Rust hash suffix so sym matches perf's names
            current_sym = _RUST_HASH_RE.sub("", current_sym)
            lines.append(SeparatorLine())
            lines.append(FunctionHeader(name=current_sym, base_addr=current_base))
            continue

        insn_m = _INSN_LINE_RE.match(raw)
        if insn_m:
            addr_s = insn_m.group(1)
            file_addr = int(addr_s, 16)
            offset = file_addr - current_base
            # Convert file address to runtime IP for lookup in ip_to_key
            runtime_ip = file_addr + load_base
            key = ip_to_key.get(runtime_ip)
            if key is not None:
                _matched_ip += 1
            else:
                key = f"{current_sym}\0{offset}" if current_sym else ""
                if key in stats:
                    _matched_sym += 1
                else:
                    _unmatched += 1
            lines.append(
                InstructionLine(
                    addr=addr_s,
                    offset=offset,
                    disasm=insn_m.group(2),
                    sym=current_sym,
                    stats=stats.get(key, InsnStats()),
                )
            )
            continue

        # Source lines from objdump -S or perf annotate (": <source>")
        src_m = _SOURCE_LINE_RE.match(raw)
        if src_m:
            lines.append(SourceLine(text=src_m.group(1)))
        elif current_sym is not None:
            lines.append(SourceLine(text=raw.rstrip()))

    import sys
    total = _matched_ip + _matched_sym + _unmatched
    if total:
        print(
            f"  Instruction matching: {_matched_ip} by IP, "
            f"{_matched_sym} by sym+offset, "
            f"{_unmatched} unmatched "
            f"(load_base=0x{load_base:x})",
            file=sys.stderr,
        )

    return lines


# -- Phase 2b: raw IBS register parsing from perf script -D --------------------


def _ibs_int(line: str, field: str) -> int:
    m = re.search(rf"{field}\s+(\d+)", line)
    return int(m.group(1)) if m else 0


def parse_ibs_raw(
    stream,
    stats: dict[str, InsnStats],
    ip_to_key: dict[int, str],
) -> None:
    """Parse raw IBS fields from a perf script -D output stream."""
    _matched = 0
    _no_ip = 0
    _no_key = 0
    _no_stats = 0

    # Accumulate IBS fields per sample
    comp_to_ret = 0
    tag_to_ret = 0
    dc_miss_lat = 0
    tlb_refill_lat = 0
    mabs = 0
    mem_width = 0
    sw_pf = False
    misaligned = False
    dc_miss_no_mab = False
    dc_l1_tlb_miss = False
    dc_l2_tlb_miss = False
    dc_miss = False
    l2_miss = False
    is_mem_op = False
    have_ibs = False

    for line in stream:
        if line.startswith("ibs_op_data:"):
            comp_to_ret = _ibs_int(line, "CompToRetCtr")
            tag_to_ret = _ibs_int(line, "TagToRetCtr")
            have_ibs = True

        elif line.startswith("ibs_op_data3:"):
            dc_miss_lat = _ibs_int(line, "DcMissLat")
            tlb_refill_lat = _ibs_int(line, "TlbRefillLat")
            mabs = _ibs_int(line, "OpDcMissOpenMemReqs")
            mem_width = _ibs_int(line, "OpMemWidth")
            sw_pf = _ibs_int(line, "SwPf") != 0
            misaligned = _ibs_int(line, "DcMisAcc") != 0
            dc_miss_no_mab = _ibs_int(line, "DcMissNoMabAlloc") != 0
            dc_l1_tlb_miss = _ibs_int(line, "DcL1TlbMiss") != 0
            dc_l2_tlb_miss = _ibs_int(line, "DcL2TlbMiss") != 0
            dc_miss = _ibs_int(line, "DcMiss") != 0
            l2_miss = _ibs_int(line, "L2Miss") != 0
            is_mem_op = _ibs_int(line, "LdOp") != 0 or _ibs_int(line, "StOp") != 0
            have_ibs = True

        elif "PERF_RECORD_SAMPLE" in line and have_ibs:
            ip_m = _SAMPLE_IP_RE.search(line)
            if not ip_m:
                _no_ip += 1
            elif ip_m:
                ip = int(ip_m.group(1), 16)
                key = ip_to_key.get(ip)
                if not key:
                    _no_key += 1
                elif key not in stats:
                    _no_stats += 1
                else:
                    _matched += 1
                    ibs = stats[key].ibs
                    ibs.sample_count += 1
                    if dc_miss_lat > 0:
                        ibs.dc_miss_lat_sum += dc_miss_lat
                        ibs.dc_miss_lat_count += 1
                    if tlb_refill_lat > 0:
                        ibs.tlb_refill_lat_sum += tlb_refill_lat
                        ibs.tlb_refill_lat_count += 1
                    if comp_to_ret > 0:
                        ibs.comp_to_ret_sum += comp_to_ret
                        ibs.comp_to_ret_count += 1
                    if tag_to_ret > 0:
                        ibs.tag_to_ret_sum += tag_to_ret
                        ibs.tag_to_ret_count += 1
                    if mabs > 0:
                        ibs.mabs_sum += mabs
                        ibs.mabs_count += 1
                        if mabs > ibs.mabs_max:
                            ibs.mabs_max = mabs
                    if sw_pf:
                        ibs.sw_pf_count += 1
                    if misaligned:
                        ibs.misaligned_count += 1
                    if dc_miss_no_mab:
                        ibs.dc_miss_no_mab_count += 1
                    if mem_width > 0:
                        ibs.mem_width_counts[mem_width] += 1
                    if dc_l1_tlb_miss:
                        ibs.dc_l1_tlb_miss_count += 1
                    if dc_l2_tlb_miss:
                        ibs.dc_l2_tlb_miss_count += 1
                    if is_mem_op:
                        ibs.mem_op_count += 1
                    if dc_miss:
                        ibs.dc_miss_count += 1
                    if l2_miss:
                        ibs.l2_miss_count += 1

            # Reset for next sample
            comp_to_ret = tag_to_ret = dc_miss_lat = tlb_refill_lat = 0
            mabs = mem_width = 0
            sw_pf = misaligned = dc_miss_no_mab = False
            dc_l1_tlb_miss = dc_l2_tlb_miss = dc_miss = l2_miss = False
            is_mem_op = False
            have_ibs = False

    import sys
    total = _matched + _no_ip + _no_key + _no_stats
    if total:
        dropped = _no_ip + _no_key + _no_stats
        print(
            f"  IBS raw: {_matched} matched, {dropped} dropped "
            f"(no_ip={_no_ip}, no_key={_no_key}, no_stats={_no_stats})",
            file=sys.stderr,
        )


# -- Totals --------------------------------------------------------------------


def compute_totals(stats: dict[str, InsnStats]) -> tuple[int, int]:
    """Return (unweighted_total, weighted_total)."""
    unweighted = 0
    weighted = 0
    for s in stats.values():
        for lvl, c in s.cache_counts.items():
            unweighted += c
            weighted += c * lvl.weight
    return unweighted, weighted


# -- PMC prefetch counters (perf stat) -----------------------------------------

_STAT_RE = re.compile(r"^\s*([\d,]+)\s+(.+?)\s*$")


_PMC_EVENT_RE = re.compile(
    r"^\s*\S+\s+\d+\s+[\d.]+:\s+\d+\s+(cpu/event=0x[0-9a-fA-F]+,umask=0x[0-9a-fA-F]+/).*\s+([0-9a-fA-F]+)\s+(\S+)\+0x([0-9a-fA-F]+)"
)
# Simpler: just match event name and sym+offset from perf script output
_PMC_LINE_RE = re.compile(r"(event=0x[0-9a-fA-F]+,umask=0x[0-9a-fA-F]+)")


_PERIOD_RE = re.compile(r"^\s*(\d+)\s+cpu/")


def parse_prefetch_pmc(
    script_output: str,
    stats: dict[str, InsnStats],
    ip_to_key: dict[int, str],
) -> tuple[int, int, int, int]:
    """Parse PMC prefetch counter samples, attribute per-instruction.

    Each sample is weighted by its period (one sample = period events).
    Returns (total_dispatched, total_ineffective_dc, total_ineffective_mab, total_fills).
    """
    total_dispatched = 0
    total_ineff_dc = 0
    total_ineff_mab = 0
    total_fills = 0

    for line in script_output.splitlines():
        pmc_m = _PMC_LINE_RE.search(line)
        if not pmc_m:
            continue
        event_spec = pmc_m.group(1).lower()

        # Extract period
        period_m = _PERIOD_RE.search(line)
        if not period_m:
            raise ValueError(f"No period found in PMC line: {line!r}")
        period = int(period_m.group(1))

        # Extract sym+offset (after event spec)
        after_event = line[pmc_m.end():]
        sym_m = re.search(r"([0-9a-fA-F]{8,})\s+(.+)\+0x([0-9a-fA-F]+)\s*$", after_event)
        if not sym_m:
            continue

        key = f"{sym_m.group(2)}\0{int(sym_m.group(3), 16)}"

        if key not in stats:
            stats[key] = InsnStats()
        s = stats[key]

        if "event=0x4b" in event_spec:
            s.prefetch.dispatched += period
            total_dispatched += period
        elif "event=0x52" in event_spec:
            if "umask=0x1" in event_spec or "umask=0x01" in event_spec:
                s.prefetch.ineffective_dc_hit += period
                total_ineff_dc += period
            elif "umask=0x2" in event_spec or "umask=0x02" in event_spec:
                s.prefetch.ineffective_mab_match += period
                total_ineff_mab += period
        elif "event=0x59" in event_spec:
            s.prefetch.fills += period
            total_fills += period

    return total_dispatched, total_ineff_dc, total_ineff_mab, total_fills


# -- Function-level aggregation ------------------------------------------------


def compute_function_summaries(stats: dict[str, InsnStats]) -> list[FunctionSummary]:
    """Aggregate per-instruction stats into per-function summaries, sorted by weighted cost."""
    by_func: dict[str, FunctionSummary] = {}
    for key, s in stats.items():
        func_name = key.split("\0", 1)[0]
        if func_name not in by_func:
            by_func[func_name] = FunctionSummary(
                name=func_name, total_samples=0, weighted_cost=0,
            )
        f = by_func[func_name]
        f.total_samples += s.total_samples
        for lvl, c in s.cache_counts.items():
            f.weighted_cost += c * lvl.weight
            f.cache_counts[lvl] += c
    return sorted(by_func.values(), key=lambda f: f.weighted_cost, reverse=True)


# -- Per-function objdump via address range ------------------------------------

_RUST_HASH_RE = re.compile(r"::h[0-9a-f]+$")


def get_function_bounds(binary: str) -> dict[int, int]:
    """Return {file_addr: size} for text symbols via nm -S."""
    result = subprocess.run(
        ["nm", "-S", binary],
        capture_output=True,
        text=True,
    )
    bounds: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] in ("t", "T"):
            try:
                addr = int(parts[0], 16)
                size = int(parts[1], 16)
                bounds[addr] = size
            except ValueError:
                continue
    return bounds


_MMAP_RE = re.compile(
    r"\[0x([0-9a-fA-F]+)\(0x[0-9a-fA-F]+\)\s+@\s+0x([0-9a-fA-F]+)\s+[^\]]*\]:\s+r-xp\s+(\S+)"
)


def compute_load_base(perf_data: str, binary: str) -> int:
    """Compute ASLR load base from perf MMAP records + ELF headers.

    Bulletproof: no symbol name matching, handles Rust monomorphizations.
    load_base = mmap_addr + p_offset - mmap_file_offset - p_vaddr
    """
    import os

    binary_name = os.path.basename(binary)

    # 1. Find the r-xp MMAP record for the binary in perf data
    result = subprocess.run(
        ["perf", "script", "--show-mmap-events", "-i", perf_data],
        capture_output=True,
        text=True,
    )
    mmap_addr = mmap_foff = None
    for line in result.stdout.splitlines():
        m = _MMAP_RE.search(line)
        if m and m.group(3).endswith(binary_name):
            mmap_addr = int(m.group(1), 16)
            mmap_foff = int(m.group(2), 16)
            break

    if mmap_addr is None:
        return 0

    # 2. Find the executable LOAD segment's p_offset and p_vaddr
    result = subprocess.run(
        ["readelf", "-lW", binary],
        capture_output=True,
        text=True,
    )
    p_offset = p_vaddr = None
    for line in result.stdout.splitlines():
        parts = line.split()
        if "LOAD" in parts and "E" in parts:
            idx = parts.index("LOAD")
            p_offset = int(parts[idx + 1], 16)
            p_vaddr = int(parts[idx + 2], 16)
            break

    if p_offset is None:
        return 0

    load_base = mmap_addr + p_offset - mmap_foff - p_vaddr

    import sys
    print(f"  load_base=0x{load_base:x}", file=sys.stderr)
    return load_base


def run_objdump_function(binary: str, start_addr: int, end_addr: int) -> str:
    """Disassemble a specific address range with source interleaving."""
    result = subprocess.run(
        [
            "objdump", "-d", "-S", "--no-show-raw-insn", "--demangle",
            f"--start-address=0x{start_addr:x}",
            f"--stop-address=0x{end_addr:x}",
            binary,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"objdump failed: {result.stderr.strip()}")
    return result.stdout
