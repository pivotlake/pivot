"""Cache-funnel + dispatch-stall PMC summary.

Mirrors cf.sh: aggregates the well-known cache/stall PMC events from a
perf.data into a single overview, and (for token-stall events) attributes
samples per-instruction so the detail page can show per-line stall breakdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .model import InsnStats


# Order matters: longer/more specific names first so `dc_access_in_l2` doesn't
# get swallowed by a generic `l2_access` match etc.
_NAMED_FIELDS: list[tuple[str, str]] = [
    # Token stalls 1 (per-line eligible — see _STALL_LINE_NAMES)
    ("int_phy_reg_file_rsrc_stall", "stall1_int_phy_reg"),
    ("load_queue_rsrc_stall",       "stall1_load_q"),
    ("store_queue_rsrc_stall",      "stall1_store_q"),
    ("fp_reg_file_rsrc_stall",      "stall1_fp_reg"),
    ("fp_sch_rsrc_stall",           "stall1_fp_sch"),
    ("fp_flush_recovery_stall",     "stall1_fp_flush"),
    ("taken_brnch_buffer_rsrc",     "stall1_taken_br"),
    # Token stalls 2
    ("retire_token_stall",          "stall2_retire"),
    ("int_sch0_token_stall",        "stall2_int_sch0"),
    ("int_sch1_token_stall",        "stall2_int_sch1"),
    ("int_sch2_token_stall",        "stall2_int_sch2"),
    ("int_sch3_token_stall",        "stall2_int_sch3"),
    # Cache funnel
    ("l1-dcache-load-misses",       "l1_misses"),
    ("l1-dcache-loads",             "l1_loads"),
    ("dc_access_in_l2",             "l2_access"),
    ("dc_hit_in_l2",                "l2_hits"),
    ("ls_dmnd_fills_from_sys.local_ccx",   "dmnd_local_ccx"),
    ("ls_dmnd_fills_from_sys.near_cache",  "dmnd_near_cache"),
    ("ls_dmnd_fills_from_sys.far_cache",   "dmnd_far_cache"),
    ("ls_dmnd_fills_from_sys.dram_io_near","dmnd_dram_near"),
    ("ls_dmnd_fills_from_sys.dram_io_far", "dmnd_dram_far"),
    ("l2_pf_miss_l2_hit_l3",        "l2pf_hit_l3"),
    ("l2_pf_miss_l2_l3",            "l2pf_miss_l3"),
    ("ls_hw_pf_dc_fills.local_l2",      "hwpf_local_l2"),
    ("ls_hw_pf_dc_fills.local_ccx",     "hwpf_local_ccx"),
    ("ls_hw_pf_dc_fills.near_cache",    "hwpf_near_cache"),
    ("ls_hw_pf_dc_fills.far_cache",     "hwpf_far_cache"),
    ("ls_hw_pf_dc_fills.dram_io_near",  "hwpf_dram_near"),
    ("ls_hw_pf_dc_fills.dram_io_far",   "hwpf_dram_far"),
    ("ls_sw_pf_dc_fills.local_l2",      "swpf_local_l2"),
    ("ls_sw_pf_dc_fills.local_ccx",     "swpf_local_ccx"),
    ("ls_sw_pf_dc_fills.near_cache",    "swpf_near_cache"),
    ("ls_sw_pf_dc_fills.far_cache",     "swpf_far_cache"),
    ("ls_sw_pf_dc_fills.dram_io_near",  "swpf_dram_near"),
    ("ls_sw_pf_dc_fills.dram_io_far",   "swpf_dram_far"),
    ("ic_cache_fill_l2",            "ic_fill_l2"),
    ("ic_cache_fill_sys",           "ic_fill_sys"),
    # Dispatch slots
    ("de_no_dispatch_per_slot.backend_stalls",       "backend_stalls"),
    ("de_no_dispatch_per_slot.no_ops_from_frontend", "no_ops_from_frontend"),
    ("de_no_dispatch_per_slot.smt_contention",       "smt_contention"),
    ("de_op_queue_empty",           "op_queue_empty"),
    # UMC
    ("umc_cas_cmd.rd",              "umc_cas_rd"),
    ("umc_cas_cmd.wr",              "umc_cas_wr"),
]

# Friendly per-line label for each token-stall field key.
_STALL_LINE_NAMES: dict[str, str] = {
    "stall1_int_phy_reg": "Int reg file",
    "stall1_load_q":      "Load queue",
    "stall1_store_q":     "Store queue",
    "stall1_fp_reg":      "FP reg file",
    "stall1_fp_sch":      "FP scheduler",
    "stall1_fp_flush":    "FP flush",
    "stall1_taken_br":    "Branch buffer",
    "stall2_retire":      "Retire queue",
    "stall2_int_sch0":    "Int sched 0",
    "stall2_int_sch1":    "Int sched 1",
    "stall2_int_sch2":    "Int sched 2",
    "stall2_int_sch3":    "Int sched 3",
}


@dataclass
class PmcSummary:
    # Cache funnel
    l1_loads: int = 0
    l1_misses: int = 0
    l2_access: int = 0
    l2_hits: int = 0
    dmnd_local_ccx: int = 0
    dmnd_near_cache: int = 0
    dmnd_far_cache: int = 0
    dmnd_dram_near: int = 0
    dmnd_dram_far: int = 0
    l2pf_hit_l3: int = 0
    l2pf_miss_l3: int = 0
    hwpf_local_l2: int = 0
    hwpf_local_ccx: int = 0
    hwpf_near_cache: int = 0
    hwpf_far_cache: int = 0
    hwpf_dram_near: int = 0
    hwpf_dram_far: int = 0
    swpf_local_l2: int = 0
    swpf_local_ccx: int = 0
    swpf_near_cache: int = 0
    swpf_far_cache: int = 0
    swpf_dram_near: int = 0
    swpf_dram_far: int = 0
    ic_fill_l2: int = 0
    ic_fill_sys: int = 0
    # Token stalls
    stall1_int_phy_reg: int = 0
    stall1_load_q: int = 0
    stall1_store_q: int = 0
    stall1_fp_reg: int = 0
    stall1_fp_sch: int = 0
    stall1_fp_flush: int = 0
    stall1_taken_br: int = 0
    stall2_retire: int = 0
    stall2_int_sch0: int = 0
    stall2_int_sch1: int = 0
    stall2_int_sch2: int = 0
    stall2_int_sch3: int = 0
    # Dispatch / cycles / uncore
    backend_stalls: int = 0
    no_ops_from_frontend: int = 0
    smt_contention: int = 0
    op_queue_empty: int = 0
    bitcoin: int = 0  # raw cpu/event=0xAE,umask=0x08/
    cycles: int = 0
    umc_cas_rd: int = 0
    umc_cas_wr: int = 0

    # Set of field keys that received any sample — used to decide if the
    # summary is worth rendering.
    seen: set[str] = field(default_factory=set)

    def has_data(self) -> bool:
        return bool(self.seen)


def _classify(eid: str) -> str | None:
    """Map a perf-script event identifier to a PmcSummary field key."""
    e = eid.lower()
    for needle, key in _NAMED_FIELDS:
        if needle in e:
            return key
    if "event=0xae" in e and "umask=0x08" in e:
        return "bitcoin"
    # `cycles`, `cycles:u`, `cycles:k` … but not `cycles_not_in_halt`.
    if e == "cycles" or e.startswith("cycles:") or e.startswith("cycles/"):
        return "cycles"
    return None


# IP+sym+offset (sample location). Searched anywhere on the line — perf
# may print the period either before or after this pair. Optional on uncore
# events that have no thread context.
_IP_SYM_TAIL = re.compile(
    r"\b([0-9a-fA-F]{4,})\s+(\S+)\+0x([0-9a-fA-F]+)\b"
)
# Event-id token: a colon-suffixed word, terminated by whitespace or EOL.
# Disallow `:` inside the token so we don't gobble the timestamp `1234.5:`.
_EVENT_TOK = re.compile(r"(\S+?):(?=\s|$)")
# Pure-decimal token (period or freq). `\b` ensures we don't grab digits
# embedded in hex words like `deadbeef` or `0x10`.
_NUM_TOK = re.compile(r"(?<!\w)(\d+)(?!\w)")


def parse_pmc_summary(
    script_output: str,
    stats: dict[str, InsnStats],
) -> PmcSummary:
    """Aggregate cf.sh-style PMC events; attribute token stalls per-instruction.

    The perf-script output field order is fixed by perf, not by `-F`, and
    differs across versions (period may print before or after the event,
    timestamp may be present, comm/tid may sneak back in for some events).
    Parse defensively: locate the event id by colon-suffixed token, the
    IP+sym at line end, and the period as the lone decimal token in the
    remainder. Skip lines we can't classify rather than throwing.
    """
    s = PmcSummary()
    matched = unmatched = 0
    samples: list[str] = []  # first few matched (event, period) for debug

    for line in script_output.splitlines():
        if not line.strip() or line.startswith("#"):
            continue

        # 1. Event id — first colon-suffixed token that classifies.
        eid = None
        eid_span = (0, 0)
        key = None
        for m in _EVENT_TOK.finditer(line):
            tok = m.group(1)
            k = _classify(tok)
            if k is not None:
                eid = tok
                eid_span = m.span()
                key = k
                break
        if key is None:
            continue

        # 2. IP+sym anywhere on the line (period may print before or after it).
        tail = _IP_SYM_TAIL.search(line)

        # 3. Period — a decimal token anywhere in the line that's not part of
        #    the event id, the IP+sym, or a hex IP itself. Mask those spans
        #    out before scanning. If multiple decimal tokens remain (e.g.
        #    timestamp `1234.5` splits into small parts), pick the largest:
        #    real periods for PMC events are typically ≥1k.
        masked = list(line)
        for lo, hi in [eid_span] + ([tail.span()] if tail else []):
            for j in range(lo, hi):
                masked[j] = " "
        nums = [int(n) for n in _NUM_TOK.findall("".join(masked))]
        if not nums:
            unmatched += 1
            continue
        # Drop tiny tokens that are obviously CPU ids (≤ 4 digits is suspect
        # for a frequency-mode period; counts are usually 5-8 digits). Fall
        # back to max if nothing remains.
        big = [n for n in nums if n >= 1000]
        period = max(big) if big else max(nums)

        setattr(s, key, getattr(s, key) + period)
        s.seen.add(key)
        matched += 1
        if len(samples) < 5:
            samples.append(f"{eid}={period}")

        stall_label = _STALL_LINE_NAMES.get(key)
        if stall_label and tail:
            ikey = f"{tail.group(2)}\0{int(tail.group(3), 16)}"
            if ikey not in stats:
                stats[ikey] = InsnStats()
            stats[ikey].token_stalls[stall_label] += period

    import sys
    if matched or unmatched:
        print(
            f"  PMC summary: {matched} matched, {unmatched} unmatched"
            + (f" (sample: {', '.join(samples)})" if samples else ""),
            file=sys.stderr,
        )

    return s


# ── Rendering (cf.sh-equivalent text output) ───────────────────────────────


def _fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}" if d else "0.0"


def _gb(n: int) -> str:
    return f"{n * 64 / 1_073_741_824:.1f}"


def render_pmc_summary(s: PmcSummary) -> list[str]:
    if not s.has_data():
        return []

    out: list[str] = []
    L = out.append

    L("Cache Hierarchy Funnel")

    # L1
    l1_hits = s.l1_loads - s.l1_misses
    if s.l1_loads:
        L(f"  L1 Data Cache         {_fmt(s.l1_loads)} accesses")
        L(f"    HIT  {_fmt(l1_hits):>8}  ({_pct(l1_hits, s.l1_loads)}%)")
        L(f"    MISS {_fmt(s.l1_misses):>8}  ({_pct(s.l1_misses, s.l1_loads)}%)")

        hwpf_l3 = s.hwpf_local_ccx + s.hwpf_near_cache + s.hwpf_far_cache
        hwpf_dram = s.hwpf_dram_near + s.hwpf_dram_far
        hwpf_total = s.hwpf_local_l2 + hwpf_l3 + hwpf_dram
        if hwpf_total:
            L(f"    HW PF {_fmt(hwpf_total)} into L1  "
              f"(L2={_fmt(s.hwpf_local_l2)}, L3/CCX={_fmt(hwpf_l3)}, DRAM={_fmt(hwpf_dram)})")

        swpf_l3 = s.swpf_local_ccx + s.swpf_near_cache + s.swpf_far_cache
        swpf_dram = s.swpf_dram_near + s.swpf_dram_far
        swpf_total = s.swpf_local_l2 + swpf_l3 + swpf_dram
        if swpf_total:
            L(f"    SW PF {_fmt(swpf_total)} into L1  "
              f"(L2={_fmt(s.swpf_local_l2)}, L3/CCX={_fmt(swpf_l3)}, DRAM={_fmt(swpf_dram)})")

    # L2
    l2_misses = s.l2_access - s.l2_hits
    if s.l2_access:
        L(f"  L2 Cache              {_fmt(s.l2_access)} accesses (excl. L2 prefetch)")
        L(f"    HIT  {_fmt(s.l2_hits):>8}  ({_pct(s.l2_hits, s.l2_access)}%)")
        L(f"    MISS {_fmt(l2_misses):>8}  ({_pct(l2_misses, s.l2_access)}%)")

    # L3
    l3_dmnd_hits = s.dmnd_local_ccx + s.dmnd_near_cache + s.dmnd_far_cache
    dmnd_dram = s.dmnd_dram_near + s.dmnd_dram_far
    l3_dmnd_total = l3_dmnd_hits + dmnd_dram
    if l3_dmnd_total:
        L(f"  L3 Cache              {_fmt(l3_dmnd_total)} demand accesses")
        L(f"    HIT  {_fmt(l3_dmnd_hits):>8}  ({_pct(l3_dmnd_hits, l3_dmnd_total)}%)")
        if s.dmnd_near_cache or s.dmnd_far_cache:
            L(f"      local CCX:  {_fmt(s.dmnd_local_ccx)}")
            L(f"      near CCX:   {_fmt(s.dmnd_near_cache)}")
            L(f"      far CCX:    {_fmt(s.dmnd_far_cache)}")
        L(f"    MISS {_fmt(dmnd_dram):>8}  ({_pct(dmnd_dram, l3_dmnd_total)}%)")
        if s.dmnd_dram_far:
            L(f"      near DRAM:  {_fmt(s.dmnd_dram_near)}")
            L(f"      far DRAM:   {_fmt(s.dmnd_dram_far)}")
        l2pf_total = s.l2pf_hit_l3 + s.l2pf_miss_l3
        if l2pf_total:
            L(f"    L2 PF {_fmt(l2pf_total)}  (hit L3={_fmt(s.l2pf_hit_l3)}, miss→DRAM={_fmt(s.l2pf_miss_l3)})")

    # DRAM
    hwpf_dram = s.hwpf_dram_near + s.hwpf_dram_far
    swpf_dram = s.swpf_dram_near + s.swpf_dram_far
    core_dram = dmnd_dram + s.l2pf_miss_l3 + hwpf_dram + swpf_dram + s.ic_fill_sys
    if core_dram:
        L(f"  DRAM                  {_fmt(core_dram)} core read fills (~{_gb(core_dram)} GB @ 64B)")
        L(f"    demand:  {_gb(dmnd_dram)} GB ({_fmt(dmnd_dram)})")
        L(f"    L2 pf:   {_gb(s.l2pf_miss_l3)} GB ({_fmt(s.l2pf_miss_l3)})")
        L(f"    HW pf:   {_gb(hwpf_dram)} GB ({_fmt(hwpf_dram)})")
        L(f"    SW pf:   {_gb(swpf_dram)} GB ({_fmt(swpf_dram)})")
        L(f"    IC:      {_gb(s.ic_fill_sys)} GB ({_fmt(s.ic_fill_sys)})")

    if s.umc_cas_rd or s.umc_cas_wr:
        umc_total = s.umc_cas_rd + s.umc_cas_wr
        L(f"  UMC                   reads={_fmt(s.umc_cas_rd)} ({_gb(s.umc_cas_rd)} GB)  "
          f"writes={_fmt(s.umc_cas_wr)} ({_gb(s.umc_cas_wr)} GB)  total={_gb(umc_total)} GB")
        if s.umc_cas_rd:
            L(f"    core reads / UMC reads: {_pct(core_dram, s.umc_cas_rd)}%")

    # Dispatch stall analysis
    cycles = s.cycles
    be_items = [
        ("Int reg file",   s.stall1_int_phy_reg),
        ("Load queue",     s.stall1_load_q),
        ("Store queue",    s.stall1_store_q),
        ("FP reg file",    s.stall1_fp_reg),
        ("FP scheduler",   s.stall1_fp_sch),
        ("FP flush",       s.stall1_fp_flush),
        ("Branch buffer",  s.stall1_taken_br),
        ("Retire queue",   s.stall2_retire),
        ("Int sched 0",    s.stall2_int_sch0),
        ("Int sched 1",    s.stall2_int_sch1),
        ("Int sched 2",    s.stall2_int_sch2),
        ("Int sched 3",    s.stall2_int_sch3),
        ("Bitcoin (0xAE/08)", s.bitcoin),
    ]
    fe_items = [
        ("No ops from FE", s.no_ops_from_frontend),
        ("Op queue empty", s.op_queue_empty),
        ("SMT contention", s.smt_contention),
    ]
    has_stalls = s.backend_stalls or any(v for _, v in be_items) or any(v for _, v in fe_items)
    if has_stalls and cycles:
        L("")
        L("Dispatch Stall Analysis")
        be_pct = _pct(s.backend_stalls, cycles)
        L(f"  Backend stalls:  {_fmt(s.backend_stalls)}  ({be_pct}% of cycles)")
        be_other = s.backend_stalls
        for name, v in be_items:
            if v and s.backend_stalls and (100.0 * v / s.backend_stalls) >= 5.0:
                L(f"    {name:<18s}  {_fmt(v):>8}  ({_pct(v, s.backend_stalls)}%)")
                be_other -= v
        if be_other > 0 and s.backend_stalls:
            L(f"    {'Other':<18s}  {_fmt(be_other):>8}  ({_pct(be_other, s.backend_stalls)}%)")
        if any(v for _, v in fe_items):
            L("  Frontend:")
            for name, v in fe_items:
                if v:
                    L(f"    {name:<18s}  {_fmt(v):>8}  ({_pct(v, cycles)}% of cycles)")

    return out
