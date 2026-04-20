from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto


class CacheLevel(Enum):
    L1 = auto()
    LFB = auto()
    L2 = auto()
    L3 = auto()
    DRAM = auto()
    REMOTE = auto()
    NON_MEMORY = auto()

    @property
    def label(self) -> str:
        return _CACHE_LABELS[self]

    @property
    def weight(self) -> int:
        """Approximate cost in cycles (Zen 4)."""
        return _CACHE_WEIGHTS[self]


_CACHE_LABELS = {
    CacheLevel.L1: "L1",
    CacheLevel.LFB: "LFB",
    CacheLevel.L2: "L2",
    CacheLevel.L3: "L3",
    CacheLevel.DRAM: "DRAM",
    CacheLevel.REMOTE: "REM",
    CacheLevel.NON_MEMORY: "N-M",
}

_CACHE_WEIGHTS = {
    CacheLevel.L1: 4,
    CacheLevel.LFB: 9,
    CacheLevel.L2: 14,
    CacheLevel.L3: 50,
    CacheLevel.DRAM: 250,
    CacheLevel.REMOTE: 400,
    CacheLevel.NON_MEMORY: 1,
}

# curses color pair index per cache level
CACHE_COLOR_PAIR = {
    CacheLevel.L1: 1,       # green
    CacheLevel.LFB: 1,      # green
    CacheLevel.L2: 2,       # yellow
    CacheLevel.L3: 2,       # yellow (bold for distinction)
    CacheLevel.DRAM: 3,     # red
    CacheLevel.REMOTE: 4,   # magenta
    CacheLevel.NON_MEMORY: 0,
}

CACHE_LEVELS = list(CacheLevel)


class TlbLevel(Enum):
    L1_HIT = auto()
    L2_HIT = auto()
    MISS = auto()
    NA = auto()


class OpType(Enum):
    LOAD = auto()
    STORE = auto()
    NA = auto()

    @property
    def label(self) -> str:
        return {OpType.LOAD: "LOAD", OpType.STORE: "STORE", OpType.NA: "N/A"}[self]


class SnoopStatus(Enum):
    NA = auto()
    NONE = auto()
    HIT = auto()
    HITM = auto()
    MISS = auto()

    @property
    def label(self) -> str:
        return {
            SnoopStatus.NA: "N/A",
            SnoopStatus.NONE: "None",
            SnoopStatus.HIT: "Hit",
            SnoopStatus.HITM: "HitM",
            SnoopStatus.MISS: "Miss",
        }[self]


class DisplayMode(Enum):
    WEIGHTED = "weighted"
    PERCENT = "percent"
    ABSOLUTE = "absolute"

    @property
    def label(self) -> str:
        return {
            DisplayMode.WEIGHTED: "weighted %",
            DisplayMode.PERCENT: "%",
            DisplayMode.ABSOLUTE: "count",
        }[self]


COL_WIDTH = 5


@dataclass
class IBSRaw:
    """Aggregated raw IBS register fields for a single instruction."""

    dc_miss_lat_sum: int = 0
    dc_miss_lat_count: int = 0
    tlb_refill_lat_sum: int = 0
    tlb_refill_lat_count: int = 0
    comp_to_ret_sum: int = 0
    comp_to_ret_count: int = 0
    tag_to_ret_sum: int = 0
    tag_to_ret_count: int = 0
    mabs_sum: int = 0
    mabs_count: int = 0
    mabs_max: int = 0
    mabs_max_count: int = 0
    sw_pf_count: int = 0
    misaligned_count: int = 0
    dc_miss_no_mab_count: int = 0
    dc_l1_tlb_miss_count: int = 0
    dc_l2_tlb_miss_count: int = 0
    dc_miss_count: int = 0
    l2_miss_count: int = 0
    mem_op_count: int = 0  # LdOp=1 or StOp=1
    mem_width_counts: dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    sample_count: int = 0

    def avg(self, total: int, count: int) -> float:
        return total / count if count else 0.0

    @property
    def avg_dc_miss_lat(self) -> float:
        return self.avg(self.dc_miss_lat_sum, self.dc_miss_lat_count)

    @property
    def avg_tlb_refill_lat(self) -> float:
        return self.avg(self.tlb_refill_lat_sum, self.tlb_refill_lat_count)

    @property
    def avg_comp_to_ret(self) -> float:
        return self.avg(self.comp_to_ret_sum, self.comp_to_ret_count)

    @property
    def avg_tag_to_ret(self) -> float:
        return self.avg(self.tag_to_ret_sum, self.tag_to_ret_count)

    @property
    def avg_mabs(self) -> float:
        return self.avg(self.mabs_sum, self.mabs_count)


@dataclass
class PrefetchCounts:
    """Per-instruction SW prefetch PMC counts (PMCx04B, PMCx052, PMCx059)."""

    dispatched: int = 0
    ineffective_dc_hit: int = 0
    ineffective_mab_match: int = 0
    fills: int = 0

    @property
    def total_ineffective(self) -> int:
        return self.ineffective_dc_hit + self.ineffective_mab_match

    @property
    def effective_rate(self) -> float:
        return self.fills / self.dispatched if self.dispatched else 0.0


@dataclass
class InsnStats:
    """Aggregated IBS statistics for a single instruction address."""

    cache_counts: dict[CacheLevel, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    tlb_counts: dict[TlbLevel, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    op_counts: dict[OpType, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    snoop_counts: dict[SnoopStatus, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    locked_count: int = 0
    total_weight: int = 0
    weight_count: int = 0
    total_samples: int = 0
    cycles: int = 0
    ibs: IBSRaw = field(default_factory=IBSRaw)
    prefetch: PrefetchCounts = field(default_factory=PrefetchCounts)

    def add(
        self,
        cache: CacheLevel,
        tlb: TlbLevel,
        op: OpType,
        snoop: SnoopStatus,
        locked: bool,
        weight: int,
    ) -> None:
        self.total_samples += 1
        self.cache_counts[cache] += 1
        self.tlb_counts[tlb] += 1
        self.op_counts[op] += 1
        self.snoop_counts[snoop] += 1
        if locked:
            self.locked_count += 1
        if weight > 0:
            self.total_weight += weight
            self.weight_count += 1

    @property
    def avg_weight(self) -> float:
        return self.total_weight / self.weight_count if self.weight_count else 0.0


@dataclass
class FunctionSummary:
    """Per-function aggregate for the function picker."""

    name: str
    total_samples: int
    weighted_cost: int
    cycles: int = 0
    cache_counts: dict[CacheLevel, int] = field(
        default_factory=lambda: defaultdict(int)
    )


# -- Annotated line types ------------------------------------------------------


@dataclass
class SourceLine:
    text: str


@dataclass
class FunctionHeader:
    name: str
    base_addr: int


@dataclass
class InstructionLine:
    addr: str
    offset: int
    disasm: str
    sym: str | None
    stats: InsnStats


@dataclass
class SeparatorLine:
    pass


AnnotatedLine = SourceLine | FunctionHeader | InstructionLine | SeparatorLine


# -- Skipped samples bookkeeping ----------------------------------------------


@dataclass
class SkippedSamples:
    unknown_symbol: dict[CacheLevel, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    unknown_level: int = 0

    def summary_lines(self) -> list[str]:
        total_sym = sum(self.unknown_symbol.values())
        if total_sym == 0 and self.unknown_level == 0:
            return []
        lines: list[str] = ["Skipped samples:"]
        if total_sym:
            lines.append(f"  Unresolved symbol: {total_sym}")
            for lvl in CACHE_LEVELS:
                c = self.unknown_symbol.get(lvl, 0)
                if c:
                    lines.append(f"    {lvl.label:>4}: {c}")
        if self.unknown_level:
            lines.append(f"  Unknown cache level: {self.unknown_level}")
        return lines
