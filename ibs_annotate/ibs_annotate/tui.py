from __future__ import annotations

import curses
from collections.abc import Callable
from typing import TYPE_CHECKING

from .model import (
    CACHE_COLOR_PAIR,
    CACHE_LEVELS,
    COL_WIDTH,
    CacheLevel,
    DisplayMode,
    FunctionHeader,
    FunctionSummary,
    InsnStats,
    InstructionLine,
    OpType,
    SeparatorLine,
    SnoopStatus,
    SourceLine,
    TlbLevel,
)

if TYPE_CHECKING:
    from .model import AnnotatedLine


def _safe_addstr(win, row: int, col: int, text: str, attr: int = 0) -> None:
    h, w = win.getmaxyx()
    if row < 0 or row >= h or col >= w:
        return
    text = text[: w - col - 1]
    if text:
        try:
            win.addstr(row, col, text, attr)
        except curses.error:
            pass


# -- Color setup ---------------------------------------------------------------

_PAIR_GREEN = 1
_PAIR_YELLOW = 2
_PAIR_RED = 3
_PAIR_MAGENTA = 4
_PAIR_CYAN = 5
_PAIR_CURSOR = 6


def _init_colors() -> None:
    curses.use_default_colors()
    curses.init_pair(_PAIR_GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(_PAIR_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(_PAIR_RED, curses.COLOR_RED, -1)
    curses.init_pair(_PAIR_MAGENTA, curses.COLOR_MAGENTA, -1)
    curses.init_pair(_PAIR_CYAN, curses.COLOR_CYAN, -1)
    curses.init_pair(_PAIR_CURSOR, curses.COLOR_BLACK, curses.COLOR_WHITE)


def _level_attr(lvl: CacheLevel) -> int:
    pair = CACHE_COLOR_PAIR.get(lvl, 0)
    attr = curses.color_pair(pair) if pair else curses.A_DIM
    if lvl == CacheLevel.L3:
        attr |= curses.A_BOLD
    return attr


# -- Header text ---------------------------------------------------------------

_HEADER = " ".join(f"{lvl.label:>{COL_WIDTH}}" for lvl in CACHE_LEVELS) + " \u2502 Disassembly"
_COL_TOTAL_W = (COL_WIDTH + 1) * len(CACHE_LEVELS)
_SEP_LINE = "\u2500" * _COL_TOTAL_W + "\u253c" + "\u2500" * 60
_PADDING = " " * _COL_TOTAL_W


class CursesTUI:
    def __init__(
        self,
        func_summaries: list[FunctionSummary],
        load_function: Callable[[str], list[AnnotatedLine]],
        mode: DisplayMode,
        total_unweighted: int,
        total_weighted: int,
        skipped_lines: list[str],
    ) -> None:
        self.func_summaries = func_summaries
        self.load_function = load_function
        self.mode = mode
        self.total_uw = total_unweighted
        self.total_w = total_weighted
        self.skipped = skipped_lines
        self.total_weighted_cost = sum(f.weighted_cost for f in func_summaries)

        # Function picker state
        self.func_cursor = 0
        self.func_scroll = 0

        # Annotated view state (populated on function selection)
        self.lines: list[AnnotatedLine] = []
        self.cursor = 0
        self.scroll = 0
        self._hottest_idx = 0

        self.page: str = "functions"
        self.prev_page: str = "functions"
        self.detail_line: InstructionLine | None = None
        self.detail_scroll = 0
        self.summary_scroll = 0
        self.search_query: str = ""
        self.search_active: bool = False

    def run(self) -> None:
        curses.wrapper(self._loop)

    # -- main loop -------------------------------------------------------------

    def _loop(self, stdscr) -> None:
        curses.curs_set(0)
        _init_colors()

        while True:
            stdscr.erase()
            if self.page == "functions":
                self._draw_functions(stdscr)
            elif self.page == "main":
                self._draw_main(stdscr)
            elif self.page == "detail":
                self._draw_detail(stdscr)
            elif self.page == "summary":
                self._draw_summary(stdscr)
            stdscr.refresh()

            key = stdscr.getch()

            # Search input mode
            if self.search_active:
                if key in (27,):  # Escape cancels
                    self.search_active = False
                elif key in (curses.KEY_ENTER, 10, 13):
                    self.search_active = False
                    self._search_next(forward=True)
                elif key in (curses.KEY_BACKSPACE, 127):
                    self.search_query = self.search_query[:-1]
                elif 32 <= key < 127:
                    self.search_query += chr(key)
                continue

            if not self._handle_key(key):
                break

    # -- key handling ----------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        if key == ord("q"):
            return False

        # Mode switching (both pages)
        if key == ord("w"):
            self.mode = DisplayMode.WEIGHTED
            return True
        if key == ord("p"):
            self.mode = DisplayMode.PERCENT
            return True
        if key == ord("n"):
            self.mode = DisplayMode.ABSOLUTE
            return True

        if self.page == "functions":
            return self._handle_functions_key(key)
        elif self.page == "main":
            return self._handle_main_key(key)
        elif self.page == "detail":
            return self._handle_detail_key(key)
        elif self.page == "summary":
            return self._handle_summary_key(key)
        return True

    def _handle_functions_key(self, key: int) -> bool:
        if key == curses.KEY_UP and self.func_cursor > 0:
            self.func_cursor -= 1
        elif key == curses.KEY_DOWN and self.func_cursor < len(self.func_summaries) - 1:
            self.func_cursor += 1
        elif key == curses.KEY_PPAGE:
            self.func_cursor = max(0, self.func_cursor - 30)
        elif key == curses.KEY_NPAGE:
            self.func_cursor = min(len(self.func_summaries) - 1, self.func_cursor + 30)
        elif key in (curses.KEY_ENTER, 10, 13):
            self._load_selected_function()
        elif key == ord("s") and self.skipped:
            self.summary_scroll = 0
            self.prev_page = "functions"
            self.page = "summary"
        return True

    def _load_selected_function(self) -> None:
        if not self.func_summaries:
            return
        func = self.func_summaries[self.func_cursor]
        self.lines = self.load_function(func.name)
        self.cursor = 0
        self.scroll = 0

        # Find hottest instruction
        self._hottest_idx = 0
        best = 0
        for i, line in enumerate(self.lines):
            if isinstance(line, InstructionLine):
                w = sum(c * lvl.weight for lvl, c in line.stats.cache_counts.items())
                if w > best:
                    best = w
                    self._hottest_idx = i

        self.page = "main"

    def _handle_main_key(self, key: int) -> bool:
        if key in (27,):  # Escape → back to function picker
            self.page = "functions"
            return True
        if not self.lines:
            return True
        if key == curses.KEY_UP and self.cursor > 0:
            self.cursor -= 1
        elif key == curses.KEY_DOWN and self.cursor < len(self.lines) - 1:
            self.cursor += 1
        elif key == curses.KEY_PPAGE:
            self.cursor = max(0, self.cursor - 30)
        elif key == curses.KEY_NPAGE:
            self.cursor = min(len(self.lines) - 1, self.cursor + 30)
        elif key == curses.KEY_HOME:
            self.cursor = 0
        elif key == curses.KEY_END:
            self.cursor = len(self.lines) - 1
        elif key in (curses.KEY_ENTER, 10, 13):
            line = self.lines[self.cursor]
            if isinstance(line, InstructionLine) and line.stats.total_samples > 0:
                self.detail_line = line
                self.detail_scroll = 0
                self.page = "detail"
        elif key == ord("s") and self.skipped:
            self.summary_scroll = 0
            self.prev_page = "main"
            self.page = "summary"
        elif key == ord("H"):
            self.cursor = self._hottest_idx
        elif key == ord("/"):
            self.search_active = True
            self.search_query = ""
        elif key == ord("N"):  # next match (after search)
            self._search_next(forward=True)
        elif key == ord("P"):  # prev match
            self._search_next(forward=False)
        return True

    def _handle_detail_key(self, key: int) -> bool:
        if key in (27, curses.KEY_BACKSPACE, 127, curses.KEY_LEFT):
            self.page = "main"
        elif key == curses.KEY_UP:
            self.detail_scroll = max(0, self.detail_scroll - 1)
        elif key == curses.KEY_DOWN:
            self.detail_scroll += 1
        elif key == curses.KEY_PPAGE:
            self.detail_scroll = max(0, self.detail_scroll - 30)
        elif key == curses.KEY_NPAGE:
            self.detail_scroll += 30
        return True

    def _handle_summary_key(self, key: int) -> bool:
        if key in (27, curses.KEY_BACKSPACE, 127, curses.KEY_LEFT, ord("s")):
            self.page = self.prev_page
        elif key == curses.KEY_UP:
            self.summary_scroll = max(0, self.summary_scroll - 1)
        elif key == curses.KEY_DOWN:
            self.summary_scroll += 1
        return True

    # -- search ----------------------------------------------------------------

    def _line_text(self, line: AnnotatedLine) -> str:
        if isinstance(line, InstructionLine):
            return f"{line.addr} {line.disasm}"
        elif isinstance(line, FunctionHeader):
            return line.name
        elif isinstance(line, SourceLine):
            return line.text
        return ""

    def _search_next(self, forward: bool = True) -> None:
        if not self.search_query:
            return
        q = self.search_query.lower()
        n = len(self.lines)
        step = 1 if forward else -1
        for i in range(1, n):
            idx = (self.cursor + i * step) % n
            if q in self._line_text(self.lines[idx]).lower():
                self.cursor = idx
                return

    # -- function picker drawing -------------------------------------------------

    def _draw_functions(self, stdscr) -> None:
        h, w = stdscr.getmaxyx()

        header = f"  {'Overhead':>8}  {'Samples':>8}  Function"
        _safe_addstr(stdscr, 0, 0, header, curses.A_BOLD)
        _safe_addstr(stdscr, 1, 0, "\u2500" * min(w - 1, 80))

        visible = h - 3

        if self.func_cursor < self.func_scroll:
            self.func_scroll = self.func_cursor
        if self.func_cursor >= self.func_scroll + visible:
            self.func_scroll = self.func_cursor - visible + 1

        for i in range(visible):
            idx = self.func_scroll + i
            if idx >= len(self.func_summaries):
                break
            f = self.func_summaries[idx]
            is_cursor = idx == self.func_cursor
            pct = 100.0 * f.weighted_cost / self.total_weighted_cost if self.total_weighted_cost else 0
            text = f"  {pct:>7.2f}%  {f.total_samples:>8}  {f.name}"
            attr = curses.color_pair(_PAIR_CURSOR) if is_cursor else 0
            _safe_addstr(stdscr, i + 2, 0, text.ljust(w - 1), attr)

        footer = " [Enter] annotate  [q]uit  [s]ummary"
        _safe_addstr(stdscr, h - 1, 0, footer.ljust(w - 1), curses.A_REVERSE)

    # -- main page drawing -----------------------------------------------------

    def _draw_main(self, stdscr) -> None:
        h, w = stdscr.getmaxyx()

        # Header
        _safe_addstr(stdscr, 0, 0, _HEADER, curses.A_BOLD)
        _safe_addstr(stdscr, 1, 0, _SEP_LINE)

        visible = h - 3  # header + sep + footer

        # Keep cursor visible
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        if self.cursor >= self.scroll + visible:
            self.scroll = self.cursor - visible + 1

        for i in range(visible):
            idx = self.scroll + i
            if idx >= len(self.lines):
                break
            self._draw_main_line(stdscr, i + 2, self.lines[idx], idx == self.cursor, w)

        # Footer
        if self.search_active:
            footer = f" /{self.search_query}\u2588"
        elif self.search_query:
            footer = (
                f" [{self.mode.label}]"
                f"  [q]uit  [/]search  [N]ext  [P]rev  [H]ottest  [Enter] detail  [s]ummary"
            )
        else:
            footer = (
                f" [{self.mode.label}]"
                "  [Esc] back  [q]uit  [/]search  [H]ottest  [w]eighted  [p]ercent  [n]umber  [Enter] detail  [s]ummary"
            )
        _safe_addstr(stdscr, h - 1, 0, footer.ljust(w - 1), curses.A_REVERSE)

    def _draw_main_line(self, stdscr, row: int, line: AnnotatedLine, is_cursor: bool, w: int) -> None:
        cursor_attr = curses.color_pair(_PAIR_CURSOR) if is_cursor else 0

        if isinstance(line, InstructionLine):
            x = 0
            for lvl in CACHE_LEVELS:
                c = line.stats.cache_counts.get(lvl, 0)
                if not c:
                    val = " " * COL_WIDTH
                    attr = cursor_attr
                else:
                    val = self._fmt_val(c, lvl)
                    attr = cursor_attr if is_cursor else _level_attr(lvl)
                _safe_addstr(stdscr, row, x, val, attr)
                x += COL_WIDTH
                _safe_addstr(stdscr, row, x, " ", cursor_attr)
                x += 1
            rest = f": {line.addr}:  {line.disasm}"
            _safe_addstr(stdscr, row, x, rest, cursor_attr)

        elif isinstance(line, FunctionHeader):
            text = f"{_PADDING} {line.name}:"
            attr = (curses.color_pair(_PAIR_CYAN) | curses.A_BOLD) if not is_cursor else cursor_attr
            _safe_addstr(stdscr, row, 0, text, attr)

        elif isinstance(line, SeparatorLine):
            _safe_addstr(stdscr, row, 0, _SEP_LINE, curses.A_DIM | cursor_attr)

        elif isinstance(line, SourceLine):
            text = f"{_PADDING}   {line.text}"
            _safe_addstr(stdscr, row, 0, text, curses.A_DIM | cursor_attr)

    def _fmt_val(self, count: int, lvl: CacheLevel) -> str:
        if self.mode == DisplayMode.ABSOLUTE:
            return f"{count:>{COL_WIDTH}}"
        elif self.mode == DisplayMode.PERCENT:
            return f"{100.0 * count / self.total_uw:>{COL_WIDTH}.1f}"
        else:
            return f"{100.0 * count * lvl.weight / self.total_w:>{COL_WIDTH}.1f}"

    # -- summary page drawing --------------------------------------------------

    def _draw_summary(self, stdscr) -> None:
        h, w = stdscr.getmaxyx()
        max_scroll = max(0, len(self.skipped) - (h - 1))
        self.summary_scroll = min(self.summary_scroll, max_scroll)

        for i, text in enumerate(self.skipped[self.summary_scroll:]):
            if i >= h - 1:
                break
            _safe_addstr(stdscr, i, 2, text)

        footer = " [Esc/s] back  [q]uit"
        _safe_addstr(stdscr, h - 1, 0, footer.ljust(w - 1), curses.A_REVERSE)

    # -- detail page drawing ---------------------------------------------------

    def _draw_detail(self, stdscr) -> None:
        h, w = stdscr.getmaxyx()
        assert self.detail_line is not None
        rows = self._build_detail_rows(self.detail_line)

        # Clamp scroll
        max_scroll = max(0, len(rows) - (h - 1))
        self.detail_scroll = min(self.detail_scroll, max_scroll)

        for i, (text, attr) in enumerate(rows[self.detail_scroll :]):
            if i >= h - 1:
                break
            _safe_addstr(stdscr, i, 0, text, attr)

        footer = f" [{self.mode.label}]  [Esc/\u2190] back  [q]uit  [\u2191\u2193] scroll"
        _safe_addstr(stdscr, h - 1, 0, footer.ljust(w - 1), curses.A_REVERSE)

    def _build_detail_rows(self, ln: InstructionLine) -> list[tuple[str, int]]:
        s = ln.stats
        rows: list[tuple[str, int]] = []
        B = curses.A_BOLD
        D = curses.A_DIM
        N = 0

        sym = f"{ln.sym}+0x{ln.offset:x}" if ln.sym else f"0x{ln.addr}"
        rows.append((f"  {sym}", B))
        rows.append((f"  {ln.disasm}", N))
        rows.append(("", N))
        rows.append((f"  Total samples: {s.total_samples}", N))
        rows.append(("", N))

        # -- Cache level -------------------------------------------------------
        rows.append(("  Cache Level        Count        %", B))
        rows.append(("  " + "\u2500" * 52, D))
        total_cache = sum(s.cache_counts.values())
        max_c = max(s.cache_counts.values(), default=0)
        for lvl in CACHE_LEVELS:
            c = s.cache_counts.get(lvl, 0)
            if not c:
                continue
            pct = 100.0 * c / total_cache if total_cache else 0
            bar = "\u2588" * (int(25 * c / max_c) if max_c else 0)
            rows.append((
                f"    {lvl.label:>4}  {c:>10}  {pct:6.1f}%  {bar}",
                _level_attr(lvl),
            ))
        rows.append(("", N))

        # -- TLB ---------------------------------------------------------------
        tlb_mem = {k: v for k, v in s.tlb_counts.items() if k != TlbLevel.NA}
        if tlb_mem:
            total_tlb = sum(tlb_mem.values())
            rows.append(("  TLB                                    ", B))
            rows.append(("  " + "\u2500" * 52, D))

            labels = {
                TlbLevel.L1_HIT: "L1 DTLB hit",
                TlbLevel.L2_HIT: "L1 miss, L2 hit",
                TlbLevel.MISS: "L1+L2 miss (walk)",
            }
            for tlb in (TlbLevel.L1_HIT, TlbLevel.L2_HIT, TlbLevel.MISS):
                c = tlb_mem.get(tlb, 0)
                if not c:
                    continue
                pct = 100.0 * c / total_tlb
                rows.append((f"    {labels[tlb]:<20s}  {c:>6}  {pct:6.1f}%", N))

            l1_miss = tlb_mem.get(TlbLevel.L2_HIT, 0) + tlb_mem.get(TlbLevel.MISS, 0)
            l2_miss = tlb_mem.get(TlbLevel.MISS, 0)
            rows.append(("", N))
            if total_tlb:
                rows.append((f"    L1 miss rate: {100.0 * l1_miss / total_tlb:5.1f}%  ({l1_miss}/{total_tlb})", N))
            l2_denom = tlb_mem.get(TlbLevel.L2_HIT, 0) + l2_miss
            if l2_denom:
                rows.append((f"    L2 miss rate: {100.0 * l2_miss / l2_denom:5.1f}%  ({l2_miss}/{l2_denom})", N))
            rows.append(("", N))

        # -- Snoop --------------------------------------------------------------
        snoop_interesting = {k: v for k, v in s.snoop_counts.items() if k not in (SnoopStatus.NA, SnoopStatus.NONE)}
        if snoop_interesting:
            rows.append(("  Snoop (cache coherency)", B))
            rows.append(("  " + "\u2500" * 52, D))
            for st in (SnoopStatus.HIT, SnoopStatus.HITM, SnoopStatus.MISS):
                c = snoop_interesting.get(st, 0)
                if c:
                    rows.append((f"    {st.label:<6s}  {c:>6}", N))
            rows.append(("", N))

        # -- Operations ---------------------------------------------------------
        ops_mem = {k: v for k, v in s.op_counts.items() if k != OpType.NA}
        if ops_mem:
            total_ops = sum(ops_mem.values())
            rows.append(("  Operations", B))
            rows.append(("  " + "\u2500" * 52, D))
            for op in (OpType.LOAD, OpType.STORE):
                c = ops_mem.get(op, 0)
                if c:
                    pct = 100.0 * c / total_ops
                    rows.append((f"    {op.label:<5s}  {c:>8}  {pct:6.1f}%", N))
            rows.append(("", N))

        # -- Latency (weight) ---------------------------------------------------
        if s.weight_count:
            rows.append(("  DC Miss Latency", B))
            rows.append(("  " + "\u2500" * 52, D))
            rows.append((f"    Avg: {s.avg_weight:.0f} cycles  ({s.weight_count} samples with latency)", N))
            rows.append(("", N))

        # -- Misc ---------------------------------------------------------------
        if s.locked_count:
            rows.append((f"  Locked ops: {s.locked_count}", N))
            rows.append(("", N))

        # -- Raw IBS data (from perf script -D) ---------------------------------
        ibs = s.ibs
        if ibs.sample_count > 0:
            rows.append(("  Latency (cycles)         Avg     Samples", B))
            rows.append(("  " + "\u2500" * 52, D))
            if ibs.tag_to_ret_count:
                rows.append((f"    Tag-to-retire     {ibs.avg_tag_to_ret:>8.0f}     {ibs.tag_to_ret_count:>6}", N))
            if ibs.comp_to_ret_count:
                rows.append((f"    Comp-to-retire    {ibs.avg_comp_to_ret:>8.0f}     {ibs.comp_to_ret_count:>6}", N))
            if ibs.dc_miss_lat_count:
                rows.append((
                    f"    DC miss latency   {ibs.avg_dc_miss_lat:>8.0f}     {ibs.dc_miss_lat_count:>6}",
                    curses.color_pair(_PAIR_RED),
                ))
            if ibs.tlb_refill_lat_count:
                rows.append((f"    TLB refill lat    {ibs.avg_tlb_refill_lat:>8.0f}     {ibs.tlb_refill_lat_count:>6}", N))
            rows.append(("", N))

            if ibs.mabs_count:
                rows.append(("  Memory-Level Parallelism", B))
                rows.append(("  " + "\u2500" * 52, D))
                rows.append((f"    Avg MABs in flight:  {ibs.avg_mabs:.1f}  (max {ibs.mabs_max}, {ibs.mabs_count} samples)", N))
                if ibs.dc_miss_no_mab_count:
                    rows.append((f"    MAB coalesced (hit existing): {ibs.dc_miss_no_mab_count}", N))
                rows.append(("", N))

            if ibs.mem_width_counts:
                rows.append(("  Access Width", B))
                rows.append(("  " + "\u2500" * 52, D))
                for w in sorted(ibs.mem_width_counts):
                    c = ibs.mem_width_counts[w]
                    rows.append((f"    {w:>3} bytes  {c:>8}", N))
                rows.append(("", N))

            # Raw IBS TLB bits (cross-check with data_src TLB above)
            if ibs.dc_l1_tlb_miss_count or ibs.dc_l2_tlb_miss_count:
                rows.append(("  TLB (raw IBS bits)", B))
                rows.append(("  " + "\u2500" * 52, D))
                rows.append((f"    DcL1TlbMiss:  {ibs.dc_l1_tlb_miss_count:>6}  / {ibs.sample_count}", N))
                rows.append((f"    DcL2TlbMiss:  {ibs.dc_l2_tlb_miss_count:>6}  / {ibs.sample_count}", N))
                rows.append(("", N))

            # Raw IBS cache bits
            if ibs.dc_miss_count or ibs.l2_miss_count:
                rows.append(("  Cache (raw IBS bits)", B))
                rows.append(("  " + "\u2500" * 52, D))
                rows.append((f"    DcMiss (L1):  {ibs.dc_miss_count:>6}  / {ibs.sample_count}", N))
                rows.append((f"    L2Miss:       {ibs.l2_miss_count:>6}  / {ibs.sample_count}", N))
                rows.append(("", N))

            misc_lines = []
            if ibs.sw_pf_count:
                misc_lines.append(f"SW prefetch: {ibs.sw_pf_count}")
            if ibs.misaligned_count:
                misc_lines.append(f"Misaligned: {ibs.misaligned_count}")
            if misc_lines:
                rows.append(("  " + "  |  ".join(misc_lines), N))
                rows.append(("", N))


        return rows
