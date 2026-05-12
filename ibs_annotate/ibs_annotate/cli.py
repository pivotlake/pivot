from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import click

from .model import (
    CACHE_LEVELS,
    COL_WIDTH,
    DisplayMode,
    FunctionHeader,
    InstructionLine,
    SeparatorLine,
    SourceLine,
)
from .parse import (
    build_annotated_lines,
    compute_function_summaries,
    compute_load_base,
    compute_totals,
    find_binary,
    get_function_bounds,
    parse_cycles,
    parse_ibs_raw,
    parse_prefetch_pmc,
    parse_samples,
    run_objdump,
    run_objdump_function,
    run_perf,
)
from .pmc_summary import parse_pmc_summary, render_pmc_summary


def _print_stdout(lines, mode, total_uw, total_w, total_cycles, skipped_lines) -> None:
    ncols = len(CACHE_LEVELS) + 1  # +1 for CYC
    padding = " ".join(" " * COL_WIDTH for _ in range(ncols))

    for s in skipped_lines:
        print(f"# {s}")
    if skipped_lines:
        print()

    print(
        " ".join(f"{lvl.label:>{COL_WIDTH}}" for lvl in CACHE_LEVELS)
        + f" {'CYC':>{COL_WIDTH}}"
        + " \u2502 Disassembly"
    )
    sep = "\u2500" * ((COL_WIDTH + 1) * ncols) + "\u253c" + "\u2500" * 60
    print(sep)

    for line in lines:
        match line:
            case InstructionLine():
                cols = []
                for lvl in CACHE_LEVELS:
                    c = line.stats.cache_counts.get(lvl, 0)
                    if not c:
                        cols.append(" " * COL_WIDTH)
                    elif mode == DisplayMode.ABSOLUTE:
                        cols.append(f"{c:>{COL_WIDTH}}")
                    elif mode == DisplayMode.PERCENT:
                        cols.append(f"{100.0 * c / total_uw:>{COL_WIDTH}.1f}")
                    else:
                        cols.append(f"{100.0 * c * lvl.weight / total_w:>{COL_WIDTH}.1f}")
                # Cycles column
                cyc = line.stats.cycles
                if not cyc:
                    cols.append(" " * COL_WIDTH)
                elif mode == DisplayMode.ABSOLUTE:
                    cols.append(f"{cyc:>{COL_WIDTH}}")
                elif total_cycles:
                    cols.append(f"{100.0 * cyc / total_cycles:>{COL_WIDTH}.1f}")
                else:
                    cols.append(" " * COL_WIDTH)
                print(" ".join(cols) + f" : {line.addr}:  {line.disasm}")
            case FunctionHeader():
                print(f"\n{padding}   {line.name}:")
            case SeparatorLine():
                print(sep)
            case SourceLine():
                print(f"{padding}   {line.text}")


@click.command()
@click.argument("perf_data", default="perf.data", type=click.Path(exists=True))
@click.option("-n", "--absolute", is_flag=True, help="Show absolute sample counts.")
@click.option("-p", "--percent", is_flag=True, help="Show unweighted percentages.")
@click.option(
    "--stdout", is_flag=True, help="Print to stdout instead of launching TUI."
)
@click.option(
    "--binary",
    type=click.Path(exists=True),
    default=None,
    help="Path to binary to disassemble (auto-detected from perf.data if omitted).",
)
@click.option(
    "--sym",
    default=None,
    help="Function name to annotate (substring match). Implies --stdout.",
)
def main(
    perf_data: str,
    absolute: bool,
    percent: bool,
    stdout: bool,
    binary: str | None,
    sym: str | None,
) -> None:
    """AMD IBS cache-level annotator with interactive TUI."""
    if absolute:
        mode = DisplayMode.ABSOLUTE
    elif percent:
        mode = DisplayMode.PERCENT
    else:
        mode = DisplayMode.WEIGHTED

    click.echo("Parsing perf data...", err=True)

    # Launch all perf processes in parallel so their I/O overlaps.
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
        ["perf", "script", "-F", "comm,event,ip,sym,symoff",
         "-i", perf_data],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    # Drain small outputs in background (prevents pipe buffer stall)
    drain_pool = ThreadPoolExecutor(max_workers=2)
    pmc_drain = drain_pool.submit(proc_pmc.communicate)
    cycles_drain = drain_pool.submit(proc_cycles.communicate)

    # Stage 1: parse IBS samples (while other processes run in background)
    script_output, script_err = proc_samples.communicate()
    if proc_samples.returncode != 0:
        raise RuntimeError(f"perf script failed: {script_err.strip()}")
    stats, skipped, ip_to_key = parse_samples(script_output)
    total_uw, total_w = compute_totals(stats)

    if total_uw == 0:
        proc_ibs.kill()
        proc_pmc.kill()
        proc_cycles.kill()
        click.echo("No resolved samples found.", err=True)
        sys.exit(1)

    # Stages 2-4: parse IBS raw, PMC, and cycles in parallel
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

    # cf.sh-style cache funnel + dispatch stalls (any subset of the events
    # listed in cf.sh; absent events just leave their fields zero).
    pmc_sum = parse_pmc_summary(pmc_output, stats)

    # Global per-label token-stall totals (denominator for percent modes)
    stall_totals: dict[str, int] = {}
    for s in stats.values():
        for label, c in s.token_stalls.items():
            stall_totals[label] = stall_totals.get(label, 0) + c

    binary_path = binary or find_binary(perf_data, ip_to_key)
    func_summaries = compute_function_summaries(stats)
    load_base = compute_load_base(perf_data, binary_path)
    func_bounds = get_function_bounds(binary_path)

    skipped_lines = skipped.summary_lines()

    # Cache hierarchy breakdown from data_src
    from .model import CacheLevel
    total_l1 = sum(s.cache_counts.get(CacheLevel.L1, 0) for s in stats.values())
    total_lfb = sum(s.cache_counts.get(CacheLevel.LFB, 0) for s in stats.values())
    total_l2 = sum(s.cache_counts.get(CacheLevel.L2, 0) for s in stats.values())
    total_l3 = sum(s.cache_counts.get(CacheLevel.L3, 0) for s in stats.values())
    total_dram = sum(s.cache_counts.get(CacheLevel.DRAM, 0) for s in stats.values())
    total_remote = sum(s.cache_counts.get(CacheLevel.REMOTE, 0) for s in stats.values())
    total_mem = total_l1 + total_lfb + total_l2 + total_l3 + total_dram + total_remote
    l1_misses = total_mem - total_l1

    cache_lines: list[str] = []
    if total_mem:
        cache_lines.append("Cache Hierarchy Summary")
        cache_lines.append(f"  Total memory samples: {total_mem}")
        cache_lines.append(f"  L1 hit:   {total_l1:>10}  ({100.0 * total_l1 / total_mem:5.1f}%)")
        if l1_misses:
            cache_lines.append(f"  L1 miss:  {l1_misses:>10}  ({100.0 * l1_misses / total_mem:5.1f}%)")
            cache_lines.append(f"    of L1 misses:")
            if total_lfb:
                cache_lines.append(f"      LFB hit:  {total_lfb:>8}  ({100.0 * total_lfb / l1_misses:5.1f}%)")
            if total_l2:
                cache_lines.append(f"      L2 hit:   {total_l2:>8}  ({100.0 * total_l2 / l1_misses:5.1f}%)")
            if total_l3:
                cache_lines.append(f"      L3 hit:   {total_l3:>8}  ({100.0 * total_l3 / l1_misses:5.1f}%)")
            if total_dram:
                cache_lines.append(f"      DRAM:     {total_dram:>8}  ({100.0 * total_dram / l1_misses:5.1f}%)")
            if total_remote:
                cache_lines.append(f"      Remote:   {total_remote:>8}  ({100.0 * total_remote / l1_misses:5.1f}%)")
        cache_lines.append("")

    funnel_lines = render_pmc_summary(pmc_sum)
    if funnel_lines:
        funnel_lines.append("")

    pf_lines: list[str] = []
    if pf_disp:
        pf_total_ineff = pf_dc + pf_mab
        pf_dropped = pf_disp - pf_fills - pf_total_ineff
        pf_lines.append("SW Prefetch Summary (PMC samples)")
        pf_lines.append(f"  Dispatched:    {pf_disp:>10}")
        pf_lines.append(f"  Filled:        {pf_fills:>10}  ({100.0 * pf_fills / pf_disp:5.1f}%)")
        pf_lines.append(f"  Ineffective:   {pf_total_ineff:>10}  ({100.0 * pf_total_ineff / pf_disp:5.1f}%)")
        if pf_dc:
            pf_lines.append(f"    DC hit (redundant):    {pf_dc:>8}")
        if pf_mab:
            pf_lines.append(f"    MAB match (in-flight): {pf_mab:>8}")
        if pf_dropped > 0:
            pf_lines.append(f"  Dropped/other: {pf_dropped:>10}  ({100.0 * pf_dropped / pf_disp:5.1f}%)")

    def load_function(func_name: str) -> list:
        """Load annotated lines for a single function via objdump."""
        # Find runtime base from perf data, convert to file address
        for ip, key in ip_to_key.items():
            sym, offset_s = key.split("\0", 1)
            if sym == func_name:
                runtime_base = ip - int(offset_s)
                file_addr = runtime_base - load_base
                size = func_bounds.get(file_addr, 0x10000)
                out = run_objdump_function(binary_path, file_addr, file_addr + size)
                return build_annotated_lines(out, stats, ip_to_key, load_base)
        return []

    if sym:
        # Find matching function name (substring match)
        matches = [f for f in func_summaries if sym in f.name]
        if not matches:
            click.echo(f"No function matching '{sym}'", err=True)
            sys.exit(1)
        if len(matches) > 1:
            click.echo(f"Multiple matches for '{sym}':", err=True)
            for f in matches[:10]:
                cyc_pct = 100.0 * f.cycles / total_cycles if total_cycles else 0
                click.echo(f"  {cyc_pct:>6.2f}%  {f.name}", err=True)
            sys.exit(1)
        lines = load_function(matches[0].name)
        for s in funnel_lines + cache_lines + pf_lines:
            print(f"# {s}")
        if funnel_lines or cache_lines or pf_lines:
            print()
        _print_stdout(lines, mode, total_uw, total_w, total_cycles, skipped_lines)
    elif stdout or not sys.stdout.isatty():
        click.echo(f"Disassembling {binary_path}...", err=True)
        annotate_output = run_objdump(binary_path)
        lines = build_annotated_lines(annotate_output, stats, ip_to_key, load_base)
        for s in funnel_lines + cache_lines + pf_lines:
            print(f"# {s}")
        if funnel_lines or cache_lines or pf_lines:
            print()
        _print_stdout(lines, mode, total_uw, total_w, total_cycles, skipped_lines)
    else:
        from .tui import CursesTUI

        tui = CursesTUI(
            func_summaries, load_function, mode, total_uw, total_w,
            total_cycles, funnel_lines + cache_lines + pf_lines + skipped_lines,
            stall_totals=stall_totals,
        )
        tui.run()
