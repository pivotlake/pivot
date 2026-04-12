from __future__ import annotations

import sys

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
    compute_totals,
    find_binary,
    parse_ibs_raw,
    parse_prefetch_pmc,
    parse_samples,
    run_objdump,
    run_perf,
)


def _print_stdout(lines, mode, total_uw, total_w, skipped_lines) -> None:
    padding = " ".join(" " * COL_WIDTH for _ in CACHE_LEVELS)

    for s in skipped_lines:
        print(f"# {s}")
    if skipped_lines:
        print()

    print(
        " ".join(f"{lvl.label:>{COL_WIDTH}}" for lvl in CACHE_LEVELS)
        + " \u2502 Disassembly"
    )
    print(
        "\u2500" * ((COL_WIDTH + 1) * len(CACHE_LEVELS))
        + "\u253c"
        + "\u2500" * 60
    )

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
                print(" ".join(cols) + f" : {line.addr}:  {line.disasm}")
            case FunctionHeader():
                print(f"\n{padding}   {line.name}:")
            case SeparatorLine():
                print(
                    "\u2500" * ((COL_WIDTH + 1) * len(CACHE_LEVELS))
                    + "\u253c"
                    + "\u2500" * 60
                )
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
def main(
    perf_data: str,
    absolute: bool,
    percent: bool,
    stdout: bool,
    binary: str | None,
) -> None:
    """AMD IBS cache-level annotator with interactive TUI."""
    if absolute:
        mode = DisplayMode.ABSOLUTE
    elif percent:
        mode = DisplayMode.PERCENT
    else:
        mode = DisplayMode.WEIGHTED

    click.echo("Parsing perf data...", err=True)
    script_output = run_perf(
        ["script", "-F", "ip,sym,symoff,data_src", "-G"], perf_data
    )

    stats, skipped, ip_to_key = parse_samples(script_output)
    total_uw, total_w = compute_totals(stats)

    click.echo("Parsing raw IBS registers...", err=True)
    parse_ibs_raw(perf_data, stats, ip_to_key)

    click.echo("Parsing prefetch PMC samples...", err=True)
    pf_disp, pf_dc, pf_mab, pf_fills = parse_prefetch_pmc(perf_data, stats, ip_to_key)

    if total_uw == 0:
        click.echo("No resolved samples found.", err=True)
        sys.exit(1)

    binary_path = binary or find_binary(perf_data)
    click.echo(f"Disassembling {binary_path}...", err=True)
    annotate_output = run_objdump(binary_path)

    lines = build_annotated_lines(annotate_output, stats, ip_to_key)
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

    if stdout or not sys.stdout.isatty():
        for s in cache_lines + pf_lines:
            print(f"# {s}")
        if cache_lines or pf_lines:
            print()
        _print_stdout(lines, mode, total_uw, total_w, skipped_lines)
    else:
        from .tui import CursesTUI

        tui = CursesTUI(lines, mode, total_uw, total_w, cache_lines + pf_lines + skipped_lines)
        tui.run()
