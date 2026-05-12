# ibs-annotate

Per-instruction cache-level profiler for AMD IBS. Combines IBS memory-level data with cycle samples to show exactly where time is spent and how each instruction interacts with the memory hierarchy.

## Install

```bash
cd ibs_annotate
poetry install
```

## Recording

Record with IBS and cycles simultaneously:

```bash
perf record -e cycles -e ibs_op// -c 100000 -a -- ./your_binary
```

For software prefetch analysis, add PMC events:

```bash
perf record -e cycles \
  -e ibs_op// -c 100000 \
  -e cpu/event=0x4b,umask=0x07/ \
  -e cpu/event=0x52,umask=0x01/ \
  -e cpu/event=0x52,umask=0x02/ \
  -e cpu/event=0x59,umask=0xff/ \
  -a -- ./your_binary
```

## Usage

### TUI (interactive)

```bash
ibs-annotate perf.data
```

Opens a function picker sorted by cycle overhead. Select a function with Enter to see per-instruction annotation.

### stdout

```bash
# Single function (fast - only disassembles what you need)
ibs-annotate perf.data --sym Probe

# Full binary
ibs-annotate perf.data --stdout
```

### Options

| Flag | Description |
|------|-------------|
| `--sym <name>` | Annotate a single function (substring match), print to stdout |
| `--stdout` | Print all functions to stdout |
| `-n` / `--absolute` | Show absolute sample counts instead of percentages |
| `-p` / `--percent` | Show unweighted percentages |
| `--binary <path>` | Path to binary (auto-detected from perf.data if omitted) |

## Columns

```
   L1   LFB    L2    L3  DRAM   REM   N-M   CYC | Disassembly
```

| Column | Source | What it shows |
|--------|--------|---------------|
| L1 | IBS data_src | L1 data cache hit |
| LFB | IBS data_src | Line fill buffer / MAB hit |
| L2 | IBS data_src | L2 cache hit |
| L3 | IBS data_src | L3 cache hit |
| DRAM | IBS data_src | Local DRAM access |
| REM | IBS data_src | Remote NUMA access |
| N-M | IBS data_src | Non-memory operation (register, ALU, branch) |
| CYC | cycles PMC | Cycle sample % - where wall-clock time is spent |

In default mode, cache columns show weighted percentages (higher cache levels weighted more). Use `-n` for raw counts, `-p` for unweighted percentages.

## TUI keybindings

### Function picker
| Key | Action |
|-----|--------|
| Enter | Annotate selected function |
| q | Quit |
| s | Summary (cache hierarchy, prefetch stats) |

### Annotation view
| Key | Action |
|-----|--------|
| Enter | Instruction detail (cache/TLB/IBS breakdown) |
| Esc | Back to function picker |
| H | Jump to hottest instruction |
| / | Search disassembly |
| N / P | Next / previous search match |
| w | Weighted % mode |
| p | Unweighted % mode |
| n | Absolute count mode |
| s | Summary |
| q | Quit |

## Requirements

- Linux with `perf`, `objdump`, `nm`, `readelf` in PATH
- AMD CPU with IBS support
- Binary built with debug info (`[profile.release] debug = true` for Rust)
