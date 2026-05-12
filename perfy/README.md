# perfy

In-house perf.data analysis UI, modeled on the Firefox profiler / Samply
flow but tailored for our dispatch runs and AMD IBS data.

The backend is a Rust workspace:

- **`crates/ibs-annotate`** — reads `perf.data` files **directly** as binary
  via [`linux-perf-data`](https://crates.io/crates/linux-perf-data); no
  `perf script` subprocess. Decodes the `PERF_SAMPLE_DATA_SRC` bitfield
  (`data_src.rs`) and the AMD `IBS_OP_DATA{,3}` MSRs out of the raw payload
  (`ibs_msrs.rs`) ourselves so we can populate the same `InsnStats` shape
  the Python tool produced. Symbols come from
  [`object::SymbolMap`](https://docs.rs/object) on the binaries named by
  the perf MMAP2 records, with Rust + Itanium-C++ demangling.
- **`crates/perfy-server`** — `axum` HTTP backend that drives `ibs-annotate`
  at startup, builds an in-memory `Profile` (interned call stacks +
  per-instruction stats), and serves the analysis API.

The frontend is unchanged React + Vite + TS.

## Layout

```
perfy/
├── Cargo.toml                workspace root
├── crates/
│   ├── ibs-annotate/         Rust port of ibs_annotate
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs      enums, IBSRaw, InsnStats, JumpGraph, …
│   │       ├── parse.rs      perf script / objdump / nm / readelf parsing
│   │       ├── pmc_summary.rs cf.sh-equivalent cache-funnel summary
│   │       └── cmd.rs        subprocess wrappers
│   └── perfy-server/
│       └── src/
│           ├── main.rs       CLI: `perfy serve perf.data`
│           ├── parser.rs     perf script → Profile (with rayon)
│           ├── profile.rs    Profile, Sample, FrameTable, Category
│           ├── tracks.rs     time bucketing per (category, [cpu])
│           ├── flamegraph.rs callchain → tree
│           ├── annotate.rs   source + asm via ibs-annotate
│           └── api.rs        axum routes
└── frontend/                 React + Vite app (unchanged)
```

## Why native parsing

The original Python `ibs_annotate` shells out to `perf script` and parses
its text. That has two real costs: text parsing dominates wall time on
big captures, and the format drifts across perf versions (right-padded
`comm` fields, hex periods after the event name, optional `(dso)` tails,
…). The Rust path skips `perf script` entirely:

1. **`linux-perf-data` reads the binary `perf.data` file** — file header,
   attribute records, MMAP2 / COMM / FORK / SAMPLE records — and hands us
   typed structs.
2. **Custom `PERF_RECORD_SAMPLE` decoder** (`ibs-annotate/src/sample.rs`)
   exposes `data_src` and `weight`, which `linux-perf-event-reader` 0.10
   reads off the wire but discards.
3. **Bitfield decoders we own**: `data_src.rs` translates the
   `perf_mem_data_src` u64 into our `(CacheLevel, TlbLevel, OpType,
   SnoopStatus, locked)` tuple — same mapping the Python regex did,
   without the regex. Honours `mem_lvl_num` (modern) first, falls back
   to `mem_lvl` (legacy).
4. **AMD IBS MSR decoder** (`ibs_msrs.rs`): walks the
   `[u32 size][u32 caps][u64 regs[]]` raw payload emitted by the kernel's
   ibs_op event, decodes `IBS_OP_DATA` (CompToRet, BrnRet, …) and
   `IBS_OP_DATA3` (DcMissLat, TlbRefillLat, OpMemWidth, …) per the AMD
   PPR layout. Folds into the existing `IBSRaw` struct.
5. **Symbol resolution** (`symbols.rs`): per-pid mmap2 layout from the
   perf records → `object::SymbolMap` for ELF symbol lookup → Rust + C++
   demangling. No `perf script` symbol resolution involved.

Outcome: one process, one binary read, every datum from the original
text pipeline available, none of the format-drift fragility.

## Recording

```bash
perf record -e cycles -e ibs_op// -c 100000 -a -- ./your_binary
```

For pure cycle profiles (only the Cycles track will show up):

```bash
perf record -e cycles -F 999 --call-graph dwarf -- ./your_binary
```

## Build & run

```bash
cd perfy
cargo build --release -p perfy-server          # produces target/release/perfy
( cd frontend && npm install )

# In one terminal:
./target/release/perfy serve /path/to/perf.data
# Loaded N samples across K CPUs (cycles, l1, l2, l3, dram) in 1.23s.
# Serving on http://127.0.0.1:5005

# In another:
cd frontend
npm run dev
# http://localhost:5173
```

Vite proxies `/api/*` to the backend so just open the frontend URL.

`perfy dump-meta perf.data` prints the parsed metadata as JSON without
starting the HTTP server — handy for sanity-checking a new perf.data.

## API

- `GET /api/meta` — perf.data path, binary, CPUs, categories, time
  range, sample counts.
- `GET /api/tracks?categories=cycles,dram,l1&consolidated=true&buckets=1500`
  — per-track time-bucketed sample counts.
- `GET /api/flamegraph?track=cat:cycles:cpu:5` — callchain tree for a
  track. Or pass `?categories=…&cpus=…&time_lo_ns=&time_hi_ns=` to combine.
- `GET /api/annotate?function=dispatch::worker::Worker::run` —
  interleaved source + assembly with cycles % and per-cache-level %.

## How tracks are categorised

| Track   | Source                                                |
|---------|-------------------------------------------------------|
| Cycles  | `cycles` (or `cpu-cycles`, `cpu-clock`) PMC samples   |
| L1      | `ibs_op` samples with data_src L1 hit *or* LFB/MAB    |
| L2      | `ibs_op` samples with data_src L2 hit                 |
| L3      | `ibs_op` samples with data_src L3 hit                 |
| DRAM    | `ibs_op` samples with data_src local *or* remote DRAM |

LFB folds into L1 (LFB hits are for in-flight L1 misses on the same
line). Remote DRAM folds into DRAM to keep the timeline focused on the
latency tier.

## UI flow

1. Pick categories (Cycles / DRAM / L1 / L2 / L3).
2. Toggle Consolidated to switch between one row per category vs. one
   row per (category, CPU).
3. Click a track — flamegraph for that track loads underneath.
4. Click a flamegraph frame to zoom into it; double-click (or
   shift+click) to open source + assembly for that function.

The "Memory throughput" panel at the bottom is a placeholder; the next
iteration will populate it from PMC bandwidth counters.

## Testing the parser

```bash
cargo test -p ibs-annotate
```

Covers: `extract_field`, the cache-level / TLB / op / snoop mappings,
the jump simplifier, `ibs_int` substring discrimination
(`DcMiss` vs. `DcMissLat`), and a full chunked-merge of `parse_ibs_raw`
on a synthetic two-sample stream.
