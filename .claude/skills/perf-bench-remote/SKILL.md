---
name: perf-bench-remote
description: SSH to the pivotdb benchmark box and run ClickBench performance comparisons (pivot vs DuckDB), including PGO builds. Use when asked to benchmark a query, compare pivot to DuckDB, profile, or measure hot/cold timings on the remote machine.
---

# Remote performance benchmarking (pivotdb vs DuckDB)

## The box
- Find a stopped perf machine (it should be named perf-x) in aws (using aws cli) and start it (eu-central-1). NEVER use a running machine unless explicitly told to do so.
- For x86 benchmarking use `perf-x86-1` (c7a.4xlarge, AMD Zen 4, 16 cores, 32GB RAM), same eu-central-1 region and home layout as the Graviton fleet. It is the only x86 box; the perf-1..5 boxes are ARM and their binaries do not run on it.
- `duckdb` is at `~/.duckdb/cli/1.5.3/duckdb` — **NOT on the default non-interactive PATH**. `cargo` is at `~/.cargo/bin`. Always export both:
  `export PATH=$PATH:$HOME/.duckdb/cli/1.5.3:$HOME/.cargo/bin`
- Repo: `~/pivotdb` (no `.git` on the box — edit locally and `scp`, or edit in place). Data: `~/hits/*.parquet` (~100M-row ClickBench `hits`). Box has 125GB RAM, 16 cores.
- Real ClickBench query texts: `~/ClickBench/*/queries.sql` (use these verbatim; don't invent queries).


## ⚠️ SSH drops on long commands
The link **times out and kills foreground commands** that run more than ~1-2 min (builds, instrumented PGO runs, full benchmarks). For anything long, run it detached and poll for a sentinel — but the sentinel must fire on **every** exit path, or an early failure leaves no marker and the poll hangs forever.

1. Write a script that records its PID and emits the sentinel via an `EXIT` trap (so it fires on success, on `set -e` failure, and on normal exit), including the real exit code:

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   echo $$ >/tmp/run.pid
   trap 'echo "=== BENCH COMPLETE exit=$? ==="' EXIT

   # ... your build / PGO run / benchmark here ...
   ```

2. `scp` it to `/tmp/run.sh`, then launch detached: `setsid /tmp/run.sh >/tmp/run.log 2>&1 </dev/null &`

3. Poll from a fresh ssh (use `-o ServerAliveInterval=20`). Match the sentinel, **and** watch the PID so an un-trappable SIGKILL (OOM on the EPYC box) also ends the poll instead of hanging:

   ```bash
   pid=$(cat /tmp/run.pid)
   while ! grep -q 'BENCH COMPLETE' /tmp/run.log; do
     if ! kill -0 "$pid" 2>/dev/null; then
       echo "process gone without sentinel — likely OOM/SIGKILL"
       tail -n 30 /tmp/run.log; exit 1
     fi
     sleep 5
   done
   tail -n 20 /tmp/run.log   # exit=0 → success; anything else → it failed, reason is above
   ```

**Why the trap, not a trailing `echo`:** a plain `echo "=== BENCH COMPLETE ==="` at the end only runs on the happy path, so any early failure (build error, missing file, non-zero exit) leaves no marker and the poll spins forever. The `EXIT` trap fires regardless, and `$?` inside it is the true exit status.

**Why `echo $$` inside the script, not `echo $!` after `setsid`:** `setsid` forks and the parent exits, so `$!` is the wrong PID. Capturing `$$` from inside is the reliable handle for `kill -0`. SIGKILL can't be trapped, so the PID liveness check is what covers OOM/`kill -9`.

## Running benchmarks
The crate lives in `~/pivotdb/benchmarks`; each benchmark is a suite subdirectory. ClickBench query/oracle pairs are `clickbench/qNN.sql` + `qNN.tsv` (drop in two files to add a query; IDs match ClickBench numbering, e.g. q02, q32). The ClickBench driver scripts live in `clickbench/`; `cargo`/`just` still run from the crate root `~/pivotdb/benchmarks`.

Build the binary first (see PGO below), then point the drivers at it. **None of
these build** — `--binary` is required, so a measurement can never turn into a
rebuild halfway through.

```bash
T=aarch64-unknown-linux-gnu
B=~/bench/working/target-pgouse/$T/profiling/pivot-bench
```

- **pivot only:** `$B --source ~/hits --query q32 --iterations 6`
  Prints `[i/N] Query qNN — Xms`. `--update-results` rewrites the `.tsv` (pivot
  grading itself) — only when adding/changing a query or when its result isn't a
  stable oracle. `--skip-check` skips comparison but still times.
- **DuckDB only:** `clickbench/run-duckdb.sh --source ~/hits --query 32 --iterations 4 --no-drop-caches` → `Run Time (s): real 0.xxx`.
- **Side-by-side (pivot vs DuckDB/ClickHouse, cold + hot + speedup):**
  `clickbench/benchmark.sh --binary $B --source ~/hits --query 32 --iterations 6`
  Add `--duckdb`, or `--clickhouse <binary>`; a bare run is pivot-only. The
  canonical invocation is `--restart-server --iterations 3 --skip-check`:
  `--restart-server` restarts the server and drops caches **per query**, so every
  query's iteration 1 is a true cold read. Without it the cache is dropped once
  and only the first query is cold.
  ⚠️ `--duckdb-process` defaults to `per-iteration`, which is how ClickBench
  measures DuckDB but is **unfair to it here** (pivot runs against one warm
  server while DuckDB is forced engine-cold every iteration). Use `single` for a
  symmetric comparison.

### Other suites
`benchmark.sh` is ClickBench-only. For everything else, run pivot with `--suite`
and the suite's own DuckDB driver, then compare by hand:

```bash
$B --suite tpch-flat --source ~/tpch-flat --query q01 --iterations 6
tpch/run-duckdb.sh --source ~/tpch-flat --query 1 --iterations 6      # also tpch/, jsonbench/
```

`--suite` is accepted by `pivot-bench`, `setup-bench` and `bench-build` alike,
and defaults to `clickbench`. Profile the suite you measure: `setup-bench
--suite X` profiles suite X, and a ClickBench profile doesn't describe TPC-H's
hot paths.

## Running benchmarks with PGO
When you begin working, to initialize, run:
`just setup-bench --suite tpch-flat --pgo-source ~/tpch-flat-subset` (replace with your suite and subset)

This will create a directory like so:
```
~/bench/
  pgo/active.profdata              # the profile the compiler reads, for both sides
  baseline/  pgo/  target-pgogen/  target-pgouse/
  working/   pgo/  target-pgogen/  target-pgouse/
```

Every time you want to make a change, you can run
`just bench-build --regen-profile --suite tpch-flat --pgo-source ~/tpch-flat-subset`
which will only affect the `working/` directory. This will ONLY rebuild planner and up (and not all our dependencies). 
This is on PURPOSE, as we don't need our dependencies to rebuild from a new perf profile every single run

Both default to `--profile profiling` (inherits release, adds debug info, so perf
resolves source lines and inlined frames; codegen is unchanged so these are
release timings). Pass the same `--profile` to both, or they land in different
artifact directories.

Suites:
| Suite | Queries | `--source` layout | Data |
|---|---|---|---|
| `clickbench` | 43 | flat dir of parquet | `~/hits`, PGO subset `~/hits-pgo-subset` |
| `tpch` | 6 | **one subdirectory per table** (`{source}/lineitem`, `/orders`, …) | `tpch/prep-tpch-data.sh` |
| `tpch-flat` | 8 | flat dir of parquet (one denormalized table) | `tpch-flat/prep-tpch-flat.sh` |
| `jsonbench` | 5 | dir of newline-delimited JSON | `jsonbench/prep-jsonbench-data.sh` |

Only the ClickBench dataset is on the box by default; the others need their prep
script run first. (`~/bench-data*` is ClickBench mode data for `bench-modes.sh`,
not TPC-H.)

Then measure each binary against the **full** dataset, never `~/hits-pgo-subset` (or equivalent subsets):

For example, to check q32:
```bash
T=aarch64-unknown-linux-gnu
~/bench/baseline/target-pgouse/$T/profiling/pivot-bench --source ~/hits --query q32 --iterations 6 --skip-check
~/bench/working/target-pgouse/$T/profiling/pivot-bench  --source ~/hits --query q32 --iterations 6 --skip-check
```

⚠️ **Run them one at a time, and never alongside a build.** `pivot-bench` sizes
its ring at 4/5 of RAM (~24GB of the 30GB a c8g perf box has), so two at once,
or one next to a compile, triggers the OOM killer and takes both down.

## Reading the numbers
- **Cold** = iteration 1 (fresh process / page cache). **Hot** = mean of iterations 2..N (steady state) — this is the headline metric.
- `benchmark.sh` runs **all pivot queries, then all DuckDB queries, in one session** → pivot's large hash tables can evict DuckDB's working set and inflate DuckDB's time. For a *fair* head-to-head, also run each engine standalone back-to-back.
- High pivot variance across iterations on big-memory queries usually = buffer-pool zeroing as the pre-zeroed pool depletes; the first 1-3 iterations are fastest.

## Profiling (perf is available, `perf_event_paranoid=-1`)
Note that for perf to work, you must make sure mem lock limit is high! the default is too low and it hangs.

```bash
perf record -g -F 499 -o /tmp/p.data -- ./target/release/pivot-bench --source ~/hits --query q32 --iterations 4 --update-results >/dev/null 2>&1
perf report -i /tmp/p.data --stdio --no-children | grep -vE '^#' | head -20
perf diff /tmp/a.data /tmp/b.data        # compare two builds/queries
```

Ignore `MemoryContext::prefault_buffers` for steady-state — it's largely one-time buffer-pool warmup.

Note that for perf annotate;
perf annotate's symbol filter requires the exact, fully-qualified name as perf report prints it — there is no substring or fuzzy matching. So:
- consume_window / merge_combined (bare leaf names) match nothing → that's why it "fails for every symbol." It's not the build, not the event, not perf.
- For Rust you must pass the whole path including generic params, e.g. dispatch::operations::unary::group::hashtables::aggregated_table::AggregatedTable<K,V>::consume_window — quoted (the <, >, ::, spaces matter).

## Inspecting a query's plan
DuckDB logical plan that pivot consumes (pivot disables 3 optimizers — match them):

```bash
duckdb -c "CREATE VIEW hits AS SELECT * FROM read_parquet('$HOME/hits/*.parquet');
SET disabled_optimizers='compressed_materialization,late_materialization,empty_result_pullup';
SET explain_output='optimized_only';
EXPLAIN <query>;"
```

For pivot's translated plan, gate a dump behind an env var in `planner::Planner::plan` (`PIVOT_DUMP_PLAN`) — and revert it after.
