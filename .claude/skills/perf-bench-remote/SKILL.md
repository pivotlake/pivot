---
name: perf-bench-remote
description: SSH to the pivotdb benchmark box and run ClickBench performance comparisons (pivot vs DuckDB), including PGO builds. Use when asked to benchmark a query, compare pivot to DuckDB, profile, or measure hot/cold timings on the remote machine.
---

# Remote performance benchmarking (pivotdb vs DuckDB)

## The box
- Find a stopped perf machine (it should be named perf-x) in aws (using aws cli) and start it.
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
The suite lives in `~/pivotdb/benchmarks`. Query/oracle pairs are `clickbench/qNN.sql` + `qNN.tsv` (drop in two files to add a query; IDs match ClickBench numbering, e.g. q02, q32).

- **pivot only, quick timing (non-PGO):**
  `cd ~/pivotdb/benchmarks && cargo build --release && ./target/release/pivot-bench --source ~/hits --query q32 --iterations 6 --update-results`
  Prints `[i/N] Query qNN — Xms`. `--update-results` writes the `.tsv` (pivot grading itself) — needed when adding/changing a query or its result isn't a stable oracle.
- **DuckDB only:** `./run-duckdb.sh --source ~/hits --query 32 --iterations 4 --no-drop-caches` → `Run Time (s): real 0.xxx`.
- **Side-by-side table (pivot vs DuckDB, cold + hot + speedup):**
  `./benchmark.sh --source ~/hits --query 32 --iterations 6`
  `benchmark.sh` drives pivot via `just pgo-use run` (**needs a PGO profile first**, see below) and DuckDB via `run-duckdb.sh`.

## PGO build (what `benchmark.sh` expects)
Generate the merged profile once with a representative workload, then `benchmark.sh` / `just pgo-use` reuse it. The representative workload is at `~/hits-pgo-subset`. NEVER use that directory for ACTUAL perf numbers, only for creating a pgo build.

```bash
cd ~/pivotdb/benchmarks
just pgo-gen run --release -- --source ~/hits-pgo-subset --iterations 1 --update-results   # instrumented = ~80x slower, slow!
# then:
./benchmark.sh --source ~/hits --query 32 --iterations 6 --no-drop-caches
```

PGO is the user's expected default for `benchmark.sh` numbers. (Note: PGO sometimes *hurt* memory-bound queries — sanity-check vs the non-PGO build.)

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

Note tht for perf annotate;
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
