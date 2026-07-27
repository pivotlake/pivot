# benchmarks

Performance harness for pivotdb. Boots `dispatch` + `server` + `DeltaDatastore`
in-process, connects with `tokio-postgres`, runs a suite of SQL queries
through the pgwire path, verifies output, and compares timings against a
saved baseline.

## Quick start

```sh
# from the workspace root
cd benchmarks

# run the full clickbench suite
cargo run --release -- --source ~/hits

# run just q07 and q20, three iterations each, sleeping 500ms between
cargo run --release -- --source ~/hits --query 7,20 --iterations 3 --sleep 500

# regenerate expected output (use after a deliberate semantic change, or
# when running against a smaller/different dataset)
cargo run --release -- --source ~/hits --update-results

# compare against the baseline; overwrite it (appending a timestamped history
# row) if *either* the cold or the hot suite total beats it by more than the
# regression threshold
cargo run --release -- --source ~/hits --iterations 5 --save-if-better

# compare against a baseline stored in GCS
cargo run --release -- --source ~/hits --baseline gs://my-bucket/clickbench.json

# show the recorded results without running anything
cargo run -- --show
cargo run -- --show --baseline gs://my-bucket/clickbench.json
```

## Suite layout

A suite is a self-contained directory under `benchmarks/<name>/`: its schema,
its queries + oracles, and any suite-specific driver/data-prep scripts all live
together, so `clickbench` and `tpch` don't sit flat side by side. The shared
Rust harness (`src/`, `Cargo.toml`, `justfile`) stays at the crate root and
serves every suite via `--suite <name>`.

```
benchmarks/
├── src/ Cargo.toml justfile      # shared harness (suite-agnostic)
├── clickbench/
│   ├── setup.sql                 # CREATE TABLE etc. {source} → --source
│   ├── q07.sql  q07.tsv          # one query per file; stem is the query ID;
│   ├── q20.sql  q20.tsv          #   qNN.tsv is the expected pgwire output
│   ├── baseline.json             # default location for saved timings
│   ├── benchmark.sh bench-modes.sh bench-ab.sh   # ClickBench drivers
│   ├── prep-modes-data.sh prep-clickhouse-native.sh
│   ├── run-duckdb.sh run-clickhouse.sh
│   └── duckdb-official/ clickhouse-official/      # vendored native schemas
├── tpch/
│   ├── setup.sql                 # the 8 normalized TPC-H tables
│   ├── qNN.sql  qNN.tsv          # official TPC-H query texts
│   └── prep-tpch-data.sh         # sync the parquet dataset from S3
└── tpch-flat/
    ├── setup.sql                 # the denormalized flat table
    ├── qNN.sql  qNN.tsv
    └── prep-tpch-flat.sh         # tpchgen-cli → DuckDB denormalize → flat parquet
```

`cargo` / `just` always run from the crate root (`benchmarks/`); the suite's
shell scripts are invoked by path (e.g. `clickbench/benchmark.sh`). Those
scripts internally `cd` to the crate root for `just`, so they work from any cwd.

Adding a query: drop in `qNN.sql` + `qNN.tsv` under the suite dir. The harness
picks them up via directory listing - no code change. Give every query a
*total* `ORDER BY`: output is compared exact-string, so any tie in row order
makes the `.tsv` flaky. For the expected `.tsv`, prefer an independent oracle
over pivot grading its own output - `clickbench/run-duckdb.sh --source ~/hits
--query NN --write-expected` runs the same query through DuckDB on the same
parquet and writes `qNN.tsv` in pivot's wire format. Then confirm pivot agrees
with a plain run. (DuckDB rewrites a few columns - chiefly `EventDate` → a real
`DATE` - so for `SELECT *`/date queries that path won't match; fall back to
`--update-results` and eyeball.)

Adding a suite: `mkdir benchmarks/<name>`, fill in `setup.sql` and the
queries, then run with `--suite <name>`.

## Output verification

Every iteration's result rows are captured as TSV and compared against
`<query>.tsv`. A faster run that broke correctness is rejected with
`ResultMismatch`. Use `--update-results` to overwrite the expected file when
the change is intended.

## Baselines

Each run records two numbers per query:

- **cold** — the first iteration: what a fresh process pays before page cache,
  planner thread-locals, and allocator arenas are warm.
- **hot** — the arithmetic mean of every iteration *after* the first: steady
  state. A run with a single iteration has no hot number.

These are persisted as JSON. The file has two parts:

- `current[suite][query]` — the latest cold/hot per query. Keyed by *suite
  first*, so two suites that happen to share a query stem (`q01`) never
  overwrite each other.
- `history` — an append-only log, one timestamped row per `(suite, query)`
  every time a run is saved. Each row carries its suite, so slicing the log
  later is trivial.

Default location is `<suite_dir>/baseline.json`; override with `--baseline`:

| Form | Read | Write |
|------|------|-------|
| `path/to/file.json` | local FS | local FS |
| `gs://bucket/key`   | HTTP GET on the public `storage.googleapis.com` endpoint | `gsutil cp` (or `gcloud storage cp`) |
| `https://...`       | HTTP GET | not supported |

Saving:
- (no flag) — read-only, just print the comparison table.
- `--save-if-better` — overwrite if **either** the cold suite total or the hot
  suite total improved by more than `--regression-pct` (default 5%). One solid
  win is enough; the other axis may be flat or slightly worse. Sub-threshold
  wobble on both axes does not save. (A single-iteration run has no hot total,
  so it's judged on cold alone.)
- `--force-save` — overwrite unconditionally.

Either save path appends to `history`; it never rewrites past rows.

`--show` prints what's currently recorded (cold/hot per suite & query, run
count, last-saved timestamp) and exits — handy for inspecting a baseline
without a dataset on hand. The cold/hot cells are coloured against the previous
recorded run for that query (green = faster, red = slower).

The comparison table shows before/after/Δ% for cold and hot side by side, with
the Δ% cells coloured by direction (green faster, red slower) and a `status`
column: `REGRESSION` (red) if either metric is slower than the baseline by more
than `--regression-pct`, `improved` (green) if cold improved past that and hot
didn't regress, `ok` otherwise. Queries new this run print as `new`; baseline
queries that weren't run print as `missing`. Colour is suppressed when stdout
isn't a terminal or `NO_COLOR` is set.

## All flags

| Flag | Env var | Default | Notes |
|------|---------|---------|-------|
| `--source <DIR>` | `SOURCE_DIRECTORY` | required (unless `--show`) | parquet directory passed to `setup.sql` as `{source}` |
| `--suite <NAME>` | — | `clickbench` | suite directory name |
| `--suite-dir <PATH>` | — | `<crate>/<suite>` | override the resolved suite path |
| `--workers <N>` | `WORKER_COUNT` | core count | dispatch worker threads |
| `--query <IDS>` | `QUERY` | all | comma-separated; `7,20` and `q07,q20` both accepted |
| `--iterations <N>` | `QUERY_TEST_COUNT` | 1 | per-query iterations |
| `--sleep <MS>` | `SLEEP` | 0 | milliseconds to sleep between iterations |
| `--baseline <REF>` | — | `<suite_dir>/baseline.json` | local path / `gs://` / `https://` |
| `--save-if-better` | — | off | save baseline if either cold or hot suite total improved by more than `--regression-pct` |
| `--force-save` | — | off | save baseline unconditionally |
| `--regression-pct <PCT>` | — | 5 | threshold for the REGRESSION tag |
| `--update-results` | — | off | overwrite expected `.tsv` files instead of comparing |
| `--skip-check` | — | off | skip the output-vs-expected comparison (still records timings) |
| `--show` | — | off | print the baseline's recorded results and exit; no server, no run |

## Notes

- Run with `--release` for meaningful numbers — debug builds are easily 10×
  slower and the comparison table is then noise.
- Use `--iterations >= 2` whenever you care about the hot number; iteration 1
  is the cold sample, the rest feed hot. (`--save-if-better` still works with
  one iteration — it just judges on cold alone.)
- The harness re-uses one tokio-postgres connection per suite run, so query
  cost reflects steady-state planner/executor behaviour, not connection setup.
