# benchmarks

Performance harness for pivotdb. Boots `dispatch` + `server` + `ParquetCatalog`
in-process, connects with `tokio-postgres`, runs a suite of SQL queries
through the pgwire path, verifies output, and compares timings against a
saved baseline.

## Quick start

```sh
# from the workspace root
cd benchmarks

# download/use ~/hits/hits.parquet, then run the full clickbench suite
./benchmark.sh

# run just q07 and q20, three iterations each, sleeping 500ms between
cargo run --release --bin pivot-bench -- --source ~/hits --query 7,20 --iterations 3 --sleep 500

# generate/use ~/tpch-flat/tpch_flat.parquet, then compare pivot vs DuckDB
./benchmark.sh --suite tpch-flat --iterations 3

# regenerate expected output (use after a deliberate semantic change, or
# when running against a smaller/different dataset)
cargo run --release --bin pivot-bench -- --source ~/hits --update-results

# compare against the baseline; overwrite it (appending a timestamped history
# row) if *either* the cold or the hot suite total beats it by more than the
# regression threshold
cargo run --release --bin pivot-bench -- --source ~/hits --iterations 5 --save-if-better

# compare against a baseline stored in GCS
cargo run --release --bin pivot-bench -- --source ~/hits --baseline gs://my-bucket/clickbench.json

# show the recorded results without running anything
cargo run --bin pivot-bench -- --show
cargo run --bin pivot-bench -- --show --baseline gs://my-bucket/clickbench.json
```

## Suite layout

A suite is a directory under `benchmarks/<name>/`.

```
benchmarks/clickbench/
├── load.sh            # optional data loader; prints the resolved source path
├── setup.sql          # CREATE TABLE etc. {source} is substituted with --source
├── duckdb-setup.sql   # DuckDB setup template used by run-duckdb.sh
├── q07.sql            # one query per file; stem is the query ID
├── q07.tsv            # expected pgwire output (TSV, tab-separated rows)
├── q20.sql
├── q20.tsv
├── ...
└── baseline.json      # default location for saved timings (created on demand)
```

Adding a query: drop in `qNN.sql` + `qNN.tsv`. The harness picks them up via
directory listing — no code change. Give every query a *total* `ORDER BY`:
output is compared exact-string, so any tie in row order makes the `.tsv`
flaky. For the expected `.tsv`, prefer an independent oracle over pivot grading
its own output — `./run-duckdb.sh --source ~/hits --query NN --write-expected`
runs the same query through DuckDB on the same parquet and writes `qNN.tsv` in
pivot's wire format. Then confirm pivot agrees with a plain run. (DuckDB rewrites
a few columns — chiefly `EventDate` → a real `DATE` — so for `SELECT *`/date
queries that path won't match; fall back to `--update-results` and eyeball.)

Adding a suite: `mkdir benchmarks/<name>`, fill in `setup.sql` and the
queries, then run with `--suite <name>`. Add `load.sh` when the suite can
provision or transform its own data. `benchmark.sh` calls it before running and
uses the path it prints as `--source`.

`clickbench/load.sh` downloads the canonical `hits.parquet` into `~/hits` by
default when no parquet files are present. Override with `HITS_SOURCE`,
`CLICKBENCH_PARQUET_URL`, or `./benchmark.sh --source <path>`.

`tpch-flat/load.sh` uses `tpchgen-cli` to generate the normalized TPCH tables,
then materializes one wide `tpch_flat.parquet` row per `lineitem` by joining in
orders, customer, part, supplier, partsupp, nation, and region fields. The wide
file is written with Snappy on every column and large row groups by default
(`TPCH_FLAT_ROW_GROUP_ROWS=1048576`). If `tpchgen-cli` is missing, the loader
installs it with `cargo install tpchgen-cli`; set `TPCHGEN_INSTALL=0` to require
a preinstalled binary. Override the scale with `TPCH_SCALE_FACTOR=<sf>`, the
flat output with `TPCH_FLAT_SOURCE`, the generated source cache with
`TPCHGEN_SOURCE`, the row-group size with `TPCH_FLAT_ROW_GROUP_ROWS`, the
transform batch size with `TPCH_FLAT_BATCH_ROWS`, or the command line path with
`./benchmark.sh --suite tpch-flat --source <dir>`.

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
