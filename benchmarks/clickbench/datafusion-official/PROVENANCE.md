# Official ClickBench DataFusion harness — vendored

These two files are copied **verbatim** from the upstream ClickBench
`datafusion` benchmark (commit 8e41713902b47ea975555f27f8e96685751601c9), so
our DataFusion comparison measures it the way the public ClickBench
leaderboard does.

- `create.sql`  — upstream `datafusion/create.sql`: an external parquet table
  `hits_raw` (`binary_as_string` so the string columns read as UTF-8 rather than
  binary) and a `hits` view over it that casts `EventDate` from its stored
  day-number integer to a real `DATE`. There is no load step: DataFusion reads
  the parquet in place.
- `queries.sql` — upstream `datafusion/queries.sql`: the 43 ClickBench queries
  in DataFusion's SQL dialect, one per line. Line *N+1* is `qNN`. Identifiers
  are double-quoted because DataFusion folds unquoted ones to lower case.

The upstream `datafusion-partitioned` benchmark uses the same files except that
its `create.sql` says `LOCATION 'partitioned'` (a directory of 100 files) in
place of `LOCATION 'hits.parquet'`; `run-datafusion.sh` rewrites that one
`LOCATION` line to whatever `--source` names, so one vendored copy covers both.

Source: https://github.com/ClickHouse/ClickBench — `datafusion/` and
`datafusion-partitioned/` directories.

## How the harness runs it (matching upstream exactly)
Upstream's `query` script is:

    datafusion-cli -f create.sql "$query_file"

i.e. a **fresh `datafusion-cli` process per query** (and per try — the driver
loops `BENCH_TRIES` times), with the table definition prepended each time. The
CLI prints `Elapsed X.YYY seconds.` after every statement and upstream takes the
last one as the query's time.

`run-datafusion.sh` does the same by default (`--datafusion-process
per-iteration`): one `datafusion-cli` per iteration, the rewritten `create.sql`
first, then the query, and the trailing `Elapsed` value reported. So DataFusion,
like DuckDB-per-iteration and ClickHouse-local, is measured engine-cold each
run — exactly the leaderboard's `datafusion` / `datafusion-partitioned`
methodology.

## Refreshing
Re-copy both files from upstream if ClickBench changes the DataFusion schema or
queries. Keep them verbatim.
