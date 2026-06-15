# Official ClickBench ClickHouse-over-parquet harness — vendored

These two files are copied **verbatim** from the upstream ClickBench
`clickhouse-parquet` benchmark so our ClickHouse comparison measures it the way
the public ClickBench leaderboard does.

- `create.sql`  — upstream `clickhouse-parquet/create.sql`: the fully-typed `hits`
  schema ending in `ENGINE = File(Parquet, 'hits.parquet')`, i.e. a ClickHouse
  table backed directly by the parquet file(s), read in place (no ingest).
- `queries.sql` — upstream `clickhouse-parquet/queries.sql`: the 43 ClickBench
  queries in ClickHouse SQL dialect, one per line. Line *N+1* is `qNN`.

Source: https://github.com/ClickHouse/ClickBench — `clickhouse-parquet/` directory.

## How the harness runs it (matching upstream exactly)
Upstream's `query` script is:

    ./clickhouse local --time --format=Pretty --query="$(cat create.sql); ${query}"

i.e. a **fresh `clickhouse local` process per query** (and per try — the driver
loops `BENCH_TRIES` times), with the table definition prepended each time and
`--time` printing the elapsed seconds to stderr. `clickhouse local` reads the
parquet in place via the `File(Parquet, …)` engine; there is no load step.

`run-clickhouse.sh` does the same: per iteration it spawns one `clickhouse local`
from the data directory, substituting the `File(Parquet, …)` glob for the source
(single dir → one `hits.parquet`; partitioned dir → `hits_*.parquet`), and reads
the `--time` value. So ClickHouse, like DuckDB-per-iteration, is measured
engine-cold each run — exactly the leaderboard's `clickhouse-parquet` /
`clickhouse-parquet-partitioned` methodology.

## Refreshing
Re-copy both files from upstream if ClickBench changes the ClickHouse schema or
queries. Keep them verbatim.
