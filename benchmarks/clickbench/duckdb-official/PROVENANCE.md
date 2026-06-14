# Official ClickBench DuckDB (native) harness — vendored

These two files are copied **verbatim** from the upstream ClickBench DuckDB
native benchmark so our `native` mode measures DuckDB the way the public
ClickBench leaderboard does, rather than an ad-hoc load of our own.

- `create.sql`  — upstream `duckdb/create.sql`: the fully-typed `hits` schema
  (EventTime/ClientEventTime/LocalEventTime as **TIMESTAMP**, EventDate as
  **DATE**, strings as TEXT/CHAR, all `NOT NULL`).
- `queries.sql` — upstream `duckdb/queries.sql`: the 43 ClickBench queries,
  one per line. Line *N+1* is query `qNN` (q0 → line 1, q42 → line 43). These
  use `EventTime` directly as a TIMESTAMP (e.g. `DATE_TRUNC('minute', EventTime)`
  for q42) — no `toDateTime` macro, because the native schema already stores it
  as a TIMESTAMP.

Source: https://github.com/ClickHouse/ClickBench — `duckdb/` directory.

## How the harness uses them
`prep-modes-data.sh` loads the native `hits.db` from this `create.sql` (then
INSERTs the parquet, converting the three packed-seconds columns with
`epoch_ms(col*1000)` and EventDate with `make_date`, exactly as upstream's
`duckdb/load`). `run-duckdb.sh --native` runs the matching line from this
`queries.sql` for each query id. The pivot side still runs `clickbench/qNN.sql`;
they're paired by id (both are ClickBench query NN).

The **parquet** modes (single/partitioned) are not vendored because our
`run-duckdb.sh` parquet view + `qNN.sql`/`q42-duckdb.sql` are already
byte-identical to upstream `duckdb-parquet/{create,queries}.sql`.

## Refreshing
Re-copy both files from upstream if ClickBench changes its DuckDB schema or
queries. Keep them verbatim — divergence here is the whole thing we're
guarding against.
