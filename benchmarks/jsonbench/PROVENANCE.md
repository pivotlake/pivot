# JSONBench — vendored upstream, and where we depart from it

Upstream: <https://github.com/ClickHouse/JSONBench>. Five queries over Bluesky
Jetstream events, shipped as gzipped ndjson (1m / 10m / 100m / 1000m events).
Every query is a JSON path extraction, which is why this suite is worth running:
it measures the variant read path against whatever the shredding write path
chose to type, end to end.

## Vendored verbatim

- `duckdb-official/ddl.sql` — upstream `duckdb/ddl.sql`: `create table bluesky
  (j JSON)`.
- `duckdb-official/queries.sql` — upstream `duckdb/queries.sql`, one query per
  line, line *N* is `qNN` (q01 → line 1).

`prep-jsonbench-data.sh` builds DuckDB's table with that ddl and upstream's
`read_ndjson_objects` load, so DuckDB is loaded the way upstream loads it.

## What each engine runs

- `qNN.sql` — pivot's dialect. The column is `VARIANT`, so paths are
  `CAST(j->'commit'->'collection' AS VARCHAR)` rather than
  `j->>'$.commit.collection'`.
- `qNN-duckdb.sql` — the same question in DuckDB's JSON dialect, run against
  DuckDB's own table by `run-duckdb.sh`.

Both engines answer the same question, so `qNN.tsv` is a single oracle for both
and the timings compare. They differ from `duckdb-official/queries.sql` in three
places, all forced:

1. **A total `ORDER BY`.** Upstream orders q01/q02 by `count DESC` alone, which
   leaves ties in an arbitrary row order. The harness compares output as an exact
   string, so every query here breaks ties on the group key.

2. **q03 buckets the hour without a timezone.** Upstream uses
   `hour(TO_TIMESTAMP(time_us / 1000000))`, and `TO_TIMESTAMP` returns TIMESTAMP
   WITH TIME ZONE, so its hour depends on the session timezone. Both engines here
   build a naive timestamp instead, so the bucket is UTC on both regardless of
   where the box is. Verified against the raw json: the 1m file holds 444523
   likes, all in hour 16.

3. **q05 spans milliseconds by subtraction.** Upstream uses
   `date_diff('milliseconds', TO_TIMESTAMP(min), TO_TIMESTAMP(max))`; both engines
   here compute `(max - min) / 1000` over the raw microseconds, which is the same
   number without going through two timezone-aware timestamps to get it.

q04 keeps upstream's `first_post_date` as a real timestamp; note that truncating
to whole seconds (as upstream's `/ 1000000` does) leaves the earliest posters
tied, which is what the added `user_id` tiebreak resolves.

## Scale

`prep-jsonbench-data.sh --scale 100m` is the biggest that fits a single box
comfortably (~12.5 GB compressed). 1000m is ~125 GB compressed and ~425 GB raw.
