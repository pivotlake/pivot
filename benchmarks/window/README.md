# Window function suite

Six queries over the TPC-H `lineitem` table, each a window function inside a
subquery with an aggregate on top, so the result is one row and the time is
the window operator's.

| Query | Window | What it exercises |
|---|---|---|
| q01 | `sum(...) OVER (PARTITION BY l_orderkey)` | One 64-bit key, tiny partitions, no order key. |
| q02 | `row_number() OVER (PARTITION BY l_partkey ORDER BY l_extendedprice DESC) <= 3` | A row limit on `row_number()`; two 64-bit key columns. |
| q03 | running `sum` and `lag` `OVER (PARTITION BY l_suppkey ORDER BY l_shipdate, l_orderkey, l_linenumber)` | Four fixed-width key columns, 192 bits together. |
| q04 | `rank() OVER (PARTITION BY l_suppkey ORDER BY l_extendedprice * (1 - l_discount) DESC)` | An expression as the order key. |
| q05 | `row_number() OVER (PARTITION BY l_suppkey ORDER BY l_shipmode, l_shipdate)` | A string in the middle of the sort key. |
| q06 | `row_number() OVER (PARTITION BY l_suppkey ORDER BY NULLIF(l_linenumber, 1), l_shipdate)` | A key column with nulls. |

q04 to q06 partition by supplier on purpose. A supplier has several hundred
lines, so most comparisons in the sort are between rows of one partition, and
those are decided by the order key.

## Data

`setup.sql` registers the parquet files under `<source>/lineitem/` as the
`lineitem` table. The measurements so far used TPC-H scale factors 10, 30 and
100, which are 60M, 180M and 600M rows.

## Running

```sh
# pivot, from benchmarks/
cargo run --release -- --suite window --source <source> --server-bin <pivot binary> \
    --iterations 5 --skip-check

# DuckDB, on its own storage
window/run-duckdb.sh <source> <database file>
```

The suite ships no expected results, because they depend on the scale factor.
`--update-results` writes them for the data at hand; q01 to q06 gave the same
results as DuckDB at scale factors 10 and 30.
