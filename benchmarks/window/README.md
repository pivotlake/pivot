# Window function suite

Eight queries over the TPC-H `lineitem` table, each a window function inside a
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
| q07 | `row_number() OVER (PARTITION BY l_suppkey ORDER BY l_comment, l_shipdate)` | A long string, 10 to 43 bytes, in the middle of the sort key. |
| q08 | `row_number() OVER (PARTITION BY l_suppkey ORDER BY l_shipinstruct, l_shipdate)` | A string with four values, two of them longer than 12 bytes, so most rows tie on it. |

q04 to q08 partition by supplier on purpose. A supplier has several hundred
lines, so most comparisons in the sort are between rows of one partition, and
those are decided by the order key.

## Microbenchmark

`dispatch/benches/window.rs` has one scenario per query. A scenario runs the
query's window, with the same input columns, types and keys as the query's
plan, over generated rows held in memory. It leaves out the scan, the
expressions around the window and the aggregate on top.

| Query | Scenario |
|---|---|
| q01 | `sum_per_order` |
| q02 | `top_3_prices_per_part` |
| q03 | `running_sum_and_lag_per_supplier` |
| q04 | `rank_by_revenue_per_supplier` |
| q05 | `row_number_by_ship_mode_then_date` |
| q06 | `row_number_by_nullable_line_then_date` |
| q07 | `row_number_by_comment_then_date` |
| q08 | `row_number_by_instructions_then_date` |

```sh
cargo bench -p dispatch --bench window
```

At its default of 60M rows a scenario takes within 8% of its query's hot time
at scale factor 10, and a profile of it splits between sorting, comparing,
gathering and evaluating as the query's does. q01 is the exception: about a
quarter of the query's time is the scan, the expressions and the aggregate
around the window, which the scenario does not run.

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
