# tpch_flat

TPC-H as a single-table ("flat") benchmark: the eight TPC-H tables are
denormalised into one lineitem-grain parquet, so every query is a pure scan +
aggregation with **no joins** - the shape pivotdb runs. All three engines
(pivot, DuckDB, ClickHouse) read the same `lineitem_flat.parquet`, so the
numbers reflect the scan/aggregate path, not a join planner.

## Data

```sh
./prep-tpch-flat-data.sh --sf 1 --root ~/tpch-flat
```

DuckDB's `tpch` extension generates the normalised tables; one query joins them
into `lineitem_flat.parquet` (one row per lineitem carrying its order /
customer / supplier / part / partsupp attributes, plus both the supplier's and
the customer's nation+region). Monetary / quantity columns are written as
`DOUBLE` (pivot evaluates decimal arithmetic in floating point, so this keeps
all three engines computing the same way). SF1 is ~6M rows, ~512 MB.

## Running

```sh
# pivot only, through the generic suite runner
cargo run --release -- --suite tpch_flat --source ~/tpch-flat --query 4

# pivot vs DuckDB vs ClickHouse, side by side
./bench-tpch-flat.sh --source ~/tpch-flat --query 4 --clickhouse clickhouse
```

## Queries

`q01, q03, q04, q05, q06, q07, q08, q09, q10, q12, q14, q19` - the TPC-H
questions answerable by a single scan (the rest need joins/subqueries:
self-joins, correlated subqueries, or two-level aggregation). TPC-H's
`INTERVAL` date offsets are precomputed to literal dates, since the planner has
no interval arithmetic (e.g. Q1's `date '1998-12-01' - interval '90' day`
becomes `date '1998-09-02'`).

## What runs today

Only **q04** runs end to end. The other queries are committed as scaffolding;
they need executor/planner features pivotdb does not have yet:

- **Aggregation over an expression** - `SUM(l_extendedprice * (1 - l_discount))`
  is rejected; an aggregate's argument must be a bare column. (Affects almost
  every query.)
- **Floating-point aggregation** - `SUM`/`AVG`/`MIN`/`MAX` read integer columns
  only (the accumulator is `i64`/`i128`); there is no float path, so money
  aggregation can't run.
- **`prefix`** - `LIKE 'PROMO%'` lowers to a `prefix()` scalar the planner
  doesn't implement (q14). `LIKE '%x%'` already lowers to the supported
  `contains()`.

q04 (`COUNT(DISTINCT l_orderkey)` over a date + commit/receipt-date filter,
grouped by order priority) needs none of these; its output matches the
canonical TPC-H SF1 reference values.
