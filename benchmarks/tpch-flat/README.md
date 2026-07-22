# tpch-flat

TPC-H as a single denormalized table, so pivot runs it as pure single-table
scan+aggregate (before pivot had a join executor). The 8 normalized TPC-H tables are
joined lineitem-centric into one wide table — one row per lineitem, with its
order / customer / part / supplier / partsupp and both nation+region names
folded in. See `setup.sql` for the schema.

## Data

```sh
# generate SF100 base tables + build the flat parquet (needs tpchgen-cli + duckdb)
./prep-tpch-flat.sh                 # → ~/tpch-data/flat
./prep-tpch-flat.sh --scale-factor 10 --root ~/tpch-sf10   # smaller

# run the suite against it (from the crate root)
cd .. && cargo run --release -- --suite tpch-flat --source ~/tpch-data/flat --iterations 3
```

At SF100 the flat table is ~600M rows (dims join 1:1, no fan-out). Money and
quantity columns are `DOUBLE` (TPC-H's `DECIMAL` cast down) so pivot's floating
aggregates apply; dates are real `DATE`.

## Queries

The suite ships the lineitem-grain queries that are expressible as single-table
scan+aggregate, numbered to match standard TPC-H. All eight run on pivot and
their results match DuckDB on the same flat parquet.

| Query | TPC-H | Notes |
|-------|-------|-------|
| q01 | Pricing Summary | group by returnflag/linestatus |
| q03 | Shipping Priority | group by order, top revenue |
| q05 | Local Supplier Volume | customer nation == supplier nation, region ASIA |
| q06 | Forecasting Revenue | single-row scan+filter |
| q10 | Returned Item Reporting | grouped by o_custkey alone (see constraints) |
| q12 | Shipping Modes | CASE priority buckets per shipmode |
| q14 | Promotion Effect | promo share of revenue |
| q19 | Discounted Revenue | multi-branch predicate |

### Engine constraint these queries work around

The queries are the standard TPC-H text (interval arithmetic, `LIKE 'x%'`, and
up to eight aggregates per group are all supported). One structural limit
remains:

- **No float column as a GROUP BY key.** Standard q10 groups by `c_acctbal`
  (a `DOUBLE`); instead it groups by `o_custkey` alone (which determines every
  other customer column) and pulls the rest through `MIN()`. The output is
  identical.

**Not included, by design.** A lineitem-grain flat table can't express queries
that live at a different grain or need anti-joins:

- part/supplier/partsupp-only (no lineitem grain): Q2, Q11, Q16, Q20.
- customers/orders that don't appear (LEFT JOIN / NOT EXISTS): Q13, Q22.
- order-level existence across sibling lines: Q21.
- year-grouping via `EXTRACT`/`year()` (not yet exercised on pivot): Q7, Q8, Q9.
- correlated-subquery thresholds (become joins → unsupported): Q17, Q18.

## Oracles

`qNN.tsv` holds the expected pgwire output. It is **scale-factor specific**
(TPC-H results scale with the data), so oracles aren't committed — generate them
for your SF on first run with `--update-results` (pivot grading its own output),
after confirming the numbers against DuckDB on the same flat parquet:

```sh
cd .. && cargo run --release -- --suite tpch-flat --source ~/tpch-data/flat --update-results
```

Every query has a total `ORDER BY` (with tie-breakers) so the `.tsv` is stable.
