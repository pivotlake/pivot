# ssb

The Star Schema Benchmark: the `lineorder` fact table joined against four
dimensions (`customer`, `supplier`, `part`, `date`) by the 13 official
queries. Where `tpch` exercises subqueries and exotic join shapes, this suite
is pure star joins: every query is dimension filters + fact scan + group by,
in four "flights" of increasing dimensionality.

## Data

The datasets are pre-generated (ssb-dbgen → DuckDB → parquet, see the
provenance notes in prep-ssb-data.sh) and hosted at `s3://pivot-benchmarks/ssb/`:

| Dataset | Path | Size |
|---|---|---|
| SF1 (smoke tests) | `s3://pivot-benchmarks/ssb/sf1/` | ~0.4 GB |
| SF100, dbgen row order | `s3://pivot-benchmarks/ssb/sf100/` | ~30 GB |
| SF100, ClickHouse sort keys | `s3://pivot-benchmarks/ssb/sf100-sorted/` | ~18 GB |

```sh
./prep-ssb-data.sh                     # sorted → ~/ssb-sf100-sorted
cd .. && cargo run --release -- --suite ssb --source ~/ssb-sf100-sorted --iterations 3
```

Layout: one directory per table (`<root>/lineorder/*.parquet`, ...). Types
follow the SSB spec: integer money values (no decimals), yyyymmdd INTEGER
date keys joined against the `date` dimension, BIGINT surrogate keys,
VARCHAR text. At SF100 lineorder is ~600M rows, the same row count as
tpch's SF100 lineitem.

Both datasets hold the same rows; they differ only in physical order. In
the sorted dataset each table is sorted by its ClickHouse ORDER BY key
(lineorder globally by `(lo_orderdate, lo_orderkey)` in ~1 GB files), so
row-group min/max stats carve the sort key into narrow ranges and
date-bounded scans prune almost everything. The dbgen-order dataset is the
faithful generator output, useful as the layout-neutral baseline.

## Queries

The official SSB query texts (O'Neil et al., default substitution
parameters). File stems drop the flight dot: `q11.sql` is Q1.1, `q43.sql` is
Q4.3 - so `--query 21` runs Q2.1, not a 21st query.

| Query | SSB | Shape |
|-------|-----|-------|
| q11-q13 | Q1.1-Q1.3 | lineorder × date, no group by: one summed revenue row under year/month/week date filters plus discount and quantity bands |
| q21-q23 | Q2.1-Q2.3 | lineorder × date × part × supplier, revenue per (year, brand), part filter narrowing from a category (q21) to a brand range (q22) to one brand (q23) |
| q31-q34 | Q3.1-Q3.4 | lineorder × date × customer × supplier, revenue per (customer geo, supplier geo, year), narrowing region → nation → city pair, then to one month in q34 |
| q41-q43 | Q4.1-Q4.3 | all 5 tables, profit (revenue - supplycost) per year and customer nation / supplier nation and category / supplier city and brand |

One deviation from the official texts: Q3.1-Q3.4 order by `revenue desc`
inside each year, and a tie in a summed revenue would make the row order (and
therefore the byte-exact oracle comparison) arbitrary, so the group keys are
appended as trailing sort keys. Rows only reorder against the official output
where revenues tie exactly.

Oracles (`qNN.tsv`) are produced by DuckDB over the same parquet files
(`run-duckdb.sh --write-expected`), in pivot's wire format, and are specific
to the dataset's scale factor. The committed ones are for SF100; regenerate
them when you run another scale factor:

```sh
./run-duckdb.sh --source ~/ssb-sf10 --write-expected
```
