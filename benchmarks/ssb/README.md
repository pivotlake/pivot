# ssb

The Star Schema Benchmark: the `lineorder` fact table joined against four
dimensions (`customer`, `supplier`, `part`, `date`) by the 13 official
queries. Where `tpch` exercises subqueries and exotic join shapes, this suite
is pure star joins: every query is dimension filters + fact scan + group by,
in four "flights" of increasing dimensionality.

## Data

Generated locally with ssb-dbgen and converted to parquet:

```sh
./prep-ssb-data.sh --scale-factor 100 --root ~/ssb-sf100   # ~30 GB parquet
cd .. && cargo run --release -- --suite ssb --source ~/ssb-sf100 --iterations 3
```

Layout: one directory per table (`<root>/lineorder/*.parquet`, ...). Types
follow the SSB spec: integer money values (no decimals), yyyymmdd INTEGER
date keys joined against the `date` dimension, BIGINT surrogate keys,
VARCHAR text. At SF100 lineorder is ~600M rows, the same row count as
tpch's SF100 lineitem.

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
