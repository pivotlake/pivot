# tpch

The real, normalized TPC-H: the 8 base tables as parquet directories and the
official query texts, joins included. (The single-table variant that predates
pivot's join executor lives in `../tpch-flat`.)

## Data

The datasets are pre-generated (tpchgen-cli) and hosted at `s3://epsio-tpch/`:

| Dataset | Path | Size |
|---|---|---|
| SF10 | `s3://epsio-tpch/sf10/` | ~3.9 GB |
| SF100, 7 MiB row groups | `s3://epsio-tpch/sf100/` | ~41.5 GB |
| SF100, 128 MiB row groups | `s3://epsio-tpch/sf100-large-row-groups/` | ~35.8 GB |

```sh
./prep-tpch-data.sh --dataset sf100        # → ~/tpch-sf100
cd .. && cargo run --release -- --suite tpch --source ~/tpch-sf100 --iterations 3
```

Layout: one directory per table (`<root>/lineitem/lineitem.N.parquet`, ...).
Types are tpchgen's parquet output: BIGINT keys, DECIMAL(15,2) money columns,
real DATEs.

## Queries

Official TPC-H query texts (default substitution parameters), added one by one
as the engine grows the features each needs. Every query carries a total ORDER
BY where the spec's ordering leaves ties (the runner compares output
exact-string).

Oracles (`qNN.tsv`) are produced by DuckDB over the same parquet files, in
pivot's wire format.
