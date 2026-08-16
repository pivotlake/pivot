# ssb

The Star Schema Benchmark: the `lineorder` fact table joined against the
`customer`, `supplier`, `part` and `date` dimensions by the 13 canonical
queries (Q1.1-Q4.3, here `q01`-`q13` in flight order). Query texts are the
official ones from the SSB paper, unmodified; note the date dimension really
is named `date` and parses fine unquoted.

## Data

`prep-ssb-data.sh` builds the dataset: it clones and compiles ssb-dbgen (the
maintained eyalroz fork), generates the .tbl files at the requested scale
factor and converts them to parquet with DuckDB, one directory per table as
setup.sql expects:

```sh
./prep-ssb-data.sh --scale 1 --output ~/ssb-sf1
./prep-ssb-data.sh --scale 100 --output ~/ssb-sf100
```

At SF100 the intermediate .tbl files are ~60GB on top of the ~25GB parquet;
they are deleted after conversion unless `--keep-tbl` is given.

## Expected output

The committed `qNN.tsv` files are for **SF1** and were written by DuckDB as an
independent oracle (`./run-duckdb.sh --source ~/ssb-sf1 --write-expected`),
then confirmed byte-identical to pivot's pgwire output. Runs against any other
scale factor need `--skip-check` (or `--update-results` on a scratch copy).

## Running

```sh
# pivot, from the crate root
cargo run --release -- --suite ssb --source ~/ssb-sf1 --server-bin <pivotdb-server>

# DuckDB on the same parquet, for a side-by-side
./run-duckdb.sh --source ~/ssb-sf1 --iterations 3
```
