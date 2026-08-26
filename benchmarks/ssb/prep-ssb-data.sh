#!/usr/bin/env bash
# prep-ssb-data.sh — sync the SSB parquet dataset from S3 to a local
# directory (one subdirectory per table), the layout `setup.sql` expects.
#
# The canonical datasets are hosted at s3://pivot-benchmarks/ssb/:
#   s3://pivot-benchmarks/ssb/sf1/           SF1,   ~0.4 GB (smoke tests)
#   s3://pivot-benchmarks/ssb/sf100/         SF100, ~30 GB, ssb-dbgen row order
#   s3://pivot-benchmarks/ssb/sf100-sorted/  SF100, ~18 GB, each table sorted
#                                            by its ClickHouse ORDER BY key
#
# The sorted dataset is the interesting one for scan pruning: lineorder is
# globally sorted by (lo_orderdate, lo_orderkey) and split into ~1 GB files,
# so row-group min/max stats carve the sort key into narrow ranges. Both
# datasets hold the same rows, and the committed qNN.tsv oracles match either.
#
# Provenance, should a dataset ever need regenerating at another scale
# factor: the tables were generated with ssb-dbgen
# (github.com/eyalroz/ssb-dbgen, cmake build; lineorder as parallel `-C/-S`
# chunks) and the pipe-separated .tbl output converted per table with
# DuckDB's read_csv into parquet. The sorted variant is one DuckDB
# `COPY (SELECT * FROM read_parquet(...) ORDER BY <sort key>) TO '<table>'
# (FORMAT PARQUET, FILE_SIZE_BYTES '1GB')` per table, with the ClickHouse
# sort keys: lineorder (lo_orderdate, lo_orderkey), customer c_custkey,
# supplier s_suppkey, part p_partkey, date d_datekey.
#
# Usage:
#   ./prep-ssb-data.sh                                   # sorted → ~/ssb-sf100-sorted
#   ./prep-ssb-data.sh --dataset sf100 --root ~/ssb-sf100
#
# Then run the suite (from the crate root):
#   cargo run --release -- --suite ssb --source <root> --iterations 3

set -euo pipefail

dataset="sf100-sorted"
root=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) dataset="$2"; shift 2 ;;
        --root)    root="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

root="${root:-$HOME/ssb-$dataset}"

mkdir -p "$root"
aws s3 sync "s3://pivot-benchmarks/ssb/$dataset/" "$root/"
echo "ready: --suite ssb --source $root"
