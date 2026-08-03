#!/usr/bin/env bash
# prep-tpch-data.sh — sync the normalized TPC-H parquet dataset from S3 to a
# local directory (one subdirectory per table), the layout `setup.sql` expects.
#
# The canonical datasets (generated with tpchgen-cli, see the schema comment in
# setup.sql):
#   s3://epsio-tpch/sf10/                      SF10,  ~3.9 GB
#   s3://epsio-tpch/sf100/                     SF100, ~41.5 GB, 7 MiB row groups
#   s3://epsio-tpch/sf100-large-row-groups/    SF100, ~35.8 GB, 128 MiB row groups
#
# And the same data as pivot's own writer produces it, which is what a table
# looks like after an INSERT rather than what another tool wrote:
#   s3://epsio-tpch/sf100-pivot/               SF100, ~26 GB, 100k-row row groups
#
# Every dataset is one directory per table, so the suite reads them all the same
# way and so does any other engine. `sf100-pivot` also carries a `_delta_log`
# per table, left by the `CREATE TABLE` that last read it; the runner clears
# those before each run and the tables' parquet files are the same either way.
#
# Usage:
#   ./prep-tpch-data.sh                                  # sf100 → ~/tpch-sf100
#   ./prep-tpch-data.sh --dataset sf10 --root ~/tpch-sf10
#   ./prep-tpch-data.sh --dataset sf100-pivot --root ~/tpch-sf100-pivot
#
# Then run the suite (from the crate root):
#   cargo run --release -- --suite tpch --source <root> --iterations 3

set -euo pipefail

dataset="sf100"
root=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) dataset="$2"; shift 2 ;;
        --root)    root="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

root="${root:-$HOME/tpch-$dataset}"

mkdir -p "$root"
aws s3 sync "s3://epsio-tpch/$dataset/" "$root/"
echo "ready: --suite tpch --source $root"
