#!/usr/bin/env bash
# fetch-native-dbs.sh — restore the comparison engines' native TPC-H databases
# from S3 into place, so a fresh box (or a wiped instance store) can rerun the
# DuckDB and ClickHouse side-by-side comparisons without rebuilding them.
#
# The archives (zstd tars, built from SF100 loads of the $PIVOT_BENCH_S3/tpch/sf100/
# parquet dataset):
#   $PIVOT_BENCH_S3/tpch/native/tpch-native-duckdb.tar.zst   -> <root>/tpch-native.duckdb
#   $PIVOT_BENCH_S3/tpch/native/clickhouse-tpch.tar.zst      -> <root>/clickhouse/
#
# Usage:
#   ./fetch-native-dbs.sh                      # both, into /mnt/nvme
#   ./fetch-native-dbs.sh --root /data         # both, into /data
#   ./fetch-native-dbs.sh --only duckdb        # just the DuckDB database
#   ./fetch-native-dbs.sh --only clickhouse    # just the ClickHouse data dir
#
# ClickHouse expects its server's data path to point at <root>/clickhouse.

set -euo pipefail

root="/mnt/nvme"
only=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) root="$2"; shift 2 ;;
        --only) only="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

fetch() {
    local archive="$1"
    bucket="${PIVOT_BENCH_S3:?set PIVOT_BENCH_S3 to the S3 prefix holding the datasets, e.g. s3://my-bucket}"
    aws s3 cp "$bucket/tpch/native/$archive" - --only-show-errors \
        | tar -I "zstd -T0" -xf - -C "$root"
    echo "restored: $root ($archive)"
}

mkdir -p "$root"
[[ -z "$only" || "$only" == "duckdb" ]] && fetch tpch-native-duckdb.tar.zst
[[ -z "$only" || "$only" == "clickhouse" ]] && fetch clickhouse-tpch.tar.zst
