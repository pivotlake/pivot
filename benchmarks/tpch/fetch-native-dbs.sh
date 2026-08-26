#!/usr/bin/env bash
# fetch-native-dbs.sh — restore the comparison engines' native TPC-H databases
# from S3 into place, so a fresh box (or a wiped instance store) can rerun the
# DuckDB and ClickHouse side-by-side comparisons without rebuilding them.
#
# The archives (zstd tars, built from SF100 loads of the s3://pivot-benchmarks/tpch/sf100/
# parquet dataset):
#   s3://pivot-benchmarks/tpch/native/tpch-native-duckdb.tar.zst   -> <root>/tpch-native.duckdb
#   s3://pivot-benchmarks/tpch/native/clickhouse-tpch.tar.zst      -> <root>/clickhouse/
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
    aws s3 cp "s3://pivot-benchmarks/tpch/native/$archive" - --only-show-errors \
        | tar -I "zstd -T0" -xf - -C "$root"
    echo "restored: $root ($archive)"
}

mkdir -p "$root"
[[ -z "$only" || "$only" == "duckdb" ]] && fetch tpch-native-duckdb.tar.zst
[[ -z "$only" || "$only" == "clickhouse" ]] && fetch clickhouse-tpch.tar.zst
