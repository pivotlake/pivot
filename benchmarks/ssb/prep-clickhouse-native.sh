#!/usr/bin/env bash
#
# prep-clickhouse-native.sh — load the SSB parquet dataset (see
# prep-ssb-data.sh) into a native ClickHouse MergeTree data directory that
# run-clickhouse.sh can serve.
#
# Usage:
#   ./prep-clickhouse-native.sh --source ~/ssb-sf100 --output ~/clickhouse-ssb
#
# Table and column names are lowercase so the suite's qNN.sql files run
# unmodified; engine and sorting keys follow ClickHouse's published SSB
# schema (lineorder ordered by (lo_orderdate, lo_orderkey), dimensions by
# their key).

set -euo pipefail

source_path=""
output=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) source_path="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        -h|--help) sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$source_path" && -n "$output" ]] || { echo "error: --source and --output are required" >&2; exit 1; }
command -v clickhouse >/dev/null || { echo "error: clickhouse not found on PATH" >&2; exit 1; }

source_path="$(cd "$source_path" && pwd)"
mkdir -p "$output"

server_log="$(mktemp)"
clickhouse server -- --path="${output%/}/" --listen_host=127.0.0.1 \
    --user_files_path="$source_path" >"$server_log" 2>&1 &
server_pid=$!
trap 'kill $server_pid 2>/dev/null; wait $server_pid 2>/dev/null; rm -f "$server_log"' EXIT

for _ in $(seq 1 120); do
    clickhouse client --query "SELECT 1" >/dev/null 2>&1 && break
    kill -0 "$server_pid" 2>/dev/null || { tail -5 "$server_log" >&2; exit 1; }
    sleep 0.5
done

clickhouse client --multiquery <<'SQL'
CREATE TABLE lineorder (
    lo_orderkey UInt32,
    lo_linenumber UInt8,
    lo_custkey UInt32,
    lo_partkey UInt32,
    lo_suppkey UInt32,
    lo_orderdate UInt32,
    lo_orderpriority String,
    lo_shippriority UInt8,
    lo_quantity UInt8,
    lo_extendedprice UInt32,
    lo_ordtotalprice UInt32,
    lo_discount UInt8,
    lo_revenue UInt32,
    lo_supplycost UInt32,
    lo_tax UInt8,
    lo_commitdate UInt32,
    lo_shipmode String
) ENGINE = MergeTree ORDER BY (lo_orderdate, lo_orderkey);

CREATE TABLE customer (
    c_custkey UInt32,
    c_name String,
    c_address String,
    c_city String,
    c_nation String,
    c_region String,
    c_phone String,
    c_mktsegment String
) ENGINE = MergeTree ORDER BY (c_custkey);

CREATE TABLE supplier (
    s_suppkey UInt32,
    s_name String,
    s_address String,
    s_city String,
    s_nation String,
    s_region String,
    s_phone String
) ENGINE = MergeTree ORDER BY (s_suppkey);

CREATE TABLE part (
    p_partkey UInt32,
    p_name String,
    p_mfgr String,
    p_category String,
    p_brand1 String,
    p_color String,
    p_type String,
    p_size UInt8,
    p_container String
) ENGINE = MergeTree ORDER BY (p_partkey);

CREATE TABLE date (
    d_datekey UInt32,
    d_date String,
    d_dayofweek String,
    d_month String,
    d_year UInt16,
    d_yearmonthnum UInt32,
    d_yearmonth String,
    d_daynuminweek UInt8,
    d_daynuminmonth UInt8,
    d_daynuminyear UInt16,
    d_monthnuminyear UInt8,
    d_weeknuminyear UInt8,
    d_sellingseason String,
    d_lastdayinweekfl UInt8,
    d_lastdayinmonthfl UInt8,
    d_holidayfl UInt8,
    d_weekdayfl UInt8
) ENGINE = MergeTree ORDER BY (d_datekey);

INSERT INTO lineorder SELECT * FROM file('lineorder/*.parquet', Parquet);
INSERT INTO customer SELECT * FROM file('customer/*.parquet', Parquet);
INSERT INTO supplier SELECT * FROM file('supplier/*.parquet', Parquet);
INSERT INTO part SELECT * FROM file('part/*.parquet', Parquet);
INSERT INTO date SELECT * FROM file('date/*.parquet', Parquet);
SQL

for t in lineorder customer supplier part date; do
    n=$(clickhouse client --query "SELECT count(*) FROM $t")
    echo "  $t: $n rows"
done
