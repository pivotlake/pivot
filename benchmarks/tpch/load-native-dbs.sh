#!/usr/bin/env bash
#
# load-native-dbs.sh - load a TPC-H parquet dataset (one directory per table,
# the layout setup.sql reads) into the comparison engines' native storage, so
# run-duckdb.sh --data native and run-clickhouse.sh query each engine on its own
# format rather than through its parquet reader:
#   <root>/tpch-native.duckdb   a DuckDB database holding the 8 base tables
#   <root>/clickhouse/          a ClickHouse data directory, MergeTree tables
#                               ordered by each table's primary key, merged
#                               with OPTIMIZE FINAL after the load
#
# Column types follow the parquet files (BIGINT keys, DECIMAL(15,2) money,
# DATE dates), so both engines hold exactly the rows pivot reads.
#
# Usage:
#   ./load-native-dbs.sh --source ~/tpch-sf100 --root /mnt/nvme
#   ./load-native-dbs.sh --source ~/tpch-sf100 --root /mnt/nvme --only clickhouse

set -euo pipefail

source_path=""
root=""
only=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) source_path="$2"; shift 2 ;;
        --root)   root="$2"; shift 2 ;;
        --only)   only="$2"; shift 2 ;;
        -h|--help) sed -n '3,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -d "$source_path/lineitem" ]] || { echo "error: --source must be a TPC-H dataset root (no $source_path/lineitem)" >&2; exit 1; }
[[ -n "$root" ]] || { echo "error: --root is required" >&2; exit 1; }
case "$only" in
    ""|duckdb|clickhouse) ;;
    *) echo "error: --only must be duckdb or clickhouse (got '$only')" >&2; exit 1 ;;
esac
source_path="$(cd "$source_path" && pwd)"
mkdir -p "$root"
root="$(cd "$root" && pwd)"

tables=(lineitem orders customer part partsupp supplier nation region)

load_duckdb() {
    command -v duckdb >/dev/null || { echo "error: duckdb not found on PATH" >&2; exit 1; }
    local db="$root/tpch-native.duckdb"
    rm -f "$db" "$db.wal"
    local sql="SET preserve_insertion_order = false;"$'\n'
    for t in "${tables[@]}"; do
        sql+="CREATE TABLE $t AS FROM read_parquet('$source_path/$t/*.parquet');"$'\n'
    done
    sql+="CHECKPOINT;"
    local start=$SECONDS
    duckdb "$db" -c "$sql"
    echo "loaded: $db ($((SECONDS - start))s, $(du -sh "$db" | cut -f1))"
}

clickhouse_schema() {
    cat <<'SQL'
CREATE TABLE lineitem (
    l_orderkey Int64, l_partkey Int64, l_suppkey Int64, l_linenumber Int32,
    l_quantity Decimal(15,2), l_extendedprice Decimal(15,2),
    l_discount Decimal(15,2), l_tax Decimal(15,2),
    l_returnflag String, l_linestatus String,
    l_shipdate Date, l_commitdate Date, l_receiptdate Date,
    l_shipinstruct String, l_shipmode String, l_comment String
) ENGINE = MergeTree ORDER BY (l_orderkey, l_linenumber);
CREATE TABLE orders (
    o_orderkey Int64, o_custkey Int64, o_orderstatus String,
    o_totalprice Decimal(15,2), o_orderdate Date, o_orderpriority String,
    o_clerk String, o_shippriority Int32, o_comment String
) ENGINE = MergeTree ORDER BY o_orderkey;
CREATE TABLE customer (
    c_custkey Int64, c_name String, c_address String, c_nationkey Int64,
    c_phone String, c_acctbal Decimal(15,2), c_mktsegment String, c_comment String
) ENGINE = MergeTree ORDER BY c_custkey;
CREATE TABLE part (
    p_partkey Int64, p_name String, p_mfgr String, p_brand String, p_type String,
    p_size Int32, p_container String, p_retailprice Decimal(15,2), p_comment String
) ENGINE = MergeTree ORDER BY p_partkey;
CREATE TABLE partsupp (
    ps_partkey Int64, ps_suppkey Int64, ps_availqty Int32,
    ps_supplycost Decimal(15,2), ps_comment String
) ENGINE = MergeTree ORDER BY (ps_partkey, ps_suppkey);
CREATE TABLE supplier (
    s_suppkey Int64, s_name String, s_address String, s_nationkey Int64,
    s_phone String, s_acctbal Decimal(15,2), s_comment String
) ENGINE = MergeTree ORDER BY s_suppkey;
CREATE TABLE nation (
    n_nationkey Int64, n_name String, n_regionkey Int64, n_comment String
) ENGINE = MergeTree ORDER BY n_nationkey;
CREATE TABLE region (
    r_regionkey Int64, r_name String, r_comment String
) ENGINE = MergeTree ORDER BY r_regionkey;
SQL
}

load_clickhouse() {
    command -v clickhouse >/dev/null || { echo "error: clickhouse not found on PATH" >&2; exit 1; }
    local data_dir="$root/clickhouse"
    local server_log="$root/clickhouse-load.log"
    rm -rf "$data_dir"
    mkdir -p "$data_dir"
    # Started from its data directory, where it writes its preprocessed config.
    (cd "$data_dir" && exec clickhouse server -- --path="$data_dir/" \
        --listen_host=127.0.0.1 --user_files_path="$source_path/") >"$server_log" 2>&1 &
    local server_pid=$!
    # shellcheck disable=SC2064
    trap "kill $server_pid 2>/dev/null; wait $server_pid 2>/dev/null" EXIT
    local ready=0
    for _ in $(seq 1 120); do
        if clickhouse client --query "SELECT 1" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 0.5
    done
    [[ "$ready" == "1" ]] || { echo "error: clickhouse server did not start, log tail:" >&2; tail -5 "$server_log" >&2; exit 1; }

    # The load and the merge are single statements that report no progress
    # for minutes at the largest scales; keep the client from timing out.
    local client=(clickhouse client --receive_timeout=86400 --send_timeout=86400)
    local start=$SECONDS
    clickhouse_schema | "${client[@]}" --multiquery
    for t in "${tables[@]}"; do
        "${client[@]}" --query "INSERT INTO $t SELECT * FROM file('$t/*.parquet', Parquet)"
    done
    echo "clickhouse: inserted in $((SECONDS - start))s, merging"
    for t in "${tables[@]}"; do
        "${client[@]}" --query "OPTIMIZE TABLE $t FINAL"
    done
    "${client[@]}" --query "
        SELECT table, sum(rows), count(), formatReadableSize(sum(bytes_on_disk))
        FROM system.parts WHERE active AND database = 'default'
        GROUP BY table ORDER BY table" --format PrettyCompactMonoBlock

    kill "$server_pid"
    wait "$server_pid" 2>/dev/null || true
    trap - EXIT
    echo "loaded: $data_dir ($((SECONDS - start))s)"
}

[[ -z "$only" || "$only" == "duckdb" ]] && load_duckdb
[[ -z "$only" || "$only" == "clickhouse" ]] && load_clickhouse
exit 0
