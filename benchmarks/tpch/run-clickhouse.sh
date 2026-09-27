#!/usr/bin/env bash
#
# run-clickhouse.sh - run the TPC-H suite through ClickHouse for a side-by-side
# comparison with pivot-bench, either over a native ClickHouse data directory
# holding the base tables as MergeTree (see load-native-dbs.sh or
# fetch-native-dbs.sh), or over the SAME normalized parquet directories pivot
# reads.
#
# Usage:
#   ./run-clickhouse.sh --source /mnt/nvme/clickhouse --query 12
#   ./run-clickhouse.sh --source /mnt/nvme/clickhouse --query 12 --iterations 3
#   ./run-clickhouse.sh --source /mnt/nvme/clickhouse --no-drop-caches
#   ./run-clickhouse.sh --source /mnt/nvme/clickhouse --clickhouse-process per-run --sleep 500
#   ./run-clickhouse.sh --data parquet --source ~/tpch-sf100
#
# The script starts its own clickhouse server and stops it when done.
#
# --data native|parquet  (default native)
#   native:  --source is a ClickHouse data directory; the server runs on it.
#   parquet: --source is the dataset root directory; the server runs on a
#     throwaway data directory under $TMPDIR (where it also spills), with one
#     File(Parquet) table per base table reading its directory in place.
#
# --clickhouse-process per-query|per-run  (default per-query)
#   per-query: before each query the server is restarted and the OS page
#     cache dropped (needs root, via sudo), so iteration 1 is engine-cold and
#     disk-cold; iterations 2+ reuse the running server's warm state.
#   per-run: ONE server executes the whole query list, the page cache dropped
#     once before it starts, so every query after the first reuses whatever
#     earlier ones left warm - symmetric with pivot's one server streaming the
#     same list and run-duckdb.sh's per-run mode.
# --no-drop-caches skips the cache drops (the server starts still happen).
#
# --sleep MS leaves a quiet gap before each query but the first, and between
# a query's iterations.
#
# --timeout SEC caps each query's execution server-side (max_execution_time);
# a query that hits it reports an error instead of a time.
#
# --suite-dir overrides where qNN.sql lives (default: this script's own
# directory), so a shipped copy of this script can run a checkout's suite.
#
# A query runs from its qNN.sql file, or qNN-clickhouse.sql when present for
# dialect differences. Every query runs with join_use_nulls = 1, the SQL
# semantics for the unmatched side of an outer join (q13 counts those NULLs).
# We report the client's own elapsed time.

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source_path=""
queries=""
iterations=1
drop_caches=1
sleep_ms=0
data="native"
clickhouse_process="per-query"
timeout_sec=0

usage() {
    sed -n '3,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --suite-dir)  suite_dir="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --timeout)    timeout_sec="$2"; shift 2 ;;
        --data)       data="$2"; shift 2 ;;
        --clickhouse-process) clickhouse_process="$2"; shift 2 ;;
        --no-drop-caches) drop_caches=0; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$clickhouse_process" in
    per-query|per-run) ;;
    *) echo "error: --clickhouse-process must be per-query|per-run (got '$clickhouse_process')" >&2; usage 1 ;;
esac

[[ -n "$source_path" ]] || { echo "error: --source is required" >&2; usage 1; }
case "$data" in
    native)
        [[ -d "$source_path/metadata" ]] || {
            echo "error: --source must be a clickhouse data directory" >&2
            exit 1
        }
        ;;
    parquet)
        [[ -d "$source_path/lineitem" ]] || {
            echo "error: --source must be the dataset root directory" >&2
            exit 1
        }
        ;;
    *) echo "error: --data must be native|parquet (got '$data')" >&2; usage 1 ;;
esac
source_path="$(cd "$source_path" && pwd)"

command -v clickhouse >/dev/null 2>&1 || {
    echo "error: clickhouse not found on PATH" >&2
    exit 1
}

flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

pause() {
    [[ "$sleep_ms" -gt 0 ]] || return 0
    sleep "$(awk "BEGIN{print $sleep_ms/1000}")"
}

server_dir="$source_path"
server_args=()
if [[ "$data" == "parquet" ]]; then
    server_dir="$(mktemp -d)"
    server_args=(--user_files_path="$source_path/")
fi
server_pid=""
server_log="$(mktemp --suffix=-clickhouse-server.log)"
time_file="$(mktemp)"

stop_server() {
    [[ -n "$server_pid" ]] || return 0
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
}
cleanup() {
    stop_server
    rm -f "$server_log" "$time_file"
    [[ "$data" == "parquet" ]] && rm -rf "$server_dir"
}
trap cleanup EXIT

start_server() {
    stop_server
    # Started from its data directory, where it writes its preprocessed config.
    (cd "$server_dir" && exec clickhouse server -- --path="$server_dir/" \
        --listen_host=127.0.0.1 "${server_args[@]}") >"$server_log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 120); do
        if clickhouse client --query "SELECT 1" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$server_pid" 2>/dev/null || {
            echo "error: clickhouse server died, log tail:" >&2
            tail -5 "$server_log" >&2
            exit 1
        }
        sleep 0.5
    done
    echo "error: clickhouse server not ready after 60s" >&2
    exit 1
}

# One File(Parquet) table per base table, reading its directory in place. The
# tables persist in the server's data directory across restarts. (Views over
# the file() table function trip a column-resolution bug on q12.)
parquet_tables_created=0
create_parquet_tables() {
    [[ "$data" == "parquet" && "$parquet_tables_created" == "0" ]] || return 0
    local t
    for t in lineitem orders customer part partsupp supplier nation region; do
        clickhouse client --query \
            "CREATE TABLE $t ENGINE = File(Parquet, '$source_path/$t/*.parquet')"
    done
    parquet_tables_created=1
}

# Build the list of qNN.sql files to run.
declare -a query_files=()
if [[ -n "$queries" ]]; then
    IFS=',' read -ra ids <<< "$queries"
    for id in "${ids[@]}"; do
        id="${id#q}"
        printf -v stem 'q%02d' "$((10#$id))"
        f="$suite_dir/$stem.sql"
        [[ -f "$f" ]] || { echo "error: no query file $f" >&2; exit 1; }
        query_files+=("$f")
    done
else
    for f in "$suite_dir"/q*.sql; do
        [[ "$f" == *-duckdb.sql || "$f" == *-clickhouse.sql ]] && continue
        query_files+=("$f")
    done
fi

client_settings=(--join_use_nulls=1)
[[ "$timeout_sec" -gt 0 ]] && client_settings+=(--max_execution_time="$timeout_sec")

echo "clickhouse $(clickhouse client --version 2>/dev/null | grep -oE '[0-9.]+' | head -1)"
echo "source: $source_path ($data)"
echo "queries: ${#query_files[@]}, iterations: $iterations, process: $clickhouse_process"
echo

if [[ "$clickhouse_process" == "per-run" ]]; then
    flush_page_cache
    start_server
    create_parquet_tables
fi

first=1
for f in "${query_files[@]}"; do
    stem="$(basename "$f" .sql)"
    [[ -f "$suite_dir/$stem-clickhouse.sql" ]] && f="$suite_dir/$stem-clickhouse.sql"
    sql="$(cat "$f")"
    [[ "$first" == "1" ]] || pause
    first=0
    echo "=== $stem ==="

    if [[ "$clickhouse_process" == "per-query" ]]; then
        flush_page_cache
        start_server
        create_parquet_tables
    fi
    for ((i = 1; i <= iterations; i++)); do
        if clickhouse client "${client_settings[@]}" --time --query "$sql" >/dev/null 2>"$time_file"; then
            echo "Run Time (s): real $(cat "$time_file")"
        else
            echo "Error:"
            cat "$time_file"
        fi
        [[ $i -lt $iterations ]] && pause
    done
    echo
done
