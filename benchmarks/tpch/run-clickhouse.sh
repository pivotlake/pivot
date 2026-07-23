#!/usr/bin/env bash
#
# run-clickhouse.sh - run the TPC-H suite through ClickHouse for a side-by-side
# comparison with pivot-bench, over a native ClickHouse data directory holding
# the base tables (see fetch-native-dbs.sh).
#
# Usage:
#   ./run-clickhouse.sh --source /mnt/nvme/clickhouse --query 12
#   ./run-clickhouse.sh --source /mnt/nvme/clickhouse --query 12 --iterations 3
#   ./run-clickhouse.sh --source /mnt/nvme/clickhouse --no-drop-caches
#
# The script starts its own clickhouse server on the data directory and stops
# it when done. Before each query the server is restarted and the OS page
# cache dropped (needs root, via sudo), so iteration 1 is engine-cold and
# disk-cold; iterations 2+ reuse the running server's warm state - symmetric
# with run-duckdb.sh's single-process mode and pivot's warm server.
# --no-drop-caches skips the cache drop (the restart still happens).
#
# A query runs from its qNN.sql file, or qNN-clickhouse.sql when present for
# dialect differences. We report the client's own elapsed time.

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source_path=""
queries=""
iterations=1
drop_caches=1
sleep_ms=0

usage() {
    sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --no-drop-caches) drop_caches=0; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$source_path" ]] || { echo "error: --source is required" >&2; usage 1; }
[[ -d "$source_path/metadata" ]] || {
    echo "error: --source must be a clickhouse data directory" >&2
    exit 1
}

command -v clickhouse >/dev/null 2>&1 || {
    echo "error: clickhouse not found on PATH" >&2
    exit 1
}

flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

server_pid=""
server_log="$(mktemp)-clickhouse-server.log"

stop_server() {
    [[ -n "$server_pid" ]] || return 0
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
}
trap 'stop_server; rm -f "$server_log"' EXIT

start_server() {
    stop_server
    clickhouse server -- --path="${source_path%/}/" --listen_host=127.0.0.1 \
        >"$server_log" 2>&1 &
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

echo "clickhouse $(clickhouse client --version 2>/dev/null | grep -oE '[0-9.]+' | head -1)"
echo "source: $source_path"
echo "queries: ${#query_files[@]}, iterations: $iterations"
echo

for f in "${query_files[@]}"; do
    stem="$(basename "$f" .sql)"
    [[ -f "$suite_dir/$stem-clickhouse.sql" ]] && f="$suite_dir/$stem-clickhouse.sql"
    sql="$(cat "$f")"
    echo "=== $stem ==="

    flush_page_cache
    start_server
    for ((i = 1; i <= iterations; i++)); do
        clickhouse client --time --query "$sql" >/dev/null 2>/tmp/ch-time.$$ \
            && echo "Run Time (s): real $(cat /tmp/ch-time.$$)" \
            || { echo "Error:"; cat /tmp/ch-time.$$; }
        rm -f /tmp/ch-time.$$
        if [[ "${sleep_ms:-0}" -gt 0 && $i -lt $iterations ]]; then
            sleep "$(awk "BEGIN{print $sleep_ms/1000}")"
        fi
    done
    echo
done
