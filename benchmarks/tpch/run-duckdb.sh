#!/usr/bin/env bash
#
# run-duckdb.sh — run the TPC-H suite through DuckDB for a side-by-side
# comparison with pivot-bench, either over the SAME normalized parquet
# directories pivot reads (see setup.sql / prep-tpch-data.sh; each base table
# is exposed as a view, so the unmodified official qNN.sql files run as-is),
# or over a native DuckDB database of the same data.
#
# Usage:
#   ./run-duckdb.sh --source ~/tpch-sf100                    # all queries, 1 run
#   ./run-duckdb.sh --source ~/tpch-sf100 --query 12         # just q12
#   ./run-duckdb.sh --source ~/tpch-sf100 --iterations 3     # 3 timed runs each
#   ./run-duckdb.sh --source ~/tpch-sf100 --sleep 500        # 500ms between queries
#   ./run-duckdb.sh --source ~/tpch-sf100 --drop-caches-between-queries
#   ./run-duckdb.sh --source ~/tpch-sf100 --query 12 --write-expected  # write q12.tsv
#   ./run-duckdb.sh --data native --source /mnt/nvme/tpch-native.duckdb --query 12
#
# We always report DuckDB's own `.timer` "Run Time" (query execution only).
# The whole suite goes to ONE duckdb process over stdin, so every query after
# the first reuses the buffer pool its predecessors warmed, the way pivot-bench's
# long-lived server does.
#
# --data parquet|native  (default parquet)
#   parquet: --source is the dataset root directory; each table is a view over
#     its parquet files.
#   native:  --source is a .duckdb database file holding the base tables (see
#     fetch-native-dbs.sh); opened read-only.
#
# --sleep <ms> waits that long between queries, so one query's tail (background
# threads winding down, dirty pages flushing) doesn't land inside the next
# one's timing. DuckDB runs each statement as it arrives on stdin, so the pause
# really does fall between queries. Matches pivot-bench's --sleep.
#
# The OS page cache is always dropped once before the first query (needs root,
# via sudo; a failure warns and continues), so a run never inherits the cache
# state of whatever ran before it. Between queries it is left alone, so each
# query sees what its predecessors warmed, which is what a power run wants.
# --drop-caches-between-queries drops before every query instead, making each
# one a cold read.
#
# --write-expected runs each query once and writes its rows to the suite's
# qNN.tsv in pivot-bench's exact wire format (tab-separated, no header, NULL as
# empty), so the expected file is an independent oracle rather than pivot
# grading its own output. Queries must carry a total ORDER BY.

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source_path=""
queries=""
iterations=1
drop_between=0
write_expected=0
sleep_ms=0
data="parquet"

usage() {
    sed -n '3,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --data)       data="$2"; shift 2 ;;
        --drop-caches-between-queries) drop_between=1; shift ;;
        --write-expected) write_expected=1; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$data" in
    parquet|native) ;;
    *) echo "error: --data must be parquet|native (got '$data')" >&2; usage 1 ;;
esac

[[ -n "$source_path" ]] || { echo "error: --source is required" >&2; usage 1; }
if [[ "$data" == "parquet" ]]; then
    [[ -d "$source_path" ]] || { echo "error: --source must be the dataset root directory" >&2; exit 1; }
else
    [[ -f "$source_path" ]] || { echo "error: --source must be a .duckdb database file" >&2; exit 1; }
fi

command -v duckdb >/dev/null 2>&1 || {
    echo "error: duckdb not found on PATH" >&2
    exit 1
}

flush_page_cache() {
    sync
    if ! echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
        echo "warning: could not drop the OS page cache (needs sudo); continuing" >&2
    fi
}

# One view per base table over its parquet directory; a native database
# already holds the base tables and needs no setup.
setup=""
if [[ "$data" == "parquet" ]]; then
    tables=(lineitem orders customer part partsupp supplier nation region)
    for t in "${tables[@]}"; do
        setup+="CREATE VIEW $t AS SELECT * FROM read_parquet('${source_path%/}/$t/*.parquet');"$'\n'
    done
fi

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
        [[ "$f" == *-duckdb.sql ]] && continue
        query_files+=("$f")
    done
fi

echo "duckdb $(duckdb --version)"
echo "source: $source_path"
if [[ "$write_expected" == "1" ]]; then
    echo "queries: ${#query_files[@]}, writing expected .tsv (no timing)"
else
    echo "queries: ${#query_files[@]}, iterations: $iterations"
fi
echo

if [[ "$write_expected" == "1" ]]; then
    for f in "${query_files[@]}"; do
        stem="$(basename "$f" .sql)"
        [[ -f "$suite_dir/$stem-duckdb.sql" ]] && f="$suite_dir/$stem-duckdb.sql"
        out_file="$suite_dir/$stem.tsv"
        expected_cmd=(duckdb)
        [[ "$data" == "native" ]] && expected_cmd=(duckdb -readonly "$source_path")
        echo "=== $stem ==="
        printf '%s\n.headers off\n.nullvalue '\'''\''\n.mode tabs\n.output %s\n%s\n' \
            "$setup" "$out_file" "$(cat "$f")" \
            | "${expected_cmd[@]}"
        echo "  wrote $out_file ($(wc -l < "$out_file" | tr -d ' ') rows)"
        echo
    done
    exit 0
fi

# Parquet mode persists the views into a throwaway .db first, so the timed run
# doesn't pay the view creation and the page-cache drop below clears whatever
# creating them read. Native mode opens the database itself, read-only so a run
# can never dirty it.
if [[ "$data" == "parquet" ]]; then
    query_db="$(mktemp -u)-tpch.db"
    trap 'rm -f "$query_db" "$query_db".wal' EXIT
    duckdb "$query_db" -c "$setup" >/dev/null 2>&1
    db_cmd=(duckdb "$query_db")
    prologue=("SET parquet_metadata_cache=true;" ".timer on")
else
    db_cmd=(duckdb -readonly "$source_path")
    prologue=(".timer on")
fi

# Write the whole suite to ONE duckdb process over stdin. DuckDB runs each
# statement as it arrives, so pausing here pauses between queries *inside* that
# process: the buffer pool one query warms is still there for the next, which is
# what pivot-bench's long-lived server gives its side. The `.print` markers ride
# the same output stream as the timings, so they stay in order with them.
feed_suite() {
    printf '%s\n' "${prologue[@]}"
    for idx in "${!query_files[@]}"; do
        f="${query_files[$idx]}"
        stem="$(basename "$f" .sql)"
        [[ -f "$suite_dir/$stem-duckdb.sql" ]] && f="$suite_dir/$stem-duckdb.sql"
        if [[ "$idx" -gt 0 ]]; then
            if [[ "$drop_between" == "1" ]]; then
                flush_page_cache
            fi
            if [[ "${sleep_ms:-0}" -gt 0 ]]; then
                sleep "$(awk "BEGIN{print $sleep_ms/1000}")"
            fi
        fi
        printf '.print === %s ===\n' "$stem"
        for ((i = 1; i <= iterations; i++)); do
            cat "$f"
            printf '\n'
        done
    done
    return 0
}

# Always start from a cold page cache, so a run never inherits whatever ran
# before it.
flush_page_cache
feed_suite | "${db_cmd[@]}" 2>&1 | grep -E '^=== |Run Time|Error' || true
