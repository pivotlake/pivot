#!/usr/bin/env bash
#
# run-duckdb.sh — run the TPC-H suite through DuckDB for a side-by-side
# comparison with pivot-bench, over the SAME normalized parquet directories
# (see setup.sql / prep-tpch-data.sh). Each base table is exposed as a view, so
# the unmodified official qNN.sql files run as-is.
#
# Usage:
#   ./run-duckdb.sh --source ~/tpch-sf100                    # all queries, 1 run
#   ./run-duckdb.sh --source ~/tpch-sf100 --query 12         # just q12
#   ./run-duckdb.sh --source ~/tpch-sf100 --iterations 3     # 3 timed runs each
#   ./run-duckdb.sh --source ~/tpch-sf100 --no-drop-caches   # skip the cache drop
#   ./run-duckdb.sh --source ~/tpch-sf100 --query 12 --write-expected  # write q12.tsv
#   ./run-duckdb.sh --source ~/tpch-sf100 --duckdb-process single
#
# We always report DuckDB's own `.timer` "Run Time" (query execution only).
#
# --duckdb-process per-iteration|single  (default per-iteration)
#   per-iteration: a FRESH duckdb process per timed iteration (engine-cold,
#     page cache warm after iteration 1).
#   single: all of a query's iterations in ONE duckdb process, so iterations
#     2+ reuse DuckDB's warm buffer pool — symmetric with pivot's warm server.
#
# The OS page cache is dropped once before each query (needs root, via sudo),
# so iteration 1 is a true cold read; --no-drop-caches skips that.
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
drop_caches=1
write_expected=0
sleep_ms=0
duckdb_process="per-iteration"

usage() {
    sed -n '3,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --duckdb-process) duckdb_process="$2"; shift 2 ;;
        --no-drop-caches) drop_caches=0; shift ;;
        --write-expected) write_expected=1; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$duckdb_process" in
    per-iteration|single) ;;
    *) echo "error: --duckdb-process must be per-iteration|single (got '$duckdb_process')" >&2; usage 1 ;;
esac

[[ -n "$source_path" ]] || { echo "error: --source is required" >&2; usage 1; }
[[ -d "$source_path" ]] || { echo "error: --source must be the dataset root directory" >&2; exit 1; }

command -v duckdb >/dev/null 2>&1 || {
    echo "error: duckdb not found on PATH" >&2
    exit 1
}

flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

# One view per base table over its parquet directory.
tables=(lineitem orders customer part partsupp supplier nation region)
setup=""
for t in "${tables[@]}"; do
    setup+="CREATE VIEW $t AS SELECT * FROM read_parquet('${source_path%/}/$t/*.parquet');"$'\n'
done

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

# Persist the views into a throwaway .db once, then open it fresh each
# iteration with parquet metadata caching on, so a timed run doesn't pay the
# view re-creation.
if [[ "$write_expected" != "1" ]]; then
    query_db="$(mktemp -u)-tpch.db"
    trap 'rm -f "$query_db" "$query_db".wal' EXIT
    duckdb "$query_db" -c "$setup" >/dev/null 2>&1
    iter_cmd=(duckdb "$query_db" -c "SET parquet_metadata_cache=true" -c ".timer on")
fi

for f in "${query_files[@]}"; do
    stem="$(basename "$f" .sql)"
    [[ -f "$suite_dir/$stem-duckdb.sql" ]] && f="$suite_dir/$stem-duckdb.sql"
    sql="$(cat "$f")"
    echo "=== $stem ==="

    if [[ "$write_expected" == "1" ]]; then
        out_file="$suite_dir/$stem.tsv"
        printf '%s\n.headers off\n.nullvalue '\'''\''\n.mode tabs\n.output %s\n%s\n' \
            "$setup" "$out_file" "$sql" \
            | duckdb
        echo "  wrote $out_file ($(wc -l < "$out_file" | tr -d ' ') rows)"
        echo
        continue
    fi

    flush_page_cache
    if [[ "$duckdb_process" == "single" ]]; then
        cmd=("${iter_cmd[@]}")
        for ((i = 1; i <= iterations; i++)); do cmd+=(-c "$sql"); done
        "${cmd[@]}" 2>&1 | grep -E 'Run Time|Error' || true
    else
        for ((i = 1; i <= iterations; i++)); do
            "${iter_cmd[@]}" -c "$sql" 2>&1 | grep -E 'Run Time|Error' || true
            if [[ "${sleep_ms:-0}" -gt 0 && $i -lt $iterations ]]; then
                sleep "$(awk "BEGIN{print $sleep_ms/1000}")"
            fi
        done
    fi
    echo
done
