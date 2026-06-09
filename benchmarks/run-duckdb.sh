#!/usr/bin/env bash
#
# run-duckdb.sh — run benchmark suite queries through DuckDB for a side-by-side
# comparison with pivot-bench.
#
# Query files (qNN.sql) and the DuckDB setup template live in the suite
# directory. DuckDB reads the same parquet data pivot-bench points at via
# --source.
#
# Usage:
#   ./run-duckdb.sh --source ~/hits                       # clickbench, 1 run
#   ./run-duckdb.sh --suite tpch-flat --source ~/tpch-flat
#   ./run-duckdb.sh --source ~/hits --query 7,20          # just q07 and q20
#   ./run-duckdb.sh --source ~/hits --iterations 3        # 3 timed runs each
#   ./run-duckdb.sh --source ~/hits --iterations 3 --sleep 500   # 500ms between runs
#   ./run-duckdb.sh --source ~/hits --no-drop-caches      # skip the cache drop
#   ./run-duckdb.sh --source ~/hits --query 7 --write-expected   # write q07.tsv
#
# --source accepts a directory (globbed for *.parquet), a single .parquet
# file, or an explicit glob.
#
# Like ClickBench, the OS page cache is dropped once before each query (Linux
# only — needs root, via sudo), so iteration 1 is a true cold read and later
# iterations measure hot, cache-resident performance.
#
# --write-expected runs each query once and writes its rows to the suite's
# qNN.tsv in pivot-bench's exact wire format (tab-separated, no header, NULL as
# empty), so the expected file is an independent oracle rather than pivot
# grading its own output. No timing, no cache drop. Two caveats: the query must
# have a *total* ORDER BY (the comparison is exact-string, so any tie in row
# order diverges), and columns whose type mapping differs from pivot's setup.sql
# — chiefly EventDate, which this script rewrites to a real DATE — will not
# match; use pivot-bench --update-results for SELECT-*/date queries instead.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

suite="clickbench"
suite_dir_arg=""
source_path=""
queries=""
iterations=1
drop_caches=1
write_expected=0
sleep_ms=0

usage() {
    awk 'NR > 2 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)     suite="$2"; shift 2 ;;
        --suite-dir) suite_dir_arg="$2"; shift 2 ;;
        --source)     source_path="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --no-drop-caches) drop_caches=0; shift ;;
        --write-expected) write_expected=1; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$source_path" ]]; then
    echo "error: --source is required" >&2
    usage 1
fi

suite_dir="${suite_dir_arg:-$here/$suite}"
if [[ ! -d "$suite_dir" ]]; then
    echo "error: suite directory not found: $suite_dir" >&2
    exit 1
fi

duckdb_setup="$suite_dir/duckdb-setup.sql"
if [[ ! -f "$duckdb_setup" ]]; then
    echo "error: no DuckDB setup template at $duckdb_setup" >&2
    exit 1
fi

command -v duckdb >/dev/null 2>&1 || {
    echo "error: duckdb not found on PATH" >&2
    exit 1
}

# Flush the Linux page cache so the next query reads from disk, the way
# ClickBench measures a cold run. Needs root, so it goes through sudo.
flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

# Resolve --source into a read_parquet() argument. A directory becomes a
# *.parquet glob; a file or explicit glob is passed through.
if [[ -d "$source_path" ]]; then
    parquet_glob="${source_path%/}/*.parquet"
else
    parquet_glob="$source_path"
fi

# Build the list of qNN.sql files to run.
declare -a query_files=()
if [[ -n "$queries" ]]; then
    IFS=',' read -ra ids <<< "$queries"
    for id in "${ids[@]}"; do
        id="${id#q}"                       # accept both "7" and "q07"
        printf -v stem 'q%02d' "$((10#$id))"
        f="$suite_dir/$stem.sql"
        [[ -f "$f" ]] || { echo "error: no query file $f" >&2; exit 1; }
        query_files+=("$f")
    done
else
    for f in "$suite_dir"/q*.sql; do
        [[ "$f" == *-duckdb.sql ]] && continue   # overrides, not standalone queries
        query_files+=("$f")
    done
fi

echo "duckdb $(duckdb --version)"
echo "suite: $suite"
echo "source: $parquet_glob"
if [[ "$write_expected" == "1" ]]; then
    echo "queries: ${#query_files[@]}, writing expected .tsv (no timing)"
else
    echo "queries: ${#query_files[@]}, iterations: $iterations"
fi
echo

setup_template="$(<"$duckdb_setup")"
setup="${setup_template//\{source\}/$parquet_glob}"

for f in "${query_files[@]}"; do
    stem="$(basename "$f" .sql)"
    # Prefer a DuckDB-specific override (e.g. q42-duckdb.sql) when one exists, so
    # a query can differ for DuckDB (e.g. wrapping EventTime in toDateTime) while
    # the shared qNN.sql stays the one pivot runs.
    [[ -f "$suite_dir/$stem-duckdb.sql" ]] && f="$suite_dir/$stem-duckdb.sql"
    sql="$(cat "$f")"
    echo "=== $stem ==="

    if [[ "$write_expected" == "1" ]]; then
        # Emit the result rows in pivot-bench's wire format so the .tsv is a
        # ground-truth oracle. `.mode tabs` is tab-separated with no quoting (the
        # same naive concatenation pivot's collect_tsv does), `.headers off` drops
        # the column-name row, and `.nullvalue ''` renders NULL as an empty field
        # — matching pivot's pgwire output, where a NULL column contributes
        # nothing between the tabs. Errors still surface on stderr.
        out_file="$suite_dir/$stem.tsv"
        printf '%s\n.headers off\n.nullvalue '\'''\''\n.mode tabs\n.output %s\n%s\n' \
            "$setup" "$out_file" "$sql" \
            | duckdb
        echo "  wrote $out_file ($(wc -l < "$out_file" | tr -d ' ') rows)"
        echo
        continue
    fi

    # Drop the page cache once per query: iteration 1 then reads cold from disk,
    # and any later iterations measure hot, cache-resident performance —
    # pivot-bench's cold/hot split.
    flush_page_cache
    # Run every iteration in ONE duckdb process with the Parquet metadata cache
    # on — matching ClickBench's run.sh (and pivot's single warm server session):
    # iteration 1 is cold, later iterations are true hot. A fresh process per
    # iteration (the old approach) re-parsed all ~100 files' Parquet metadata on
    # every run — a fixed ~15ms tax that unfairly inflated every "hot" DuckDB
    # time relative to pivot's warm session. `.output /dev/null` discards result
    # rows (q23 is SELECT *); `.timer on` leaves the per-run timing line. (No
    # inter-iteration sleep: the runs are back-to-back in-process, as upstream.)
    {
        printf '%s\nSET parquet_metadata_cache=true;\n.output /dev/null\n.timer on\n' "$setup"
        for ((i = 1; i <= iterations; i++)); do
            printf '%s\n' "$sql"
        done
    } | duckdb 2>&1 | grep -E 'Run Time|Error' || true
    echo
done
