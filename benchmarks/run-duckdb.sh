#!/usr/bin/env bash
#
# run-duckdb.sh — run ClickBench queries through DuckDB for a side-by-side
# comparison with pivot-bench.
#
# The query files (qNN.sql) and schema live in benchmarks/clickbench/. DuckDB
# reads the same parquet data pivot-bench points at via --source, exposed as a
# `hits` view so the unmodified qNN.sql files run as-is.
#
# Usage:
#   ./run-duckdb.sh --source ~/hits                       # all queries, 1 run
#   ./run-duckdb.sh --source ~/hits --query 7,20          # just q07 and q20
#   ./run-duckdb.sh --source ~/hits --iterations 3        # 3 timed runs each
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

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/clickbench"

source_path=""
queries=""
iterations=1
drop_caches=1
write_expected=0

usage() {
    sed -n '3,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
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
        query_files+=("$f")
    done
fi

echo "duckdb $(duckdb --version)"
echo "source: $parquet_glob"
if [[ "$write_expected" == "1" ]]; then
    echo "queries: ${#query_files[@]}, writing expected .tsv (no timing)"
else
    echo "queries: ${#query_files[@]}, iterations: $iterations"
fi
echo

# Official ClickBench DuckDB setup. `binary_as_string=True` decodes the parquet
# string columns (stored as BLOB) as VARCHAR so `URL LIKE ...` binds, and
# EventDate (days since epoch) is turned into a real DATE. `toDateTime` is the
# macro the ClickBench queries use to read the packed-seconds timestamp columns.
setup="CREATE VIEW hits AS
SELECT *
    REPLACE (make_date(EventDate) AS EventDate)
FROM read_parquet('${parquet_glob}', binary_as_string=True);
CREATE MACRO toDateTime(t) AS epoch_ms(t * 1000);"

for f in "${query_files[@]}"; do
    stem="$(basename "$f" .sql)"
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
    for ((i = 1; i <= iterations; i++)); do
        # Fresh process per iteration keeps DuckDB's own caches from making
        # later runs look artificially hot. `.timer on` measures the query;
        # `.output /dev/null` throws away the result rows (q23 is SELECT *)
        # while leaving the timing line on stderr.
        printf '%s\n.output /dev/null\n.timer on\n%s\n' "$setup" "$sql" \
            | duckdb 2>&1 \
            | grep -E 'Run Time|Error' || true
    done
    echo
done
