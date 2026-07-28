#!/usr/bin/env bash
#
# run-duckdb.sh — run the JSONBench queries through DuckDB, for a side-by-side
# comparison with pivot-bench.
#
# Usage:
#   ./run-duckdb.sh --native ~/data/jsonbench/bluesky.db                  # all, 1 run
#   ./run-duckdb.sh --native ~/data/jsonbench/bluesky.db --query 3        # just q03
#   ./run-duckdb.sh --native ~/data/jsonbench/bluesky.db --iterations 3   # 3 timed runs
#   ./run-duckdb.sh --native ~/data/jsonbench/bluesky.db --write-expected # write qNN.tsv
#   ./run-duckdb.sh --native ~/data/jsonbench/bluesky.db --duckdb-process single
#
# --native <db> is the only source: a persistent .db holding the `bluesky (j
# JSON)` table that prep-jsonbench-data.sh loaded from the ndjson with
# read_ndjson_objects — DuckDB's own load, as upstream JSONBench does it. Unlike
# the ClickBench driver there is no parquet mode: pointing DuckDB at the parquet
# pivot wrote would measure pivot's shredding choices through DuckDB's reader,
# which is a different (and much less interesting) question than "each engine
# loads this ndjson the way it wants to".
#
# We report DuckDB's own `.timer` "Run Time" — query execution only, excluding
# process spawn and DB-open.
#
# --duckdb-process per-iteration|single  (default per-iteration)
#   per-iteration: a FRESH duckdb process per timed iteration, which is how
#     JSONBench and ClickBench measure embedded DuckDB. Every run is engine-cold
#     (OS page cache warm, DuckDB buffer pool empty). ⚠️ Asymmetric against pivot,
#     whose iterations reuse one warm server.
#   single: all of a query's iterations in ONE process, so iterations 2+ reuse
#     DuckDB's warm buffer pool — symmetric with pivot's warm server, but not the
#     upstream method.
#
# The page cache is dropped once per query (Linux, via sudo), so iteration 1 is a
# cold read and later ones are hot — matching pivot-bench's cold/hot split.
#
# --write-expected runs each query once and writes its rows to qNN.tsv in
# pivot-bench's wire format, so the oracle is DuckDB's answer rather than pivot
# grading its own homework. The queries must have a total ORDER BY (the check is
# exact-string), which the qNN-duckdb.sql files do.

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

native_db=""
queries=""
iterations=1
drop_caches=1
write_expected=0
sleep_ms=0
duckdb_process="per-iteration"

usage() { sed -n '3,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --native)         native_db="$2"; shift 2 ;;
        --query)          queries="$2"; shift 2 ;;
        --iterations)     iterations="$2"; shift 2 ;;
        --sleep)          sleep_ms="$2"; shift 2 ;;
        --duckdb-process) duckdb_process="$2"; shift 2 ;;
        --no-drop-caches) drop_caches=0; shift ;;
        --write-expected) write_expected=1; shift ;;
        -h|--help)        usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$duckdb_process" in
    per-iteration|single) ;;
    *) echo "error: --duckdb-process must be per-iteration|single (got '$duckdb_process')" >&2; usage 1 ;;
esac

[[ -n "$native_db" ]] || { echo "error: --native <db> is required" >&2; usage 1; }
[[ -f "$native_db" ]] || { echo "error: native db not found: $native_db" >&2; exit 1; }
command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not found on PATH" >&2; exit 1; }

flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

# Build the list of queries to run. Each qNN has a -duckdb.sql: the same question
# pivot's qNN.sql asks, in DuckDB's JSON dialect. (The verbatim upstream text is
# kept in duckdb-official/queries.sql for provenance — see PROVENANCE.md for the
# two places these differ from it and why.)
declare -a stems=()
if [[ -n "$queries" ]]; then
    IFS=',' read -ra ids <<< "$queries"
    for id in "${ids[@]}"; do
        id="${id#q}"
        printf -v stem 'q%02d' "$((10#$id))"
        [[ -f "$suite_dir/$stem-duckdb.sql" ]] || { echo "error: no $stem-duckdb.sql" >&2; exit 1; }
        stems+=("$stem")
    done
else
    for f in "$suite_dir"/q*-duckdb.sql; do
        stems+=("$(basename "$f" -duckdb.sql)")
    done
fi

echo "duckdb $(duckdb --version)"
echo "source: native db $native_db (bluesky table)"
if [[ "$write_expected" == "1" ]]; then
    echo "queries: ${#stems[@]}, writing expected .tsv (no timing)"
else
    echo "queries: ${#stems[@]}, iterations: $iterations, process: $duckdb_process"
fi
echo

iter_cmd=(duckdb "$native_db" -c ".timer on")

for stem in "${stems[@]}"; do
    sql="$(cat "$suite_dir/$stem-duckdb.sql")"
    echo "=== $stem ==="

    if [[ "$write_expected" == "1" ]]; then
        # pivot-bench's wire format: tab-separated, no header, NULL as empty.
        out_file="$suite_dir/$stem.tsv"
        printf '.headers off\n.nullvalue '\'''\''\n.mode tabs\n.output %s\n%s\n' \
            "$out_file" "$sql" | duckdb "$native_db"
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
