#!/usr/bin/env bash
#
# run-datafusion.sh — run the ClickBench queries through Apache DataFusion
# (datafusion-cli) over parquet for a side-by-side comparison with pivot (and
# DuckDB, ClickHouse), the way the public ClickBench `datafusion` /
# `datafusion-partitioned` benchmark does it.
#
# Method (verbatim upstream): a FRESH `datafusion-cli` process per query/
# iteration, fed the vendored create.sql (an external parquet table plus the
# `hits` view that types EventDate, read in place — no ingest) and then the
# query from the vendored DataFusion-dialect queries.sql. The CLI prints
# "Elapsed X seconds." after every statement; the query's own line is its time.
# Schema + queries live under clickbench/datafusion-official/. Like ClickBench,
# DataFusion is measured engine-cold each run (fresh process; OS page cache warm
# after the first iteration).
#
# Output mirrors run-duckdb.sh: a "=== qNN ===" header per query then one
# "Run Time (s): real X" line per iteration, so benchmark.sh parses it uniformly.
#
# Usage:
#   ./run-datafusion.sh --source ~/hits                      # a directory of *.parquet
#   ./run-datafusion.sh --source ~/hits.parquet --query 32,33
#   ./run-datafusion.sh --source ~/hits --iterations 3
#   ./run-datafusion.sh --source ~/hits --datafusion ~/bin/datafusion-cli
#   ./run-datafusion.sh --source ~/hits --datafusion-process per-query
#
# --source is a directory (every *.parquet in it) or a single .parquet file; it
# becomes create.sql's LOCATION. --datafusion names the datafusion-cli binary
# (default: `datafusion-cli` on PATH; `cargo install datafusion-cli` builds it).
#
# --datafusion-process per-iteration|per-query  (default per-iteration)
#   per-iteration: a fresh datafusion-cli per timed iteration — how ClickBench
#     measures DataFusion, so every run, hot ones included, is engine-cold.
#   per-query: all of a query's iterations in ONE datafusion-cli, so iterations
#     2+ reuse what the process cached (parquet footers, mostly): the engine-warm
#     shape run-duckdb.sh calls `single`. Not the ClickBench method. --sleep
#     cannot separate iterations that share a process, so it is ignored here.
#
# Like ClickBench, the OS page cache is dropped once before each query (Linux,
# via sudo), so iteration 1 is a cold read and later iterations are hot;
# --no-drop-caches skips that. --sleep MS pauses between a query's iterations.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite_dir="$here"
official_dir="$suite_dir/datafusion-official"

source_path=""
queries=""
iterations=1
drop_caches=1
sleep_ms=0
datafusion_bin=""
datafusion_process="per-iteration"

usage() {
    awk 'NR >= 3 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --datafusion) datafusion_bin="$2"; shift 2 ;;
        --datafusion-process) datafusion_process="$2"; shift 2 ;;
        --no-drop-caches) drop_caches=0; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$datafusion_process" in
    per-iteration|per-query) ;;
    *) echo "error: --datafusion-process must be per-iteration|per-query (got '$datafusion_process')" >&2; usage 1 ;;
esac

[[ -n "$source_path" ]] || { echo "error: --source is required" >&2; usage 1; }
# DataFusion resolves a relative LOCATION against its own cwd, so hand it an
# absolute one. A directory keeps its trailing slash: that is how an external
# table is told to list the directory rather than open it as one file.
if [[ -d "$source_path" ]]; then
    location="$(cd "$source_path" && pwd)/"
elif [[ -f "$source_path" ]]; then
    location="$(cd "$(dirname "$source_path")" && pwd)/$(basename "$source_path")"
else
    echo "error: --source must be a directory of *.parquet or a .parquet file" >&2
    exit 1
fi

if [[ -z "$datafusion_bin" ]]; then
    command -v datafusion-cli >/dev/null 2>&1 || {
        echo "error: datafusion-cli not found on PATH (cargo install datafusion-cli), or pass --datafusion <binary>" >&2
        exit 1
    }
    datafusion_bin="datafusion-cli"
elif [[ ! -x "$datafusion_bin" ]]; then
    echo "error: --datafusion $datafusion_bin is not an executable file" >&2
    exit 1
fi

# Flush the Linux page cache so the next query reads from disk, the way
# ClickBench measures a cold run. Needs root, so it goes through sudo.
flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

# Build the list of query stems to run (qNN), accepting "7" or "q07".
declare -a stems=()
if [[ -n "$queries" ]]; then
    IFS=',' read -ra ids <<< "$queries"
    for id in "${ids[@]}"; do
        id="${id#q}"
        printf -v stem 'q%02d' "$((10#$id))"
        [[ -f "$suite_dir/$stem.sql" ]] || { echo "error: no query $stem in $suite_dir" >&2; exit 1; }
        stems+=("$stem")
    done
else
    for f in "$suite_dir"/q*.sql; do
        [[ "$f" == *-duckdb.sql ]] && continue   # overrides, not standalone queries
        stems+=("$(basename "$f" .sql)")
    done
fi

# The vendored schema with its LOCATION pointed at --source. Upstream's single
# and partitioned variants differ only in that line, so one copy serves both.
setup="$(sed "s|^LOCATION '.*'|LOCATION '$location'|" "$official_dir/create.sql")"
official_queries="$official_dir/queries.sql"

echo "datafusion-cli $("$datafusion_bin" --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -1)"
echo "source: $location"
echo "queries: ${#stems[@]}, iterations: $iterations, process: $datafusion_process"
echo

# Run the setup, a marker SELECT and then the query $2 times in one
# datafusion-cli, and print one "Run Time (s): real X" line per iteration.
#
# The CLI prints "Elapsed X seconds." after every statement, the setup's
# included, so the marker's result row (a "| === qNN === |" table line) is
# what tells the query's timings apart: the first Elapsed after it is the
# marker's own, every later one is an iteration. Fewer than $2 of them means
# a statement failed; the CLI's error lines are surfaced in their place.
run_session() {
    local stem="$1" repeats="$2" sql="$3" script out
    script="$(mktemp)"
    {
        printf '%s\n' "$setup"
        printf "SELECT '=== %s ===' AS marker;\n" "$stem"
        for ((r = 1; r <= repeats; r++)); do printf '%s\n' "$sql"; done
    } >"$script"
    out="$("$datafusion_bin" -f "$script" 2>&1)" || true
    rm -f "$script"
    local -a times=()
    mapfile -t times < <(awk '
        /^\| === q[0-9]+ === +\|$/ { marked = 1; skip_marker = 1; next }
        /^Elapsed / && marked { if (skip_marker) { skip_marker = 0; next } print $2 }
    ' <<<"$out")
    local t
    for t in "${times[@]}"; do echo "Run Time (s): real $t"; done
    if (( ${#times[@]} < repeats )); then
        echo "Error:"
        grep -iE 'error|panic' <<<"$out" | sed 's/^/  /' || tail -5 <<<"$out" | sed 's/^/  /'
    fi
}

for stem in "${stems[@]}"; do
    # The official query verbatim: queries.sql line N+1 is qNN (q00 → line 1).
    num=$((10#${stem#q}))
    sql="$(sed -n "$((num + 1))p" "$official_queries")"
    [[ -n "$sql" ]] || { echo "error: no official query for $stem (line $((num + 1)) of $official_queries)" >&2; exit 1; }
    echo "=== $stem ==="

    flush_page_cache
    if [[ "$datafusion_process" == "per-query" ]]; then
        run_session "$stem" "$iterations" "$sql"
    else
        for ((i = 1; i <= iterations; i++)); do
            run_session "$stem" 1 "$sql"
            if [[ "${sleep_ms:-0}" -gt 0 && $i -lt $iterations ]]; then
                sleep "$(awk "BEGIN{print $sleep_ms/1000}")"
            fi
        done
    fi
    echo
done
