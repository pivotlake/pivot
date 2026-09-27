#!/usr/bin/env bash
#
# run-datafusion.sh — run the TPC-H suite through Apache DataFusion
# (datafusion-cli) for a side-by-side comparison with pivot-bench, over the
# SAME normalized parquet directories pivot reads (see setup.sql /
# prep-tpch-data.sh). Each base table is exposed as an external parquet table,
# read in place, so the unmodified official qNN.sql files run as-is.
#
# Usage:
#   ./run-datafusion.sh --source ~/tpch-sf100                    # all queries, 1 run
#   ./run-datafusion.sh --source ~/tpch-sf100 --query 12         # just q12
#   ./run-datafusion.sh --source ~/tpch-sf100 --iterations 3     # 3 timed runs each
#   ./run-datafusion.sh --source ~/tpch-sf100 --no-drop-caches   # skip the cache drop
#   ./run-datafusion.sh --source ~/tpch-sf100 --sleep 500        # quiet gap between queries
#   ./run-datafusion.sh --source ~/tpch-sf100 --datafusion ~/bin/datafusion-cli
#
# We report datafusion-cli's own "Elapsed" time per statement (planning and
# execution, not process start-up). --datafusion names the binary (default:
# `datafusion-cli` on PATH; `cargo install datafusion-cli` builds it).
#
# --datafusion-process per-query|per-iteration  (default per-query)
#   per-query: a fresh datafusion-cli per query, all of that query's
#     iterations inside it, so iteration 1 is engine-cold and later ones reuse
#     what the process cached (parquet footers, mostly) — the shape
#     run-clickhouse.sh uses and run-duckdb.sh calls per-query. --sleep cannot
#     separate iterations that share a process, so it only gaps the queries.
#   per-iteration: a FRESH datafusion-cli per timed iteration (engine-cold
#     every run, page cache warm after iteration 1).
#
# The OS page cache is dropped once before each query (needs root, via sudo),
# so iteration 1 is a true cold read; --no-drop-caches skips that.
#
# --sleep MS leaves a quiet gap before each query but the first (and, in
# per-iteration mode, between a query's iterations), matching pivot-bench's
# --sleep.
#
# --timeout SEC kills a datafusion-cli session that runs longer than SEC; its
# query reports an error instead of a time.
#
# --memory-limit SIZE (e.g. 300g) bounds DataFusion's memory pool, shared
# fairly between a plan's operators so the ones that can spill do;
# --disk-limit SIZE bounds what they may spill (DataFusion's default is 100g).
# Spill files go to $TMPDIR. Without --memory-limit the pool is unbounded.
#
# --suite-dir overrides where qNN.sql live (default: this script's own
# directory), so a shipped copy of this script can run a checkout's suite.
#
# Output mirrors run-duckdb.sh: a "=== qNN ===" header per query then one
# "Run Time (s): real X" line per iteration, so the same consumers parse both.

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source_path=""
queries=""
iterations=1
drop_caches=1
sleep_ms=0
datafusion_bin=""
datafusion_process="per-query"
timeout_sec=0
memory_limit=""
disk_limit=""

usage() {
    awk 'NR >= 3 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --suite-dir)  suite_dir="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --datafusion) datafusion_bin="$2"; shift 2 ;;
        --datafusion-process) datafusion_process="$2"; shift 2 ;;
        --timeout)      timeout_sec="$2"; shift 2 ;;
        --memory-limit) memory_limit="$2"; shift 2 ;;
        --disk-limit)   disk_limit="$2"; shift 2 ;;
        --no-drop-caches) drop_caches=0; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$datafusion_process" in
    per-query|per-iteration) ;;
    *) echo "error: --datafusion-process must be per-query|per-iteration (got '$datafusion_process')" >&2; usage 1 ;;
esac

[[ -n "$source_path" ]] || { echo "error: --source is required" >&2; usage 1; }
[[ -d "$source_path" ]] || { echo "error: --source must be the dataset root directory" >&2; exit 1; }
# DataFusion resolves a relative LOCATION against its own cwd, so hand it an
# absolute one.
source_root="$(cd "$source_path" && pwd)"

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

session_cmd=("$datafusion_bin")
[[ -n "$memory_limit" ]] && session_cmd+=(--memory-limit "$memory_limit" --mem-pool-type fair)
[[ -n "$disk_limit" ]] && session_cmd+=(--disk-limit "$disk_limit")
[[ "$timeout_sec" -gt 0 ]] && session_cmd=(timeout --kill-after=30 "$timeout_sec" "${session_cmd[@]}")

flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

# One external parquet table per base table over its directory. The trailing
# slash is how DataFusion is told to list the directory rather than open it
# as one file.
tables=(lineitem orders customer part partsupp supplier nation region)
setup=""
for t in "${tables[@]}"; do
    setup+="CREATE EXTERNAL TABLE $t STORED AS PARQUET LOCATION '$source_root/$t/';"$'\n'
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
        [[ "$f" == *-duckdb.sql || "$f" == *-clickhouse.sql ]] && continue
        query_files+=("$f")
    done
fi

echo "datafusion-cli $("$datafusion_bin" --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -1)"
echo "source: $source_root"
echo "queries: ${#query_files[@]}, iterations: $iterations, process: $datafusion_process"
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
        printf '%s' "$setup"
        printf "SELECT '=== %s ===' AS marker;\n" "$stem"
        for ((r = 1; r <= repeats; r++)); do printf '%s\n' "$sql"; done
    } >"$script"
    out="$("${session_cmd[@]}" -f "$script" 2>&1)" || true
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

queries_timed=0
for f in "${query_files[@]}"; do
    stem="$(basename "$f" .sql)"
    sql="$(cat "$f")"
    echo "=== $stem ==="

    if [[ "$queries_timed" -gt 0 && "${sleep_ms:-0}" -gt 0 ]]; then
        sleep "$(awk "BEGIN{print $sleep_ms/1000}")"
    fi
    queries_timed=$((queries_timed + 1))

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
