#!/usr/bin/env bash
#
# run-clickhouse.sh — run the ClickBench queries through ClickHouse-over-parquet
# for a side-by-side comparison with pivot (and DuckDB), the way the public
# ClickBench `clickhouse-parquet` / `clickhouse-parquet-partitioned` benchmark
# does it.
#
# Method (verbatim upstream): a FRESH `clickhouse local` process per query/
# iteration, with the typed `hits` schema prepended (backed by the parquet via
# `ENGINE = File(Parquet, …)`, read in place — no ingest), and `--time` printing
# the elapsed query seconds to stderr. The schema + ClickHouse-dialect queries
# are vendored under clickbench/clickhouse-official/. Like ClickBench, ClickHouse
# is measured engine-cold each run (fresh process; OS page cache warm after the
# first iteration).
#
# Output mirrors run-duckdb.sh: a "=== qNN ===" header per query then one
# "Run Time (s): real X" line per iteration, so benchmark.sh parses it uniformly.
#
# Usage:
#   ./run-clickhouse.sh --source ~/hits                    # parquet (clickhouse-local)
#   ./run-clickhouse.sh --source ~/hits_partitioned --query 32,33
#   ./run-clickhouse.sh --source ~/hits --iterations 3
#   ./run-clickhouse.sh --source ~/hits --clickhouse ~/clickhouse   # explicit binary
#   ./run-clickhouse.sh --native --query 32                # NATIVE MergeTree (server)
#
# --source must be a directory of *.parquet (globbed in place; single dir holds
# one hits.parquet, partitioned holds hits_*.parquet). --clickhouse points at the
# `clickhouse` binary (default: `clickhouse` on PATH, else ./clickhouse here).
#
# --native queries the official ClickBench ClickHouse NATIVE engine instead of
# parquet: clickhouse-client against the running server's MergeTree `hits` table
# (set up by prep-clickhouse-native.sh). Because the server is persistent, its
# iterations 2+ are engine-warm (like pivot), unlike the parquet path's fresh
# clickhouse-local per iteration. --source is ignored with --native.
#
# Like ClickBench, the OS page cache is dropped once before each query (Linux,
# via sudo), so iteration 1 is a cold read and later iterations are hot.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite_dir="$here"
official_dir="$suite_dir/clickhouse-official"

source_path=""
queries=""
iterations=1
drop_caches=1
sleep_ms=0
ch_bin=""
native=0

usage() { sed -n '3,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --clickhouse) ch_bin="$2"; shift 2 ;;
        --native)     native=1; shift ;;
        --no-drop-caches) drop_caches=0; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

# --native queries the running ClickHouse server's MergeTree table (set up by
# prep-clickhouse-native.sh) via clickhouse-client; --source is ignored. Otherwise
# we read parquet in place with clickhouse-local (--source required).
if [[ $native -eq 0 ]]; then
    [[ -n "$source_path" ]] || { echo "error: --source is required (or --native)" >&2; usage 1; }
    [[ -d "$source_path" ]] || { echo "error: --source must be a directory of *.parquet" >&2; exit 1; }
fi

# Resolve the clickhouse binary: explicit --clickhouse, else PATH, else ./clickhouse.
if [[ -z "$ch_bin" ]]; then
    if command -v clickhouse >/dev/null 2>&1; then ch_bin="clickhouse"
    elif [[ -x "$here/clickhouse" ]]; then ch_bin="$here/clickhouse"
    else echo "error: clickhouse not found (PATH or $here/clickhouse); pass --clickhouse" >&2; exit 1; fi
fi

# Parquet mode: build the schema for this source. The vendored create.sql is
# verbatim upstream (ENGINE = File(Parquet, 'hits.parquet')); we read the parquet
# in place from the source directory (cd'd into below), so swap the File() glob to
# '*.parquet' — resolving to the single hits.parquet or the partitioned hits_*.parquet.
schema=""
if [[ $native -eq 0 ]]; then
    [[ -f "$official_dir/create.sql" ]] || { echo "error: missing $official_dir/create.sql" >&2; exit 1; }
    schema="$(sed "s|File(Parquet, 'hits.parquet')|File(Parquet, '*.parquet')|" "$official_dir/create.sql")"
fi
official_queries="$official_dir/queries.sql"

# Build the list of query IDs (stems q07) to run.
declare -a ids=()
if [[ -n "$queries" ]]; then
    IFS=',' read -ra raw <<< "$queries"
    for id in "${raw[@]}"; do
        id="${id#q}"; printf -v stem 'q%02d' "$((10#$id))"; ids+=("$stem")
    done
else
    for f in "$suite_dir"/q*.sql; do
        [[ "$f" == *-duckdb.sql ]] && continue
        ids+=("$(basename "$f" .sql)")
    done
fi

flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    if ! { sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }; then
        echo "  warning: page-cache drop failed (need sudo/Linux); continuing" >&2
    fi
}

# Per-query cold preparation (runs once, before the iterations; iter 1 is the
# cold run, the rest hot).
#   parquet: just flush the OS page cache — a fresh clickhouse-local reads cold.
#   native:  match ClickBench's RESTARTABLE=yes cold cycle. A *running* server
#            keeps its data mmapped, so a flush alone can't evict those pages and
#            the next query would read warm. So STOP the server, wait until it is
#            really down, flush, then START it — the cold (iter 1) then pays a
#            true server cold-start (re-mmap marks, reload primary keys/parts).
cold_prep() {
    if [[ $native -eq 1 && "$drop_caches" == "1" ]]; then
        sudo clickhouse stop >/dev/null 2>&1 || true
        local k
        for k in $(seq 1 30); do "$ch_bin" client -q "SELECT 1" >/dev/null 2>&1 || break; sleep 0.5; done
        sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
        sudo clickhouse start >/dev/null 2>&1 || true
        for k in $(seq 1 60); do "$ch_bin" client -q "SELECT 1" >/dev/null 2>&1 && break; sleep 1; done
    else
        flush_page_cache
    fi
}

if [[ $native -eq 1 ]]; then
    echo "clickhouse $("$ch_bin" client --query "SELECT 'native ' || version()" 2>/dev/null || echo '? (is the server up? run prep-clickhouse-native.sh)')"
    echo "source: native MergeTree hits (clickhouse server)"
else
    echo "clickhouse $("$ch_bin" local --version 2>/dev/null | head -1 || echo '?')"
    echo "source: $source_path/*.parquet (ENGINE = File(Parquet, '*.parquet'))"
fi
echo "queries: ${#ids[@]}, iterations: $iterations"
echo

for stem in "${ids[@]}"; do
    num=$((10#${stem#q}))
    sql="$(sed -n "$((num + 1))p" "$official_queries")"
    [[ -n "$sql" ]] || { echo "error: no official query for $stem (line $((num + 1)))" >&2; exit 1; }
    echo "=== $stem ==="

    # Cold-prepare (parquet: flush; native: stop→flush→start), then run each
    # iteration. `--time` prints the elapsed query seconds (a bare number; per
    # statement, so for native the SELECT is the last/only one and for parquet we
    # take the last, dropping the CREATE-table time) — we reformat it to the
    # "Run Time (s): real X" line run-duckdb.sh emits. Result rows are discarded
    # via --format=Null (the --time value is the engine's query time, unaffected
    # by output format).
    #
    #   native:  clickhouse-client --time against the server's MergeTree. iter 1
    #            is a true cold-start (the cold_prep restarted the server); iters
    #            2+ are warm against the now-running server.
    #   parquet: a fresh `clickhouse local` per iteration over File(Parquet, ...),
    #            schema prepended, cwd the data dir — upstream's per-try model
    #            (engine-cold each run).
    cold_prep
    for ((i = 1; i <= iterations; i++)); do
        if [[ $native -eq 1 ]]; then
            err="$( "$ch_bin" client --time --format=Null --query="$sql" 2>&1 >/dev/null )" || true
        else
            err="$( ( cd "$source_path" && "$ch_bin" local --time --format=Null \
                --query="$schema
$sql" ) 2>&1 >/dev/null )" || true
        fi
        secs="$(printf '%s\n' "$err" | tr '\r' '\n' | grep -E '^[0-9]+(\.[0-9]+)?$' | tail -n1)"
        if [[ -n "$secs" ]]; then
            printf 'Run Time (s): real %s\n' "$secs"
        else
            # Surface the ClickHouse error so a failure is visible (and parses as ERR).
            printf 'Error: %s\n' "$(printf '%s' "$err" | tr '\n' ' ' | head -c 300)"
        fi
        if [[ "${sleep_ms:-0}" -gt 0 && $i -lt $iterations ]]; then
            sleep "$(awk "BEGIN{print $sleep_ms/1000}")"
        fi
    done
    echo
done
