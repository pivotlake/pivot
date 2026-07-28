#!/usr/bin/env bash
#
# bench-modes.sh — run the pivot-vs-DuckDB comparison (benchmark.sh) across the
# three storage layouts prep-modes-data.sh builds, printing one table per mode:
#
#   single        both engines read one merged hits.parquet
#   partitioned   both engines glob a directory of several hits_<N>.parquet
#   native        pivot reads parquet; DuckDB reads its native .db `hits` table
#                 (ClickBench-native style) — pivot-on-parquet vs DuckDB-native
#
# Run prep-modes-data.sh first to create <root>/{single,partitioned,hits.db}.
# Like benchmark.sh, this measures a prebuilt binary rather than building one:
# pass --binary <pivot-bench>, built by `just setup-bench` / `just bench-build`
# (see the bench skills). It is forwarded to benchmark.sh with everything else.
#
# Usage:
#   ./bench-modes.sh --root ~/bench-data --query "$IDS" --iterations 3 --skip-check
#   ./bench-modes.sh --modes single,native --root ~/bench-data --query 32
#   ./bench-modes.sh --native-pivot-source partitioned ...   # pivot reads partitioned in native mode
#
# Any flag it doesn't recognise is forwarded verbatim to benchmark.sh
# (e.g. --restart-server, --no-drop-caches, --sleep, --iterations,
# --duckdb-process per-iteration|single).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

root="$HOME/bench-data"
modes="single,partitioned,native"
native_pivot_source="single"   # what pivot reads in native mode: single|partitioned
declare -a passthrough=()

usage() { sed -n '3,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)                root="$2"; shift 2 ;;
        --modes)               modes="$2"; shift 2 ;;
        --native-pivot-source) native_pivot_source="$2"; shift 2 ;;
        -h|--help)             usage 0 ;;
        *) passthrough+=("$1"); shift ;;   # forward everything else to benchmark.sh
    esac
done

part_dir="$root/partitioned"
single_dir="$root/single"
native_db="$root/hits.db"

# pivot's --source in native mode (single is simplest; partitioned is an option
# since pivot can be faster there).
case "$native_pivot_source" in
    single)      native_pivot_dir="$single_dir" ;;
    partitioned) native_pivot_dir="$part_dir" ;;
    *) echo "error: --native-pivot-source must be single|partitioned" >&2; exit 1 ;;
esac

run_mode() {
    local mode="$1"; shift
    echo
    echo "============================================================"
    echo "  MODE: $mode"
    echo "============================================================"
    case "$mode" in
        single)
            [[ -e "$single_dir/hits.parquet" ]] || { echo "  missing $single_dir/hits.parquet — run prep-modes-data.sh" >&2; return 1; }
            "$here/benchmark.sh" --source "$single_dir" --duckdb "${passthrough[@]}" ;;
        partitioned)
            [[ -d "$part_dir" ]] || { echo "  missing $part_dir — run prep-modes-data.sh" >&2; return 1; }
            "$here/benchmark.sh" --source "$part_dir" --duckdb "${passthrough[@]}" ;;
        native)
            [[ -f "$native_db" ]] || { echo "  missing $native_db — run prep-modes-data.sh" >&2; return 1; }
            "$here/benchmark.sh" --source "$native_pivot_dir" --native "$native_db" "${passthrough[@]}" ;;
        *) echo "error: unknown mode '$mode' (want single|partitioned|native)" >&2; return 1 ;;
    esac
}

IFS=',' read -ra mode_list <<< "$modes"
for m in "${mode_list[@]}"; do
    run_mode "$m"
done
