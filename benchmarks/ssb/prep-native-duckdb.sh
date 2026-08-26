#!/usr/bin/env bash
#
# prep-native-duckdb.sh - load the SSB parquet dataset (prep-ssb-data.sh) into
# a native DuckDB database file, for run-duckdb.sh --data native. Native
# storage gives DuckDB its own compression and full table statistics, so this
# is its best-case side of the comparison; the parquet mode is the
# same-files-as-pivot side.
#
# Usage:
#   ./prep-native-duckdb.sh --source ~/ssb-sf100 --out /mnt/nvme/ssb-native.duckdb

set -euo pipefail

source_path=""
out=""

usage() { sed -n '3,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) source_path="$2"; shift 2 ;;
        --out)    out="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ -d "$source_path" ]] || { echo "error: --source must be the dataset root directory" >&2; usage 1; }
[[ -n "$out" ]] || { echo "error: --out is required" >&2; usage 1; }
command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not on PATH" >&2; exit 1; }
[[ -e "$out" ]] && { echo "error: $out already exists; remove it to rebuild" >&2; exit 1; }

for t in lineorder customer supplier part date; do
    echo "  load $t"
    duckdb "$out" -c "CREATE TABLE $t AS SELECT * FROM read_parquet('${source_path%/}/$t/*.parquet');"
done
duckdb "$out" -c "CHECKPOINT;"
echo "ready: --data native --source $out"
