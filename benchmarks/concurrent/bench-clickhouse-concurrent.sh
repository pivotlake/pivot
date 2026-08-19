#!/usr/bin/env bash
#
# bench-clickhouse-concurrent.sh — concurrency benchmark against native
# ClickHouse (MergeTree), mirroring pivot-bench's --clients mode so the two
# engines' numbers are comparable.
#
# Steps:
#   1. Load the MergeTree `hits` table from --source via
#      ../clickbench/prep-clickhouse-native.sh (skipped with --skip-prep when
#      the table is already loaded).
#   2. OPTIMIZE TABLE hits FINAL, so parts are fully merged before measuring.
#   3. For each requested client count: run concurrent-clickhouse.py with a
#      warmup sweep and shuffled per-client orders (the same deterministic
#      permutations the pivot harness uses).
#
# Usage:
#   ./bench-clickhouse-concurrent.sh --source ~/hits-quarter
#   ./bench-clickhouse-concurrent.sh --source ~/hits-quarter --clients "1 3 6" \
#       --out-dir ~/conc-results --order shuffled
#   ./bench-clickhouse-concurrent.sh --skip-prep --clients 6

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source_path=""
client_counts="1 3 6"
order="shuffled"
out_dir="$here/ch-results"
skip_prep=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)    source_path="$2"; shift 2 ;;
        --clients)   client_counts="$2"; shift 2 ;;
        --order)     order="$2"; shift 2 ;;
        --out-dir)   out_dir="$2"; shift 2 ;;
        --skip-prep) skip_prep=1; shift ;;
        -h|--help)   sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ $skip_prep -eq 0 ]]; then
    [[ -n "$source_path" ]] || { echo "error: --source is required (or --skip-prep)" >&2; exit 1; }
    "$here/../clickbench/prep-clickhouse-native.sh" --source "$source_path"
fi

echo "== OPTIMIZE TABLE hits FINAL =="
clickhouse-client --query "OPTIMIZE TABLE hits FINAL"
clickhouse-client --query "SELECT count() AS rows, count(DISTINCT _part) AS parts FROM hits" \
    --format PrettyCompact

mkdir -p "$out_dir"
for n in $client_counts; do
    python3 "$here/concurrent-clickhouse.py" --clients "$n" --order "$order" \
        --warmup-sweep --json-out "$out_dir/clickhouse-$order-$n.json"
done
