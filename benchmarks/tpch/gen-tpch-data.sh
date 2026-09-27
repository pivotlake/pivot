#!/usr/bin/env bash
# gen-tpch-data.sh — generate a normalized TPC-H parquet dataset locally with
# tpchgen-cli, in the layout setup.sql expects (<root>/<table>/<table>.N.parquet).
# For scales too large to keep in S3: SF1000 is ~396 GB and generates in about
# 2.5 minutes on 192 cores, far faster than syncing it.
#
# The output matches the hosted datasets prep-tpch-data.sh syncs (the same
# tpchgen-cli version and defaults: SNAPPY, 7 MiB row groups).
#
# Usage:
#   ./gen-tpch-data.sh --scale 1000 --root /mnt/nvme/tpch/sf1000
#   ./gen-tpch-data.sh --scale 1000 --root /mnt/nvme/tpch/sf1000 --parts 10
#
# Needs tpchgen-cli 2.0.1 on PATH:
#   cargo install tpchgen-cli --version 2.0.1 --locked
# (--locked matters: the unlocked resolve pulls a tpchgen-arrow that fails to
# build against the arrow version the crate pins.)
#
# q11's HAVING fraction depends on the scale (0.0001 / SF); the suite's q11.sql
# is written for SF100, see the README.

set -euo pipefail

tpchgen_version="2.0.1"
scale=""
root=""
parts=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale) scale="$2"; shift 2 ;;
        --root)  root="$2"; shift 2 ;;
        --parts) parts="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$scale" ]] || { echo "error: --scale is required" >&2; exit 1; }
root="${root:-$HOME/tpch-sf$scale}"

tpchgen-cli --version 2>/dev/null | grep -qx "tpchgen $tpchgen_version" || {
    echo "error: tpchgen-cli $tpchgen_version not found on PATH" >&2
    exit 1
}

mkdir -p "$root"
start=$SECONDS
tpchgen-cli --scale-factor "$scale" --format parquet --parts "$parts" --output-dir "$root"
echo "generated SF$scale in $((SECONDS - start))s: $(du -sh "$root" | cut -f1) at $root"
