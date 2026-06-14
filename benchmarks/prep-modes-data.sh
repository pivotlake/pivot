#!/usr/bin/env bash
#
# prep-modes-data.sh — build the three datasets benchmark.sh compares across,
# from a small subset of ClickBench's partitioned parquet files so a run is
# cheap (a few GB, not the full 14 GB hits.parquet):
#
#   <root>/partitioned/   the downloaded hits_<N>.parquet files, left separate
#                         (the "partitioned" mode — pivot/DuckDB glob the dir)
#   <root>/single/hits.parquet
#                         all subset files merged into ONE parquet file
#                         (the "single" mode — a single-file scan)
#   <root>/hits.db        a persistent DuckDB database whose `hits` table is
#                         loaded from the subset using the OFFICIAL ClickBench
#                         native schema (clickbench/duckdb-official/create.sql:
#                         EventTime as TIMESTAMP, etc.) the way the public
#                         ClickBench leaderboard measures DuckDB's native engine
#
# The subset is the same data in all three layouts, so differences in the
# numbers are differences in the read path, not the data.
#
# Usage:
#   ./prep-modes-data.sh                         # files 0,11,18,21 → ~/bench-data
#   ./prep-modes-data.sh --root ~/bench-data --files 0,11,18,21
#   ./prep-modes-data.sh --files 0,5,10          # a different subset
#
# Idempotent: existing downloads and built artifacts are reused; pass --force to
# rebuild single/native from the (re)downloaded parquet.
#
# Needs `duckdb` on PATH (used for the merge and the native load) and curl/wget.

set -euo pipefail

root="$HOME/bench-data"
files="0,11,18,21"
base_url="https://datasets.clickhouse.com/hits_compatible/athena_partitioned"
force=0

usage() { sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)     root="$2"; shift 2 ;;
        --files)    files="$2"; shift 2 ;;
        --base-url) base_url="$2"; shift 2 ;;
        --force)    force=1; shift ;;
        -h|--help)  usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not on PATH" >&2; exit 1; }

# Pick whichever downloader is present.
fetch() {  # fetch <url> <dest>
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$2" "$1"
    else
        echo "error: need curl or wget" >&2; exit 1
    fi
}

part_dir="$root/partitioned"
single_dir="$root/single"
native_db="$root/hits.db"
mkdir -p "$part_dir" "$single_dir"

echo "subset files: $files  →  $root"

# 1. Download each subset file (skip ones already present and non-empty).
IFS=',' read -ra ids <<< "$files"
for n in "${ids[@]}"; do
    dest="$part_dir/hits_$n.parquet"
    if [[ -s "$dest" ]]; then
        echo "  have  hits_$n.parquet ($(du -h "$dest" | cut -f1))"
    else
        echo "  fetch hits_$n.parquet"
        fetch "$base_url/hits_$n.parquet" "$dest"
    fi
done

# 2. single/hits.parquet — merge the subset into one file, preserving the raw
#    parquet schema (no type rewrites) so pivot reads it exactly as it reads the
#    partitioned dir. A directory holding a single file IS the "single" source.
single_file="$single_dir/hits.parquet"
if [[ -s "$single_file" && $force -eq 0 ]]; then
    echo "  have  single/hits.parquet ($(du -h "$single_file" | cut -f1))"
else
    echo "  build single/hits.parquet (merge)"
    rm -f "$single_file"
    duckdb -c "COPY (SELECT * FROM read_parquet('$part_dir/*.parquet'))
               TO '$single_file' (FORMAT PARQUET);"
fi

# 3. hits.db — load the subset into DuckDB's native storage using the OFFICIAL
#    ClickBench native schema + load, so the comparison matches the public
#    leaderboard. We `.read` the vendored create.sql (typed: EventTime TIMESTAMP,
#    EventDate DATE, …) then INSERT the parquet, converting the three
#    packed-seconds columns with epoch_ms(col*1000) and EventDate with make_date
#    — exactly upstream's duckdb/load. run-duckdb.sh --native then runs the
#    matching duckdb-official/queries.sql line (which uses EventTime as a real
#    TIMESTAMP). `-storage_version latest` matches upstream.
official_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/clickbench/duckdb-official"
if [[ -s "$native_db" && $force -eq 0 ]]; then
    echo "  have  hits.db ($(du -h "$native_db" | cut -f1))"
else
    echo "  build hits.db (official native load: $official_dir/create.sql)"
    rm -f "$native_db"
    duckdb "$native_db" -storage_version latest <<SQL
.read $official_dir/create.sql
INSERT INTO hits
SELECT * REPLACE (
    make_date(EventDate) AS EventDate,
    epoch_ms(EventTime * 1000) AS EventTime,
    epoch_ms(ClientEventTime * 1000) AS ClientEventTime,
    epoch_ms(LocalEventTime * 1000) AS LocalEventTime)
FROM read_parquet('$part_dir/*.parquet', binary_as_string=True);
SQL
fi

rows=$(duckdb "$native_db" -noheader -list -c "SELECT count(*) FROM hits;")
echo
echo "ready ($rows rows):"
echo "  partitioned : $part_dir         (--source)"
echo "  single      : $single_dir       (--source)"
echo "  native      : $native_db        (--native)"
