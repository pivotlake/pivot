#!/usr/bin/env bash
#
# prep-modes-data.sh - fetch the datasets benchmark.sh / bench-modes.sh compare
# across, all straight from ClickBench so every layout holds the identical data:
#
#   <root>/single/hits.parquet
#                         the OFFICIAL single-file hits.parquet, downloaded
#                         verbatim (the "single" mode - a single-file scan)
#   <root>/partitioned/   the OFFICIAL hits_<N>.parquet files, downloaded
#                         verbatim (the "partitioned" mode - glob the dir)
#   <root>/hits.db        a persistent DuckDB database whose `hits` table is
#                         loaded from the partitioned files using the OFFICIAL
#                         ClickBench native schema (duckdb-official/create.sql:
#                         EventTime as TIMESTAMP, etc.) the way the public
#                         ClickBench leaderboard measures DuckDB's native engine
#
# single and partitioned are the same rows in two physical layouts - both pulled
# from ClickBench's published parquet - so differences in the numbers are
# differences in the read path, not the data. Nothing is derived from a
# DuckDB-merged parquet anymore; only the native .db is (unavoidably) DuckDB's
# own storage.
#
# Usage:
#   ./prep-modes-data.sh                         # full dataset → ~/bench-data
#   ./prep-modes-data.sh --root ~/bench-data
#   ./prep-modes-data.sh --files 0,11,18,21      # cheaper: a partitioned subset
#                                                # (single stays the full file)
#   ./prep-modes-data.sh --no-native             # skip the DuckDB .db build
#
# Idempotent: existing non-empty downloads and the built .db are reused; pass
# --force to rebuild the native .db from the (re)downloaded parquet.
#
# Needs curl or wget. `duckdb` is needed only for the native .db (skip with
# --no-native).

set -euo pipefail

root="$HOME/bench-data"
# Full partitioned set is hits_0..hits_99; override with --files for a subset.
files="$(seq -s, 0 99)"
single_url="https://datasets.clickhouse.com/hits_compatible/hits.parquet"
part_base_url="https://datasets.clickhouse.com/hits_compatible/athena_partitioned"
build_native=1
force=0

usage() { sed -n '3,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)          root="$2"; shift 2 ;;
        --files)         files="$2"; shift 2 ;;
        --single-url)    single_url="$2"; shift 2 ;;
        --part-base-url) part_base_url="$2"; shift 2 ;;
        --no-native)     build_native=0; shift ;;
        --force)         force=1; shift ;;
        -h|--help)       usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

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
single_file="$single_dir/hits.parquet"
native_db="$root/hits.db"
mkdir -p "$part_dir" "$single_dir"

echo "downloading ClickBench data → $root"

# 1. single/hits.parquet - the official single-file download, verbatim.
if [[ -s "$single_file" && $force -eq 0 ]]; then
    echo "  have  single/hits.parquet ($(du -h "$single_file" | cut -f1))"
else
    echo "  fetch single/hits.parquet (official, ~14GB)"
    fetch "$single_url" "$single_file"
fi

# 2. partitioned/hits_<N>.parquet - the official partitioned files, verbatim.
IFS=',' read -ra ids <<< "$files"
for n in "${ids[@]}"; do
    dest="$part_dir/hits_$n.parquet"
    if [[ -s "$dest" ]]; then
        echo "  have  hits_$n.parquet ($(du -h "$dest" | cut -f1))"
    else
        echo "  fetch hits_$n.parquet"
        fetch "$part_base_url/hits_$n.parquet" "$dest"
    fi
done

# 3. hits.db - load the partitioned files into DuckDB's native storage using the
#    OFFICIAL ClickBench native schema + load, so the comparison matches the
#    public leaderboard. `.read` the vendored create.sql (typed: EventTime
#    TIMESTAMP, EventDate DATE, …) then INSERT the parquet, converting the three
#    packed-seconds columns with epoch_ms(col*1000) and EventDate with make_date
#    - exactly upstream's duckdb/load. run-duckdb.sh --native then runs the
#    matching duckdb-official/queries.sql line. `-storage_version latest` matches
#    upstream.
if [[ $build_native -eq 0 ]]; then
    echo "  skip  hits.db (--no-native)"
else
    command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not on PATH (needed for the native .db; pass --no-native to skip)" >&2; exit 1; }
    official_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/duckdb-official"
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
fi

echo
echo "ready:"
echo "  single      : $single_dir       (--source)"
echo "  partitioned : $part_dir         (--source)"
[[ $build_native -eq 1 ]] && echo "  native      : $native_db        (--native)"
