#!/usr/bin/env bash
#
# prep-jsonbench-data.sh — fetch JSONBench's Bluesky ndjson and load it for both
# engines, each the way that engine would really load it.
#
# Usage:
#   ./prep-jsonbench-data.sh --scale 1m                    # 1 file, ~1m events
#   ./prep-jsonbench-data.sh --scale 100m --dir ~/data     # 100 files
#   ./prep-jsonbench-data.sh --scale 10m --engines pivot   # skip the duckdb load
#
# --scale 1m|10m|100m|1000m   how many of the 1000 upstream files to take
# --dir <path>                where data lives (default ~/data/jsonbench)
# --engines pivot,duckdb      which loads to build (default both)
#
# Layout under --dir:
#   ndjson/file_NNNN.json.gz   the download, shared by both engines
#   pivot/*.parquet            pivot's load: --source for pivot-bench
#   bluesky.db                 duckdb's load: --native for run-duckdb.sh
#
# The two loads are deliberately not the same bytes. Upstream JSONBench gives
# DuckDB `create table bluesky (j JSON)` + `read_ndjson_objects`, so that is what
# it gets here, verbatim (duckdb-official/ddl.sql). Pivot has no INSERT yet, so
# `ndjson-to-parquet` streams the same ndjson through pivot's own write pipeline
# into Parquet, shredding whatever paths each file's rows agree on. Comparing the
# two therefore compares whole stacks — the write side's shredding choices
# included — which is the point.
#
# The download is ~125 MB per file (~425 MB raw), so 100m is ~12.5 GB down and
# 1000m is ~125 GB. The raw ndjson is never decompressed to disk: both loaders
# read the .gz directly.

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$suite_dir/../.." && pwd)"

scale="1m"
data_dir="$HOME/data/jsonbench"
engines="pivot,duckdb"

usage() { sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale)   scale="$2"; shift 2 ;;
        --dir)     data_dir="$2"; shift 2 ;;
        --engines) engines="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$scale" in
    1m)    files=1 ;;
    10m)   files=10 ;;
    100m)  files=100 ;;
    1000m) files=1000 ;;
    *) echo "error: --scale must be 1m|10m|100m|1000m (got '$scale')" >&2; exit 1 ;;
esac

want() { [[ ",$engines," == *",$1,"* ]]; }

ndjson_dir="$data_dir/ndjson"
pivot_dir="$data_dir/pivot"
duck_db="$data_dir/bluesky.db"
base_url="https://clickhouse-public-datasets.s3.amazonaws.com/bluesky"

mkdir -p "$ndjson_dir"

echo "=== downloading $files file(s) → $ndjson_dir"
for ((i = 1; i <= files; i++)); do
    printf -v name 'file_%04d.json.gz' "$i"
    # --continue resumes a partial file, --timestamping skips one already here,
    # so re-running after a bigger --scale only fetches what is missing.
    wget --continue --timestamping --progress=dot:giga \
        --directory-prefix "$ndjson_dir" "$base_url/$name" 2>&1 | grep -E 'saved|already' || true
done

if want pivot; then
    echo
    echo "=== pivot: ndjson → shredded parquet → $pivot_dir"
    rm -rf "$pivot_dir"
    loader="$repo_root/target/release/ndjson-to-parquet"
    [[ -x "$loader" ]] || {
        echo "building ndjson-to-parquet"
        cargo build --release --manifest-path "$repo_root/Cargo.toml" \
            -p benchmarks --bin ndjson-to-parquet
    }
    "$loader" --input "$ndjson_dir" --output "$pivot_dir"
    echo "pivot parquet: $(du -sh "$pivot_dir" | cut -f1)"
fi

if want duckdb; then
    echo
    echo "=== duckdb: ndjson → its own storage → $duck_db"
    command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not found on PATH" >&2; exit 1; }
    rm -f "$duck_db" "$duck_db.wal"
    # Upstream's load, verbatim: the official ddl, then read_ndjson_objects over
    # the .gz files. maximum_object_size matches upstream's 1 GB ceiling.
    duckdb "$duck_db" -c "$(cat "$suite_dir/duckdb-official/ddl.sql")"
    duckdb "$duck_db" -c "INSERT INTO bluesky SELECT * FROM read_ndjson_objects('$ndjson_dir/*.json.gz', ignore_errors=false, maximum_object_size=1048576000);"
    echo "duckdb rows: $(duckdb "$duck_db" -noheader -list -c 'SELECT count(*) FROM bluesky;')"
    echo "duckdb db:   $(du -sh "$duck_db" | cut -f1)"
fi

echo
echo "ready:"
want pivot  && echo "  pivot-bench   --suite jsonbench --source $pivot_dir"
want duckdb && echo "  ./run-duckdb.sh --native $duck_db"
