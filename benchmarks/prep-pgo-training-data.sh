#!/usr/bin/env bash
#
# prep-pgo-training-data.sh - assemble the PGO training bundle and upload it
# to GCS for deploy.yml.
#
# Release binaries are PGO-optimized: deploy.yml instruments the server, runs
# every suite below against this bundle, and rebuilds the server from the
# profile (see build-pgo-server.sh). The bundle is one subdirectory per
# suite, each laid out the way that suite's --source expects:
#
#   hits/hits.parquet        clickbench + clickbench-insert: the full hits
#                            parquet (~14 GB), downloaded from the public
#                            dataset mirror
#   tpch-sf100/<table>/      tpch: SF100 parquet (~42 GB), synced from
#                            s3://epsio-tpch via prep-tpch-data.sh
#   tpch-flat-sf10/          tpch-flat: the SF10 denormalized table, built
#                            locally by prep-tpch-flat.sh (~10 GB)
#   jsonbench-10m/           jsonbench: 10 upstream ndjson files (~1.3 GB);
#                            the runner loads them by INSERT, so this suite
#                            trains the write path on real documents
#
# Run it on a box with the tooling (aws, tpchgen-cli, duckdb, wget, gcloud)
# and ~120 GB free under --root, then point the PGO_TRAINING_GCS repository
# variable at --dest. Deploy runners keep a checksum-synced copy on disk, so
# a re-upload is picked up on the next deploy with no other action.
#
# Idempotent per suite: a bundle directory that already exists locally is
# left alone (tpch-sf100 re-syncs, which is a no-op when complete); use
# --force to rebuild everything from scratch.
#
# Usage:
#   ./prep-pgo-training-data.sh --dest gs://my-bucket/pgo-training
#   ./prep-pgo-training-data.sh --root /big/disk/bundle --dest gs://...
#   ./prep-pgo-training-data.sh --root ~/pgo-training-data   # build only

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

root="$HOME/pgo-training-data"
dest=""
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)  root="$2"; shift 2 ;;
        --dest)  dest="$2"; shift 2 ;;
        --force) force=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ $force -eq 1 ]]; then
    rm -rf "$root"
fi
mkdir -p "$root"

echo "=== hits (clickbench + clickbench-insert)"
if [[ -f "$root/hits/hits.parquet" ]]; then
    echo "  have  $root/hits/hits.parquet"
else
    mkdir -p "$root/hits"
    wget --continue --progress=dot:giga \
        -O "$root/hits/hits.parquet.partial" \
        "https://datasets.clickhouse.com/hits_compatible/hits.parquet"
    mv "$root/hits/hits.parquet.partial" "$root/hits/hits.parquet"
fi

echo "=== tpch-sf100"
"$script_dir/tpch/prep-tpch-data.sh" --dataset sf100 --root "$root/tpch-sf100"

echo "=== tpch-flat-sf10"
if [[ -d "$root/tpch-flat-sf10" ]]; then
    echo "  have  $root/tpch-flat-sf10"
else
    work="$root/.work-tpch-flat"
    "$script_dir/tpch-flat/prep-tpch-flat.sh" --scale-factor 10 --root "$work"
    mv "$work/flat" "$root/tpch-flat-sf10"
    rm -rf "$work"
fi

echo "=== jsonbench-10m"
if [[ -d "$root/jsonbench-10m" ]]; then
    echo "  have  $root/jsonbench-10m"
else
    work="$root/.work-jsonbench"
    "$script_dir/jsonbench/prep-jsonbench-data.sh" --scale 10m --engines none --dir "$work"
    mv "$work/ndjson" "$root/jsonbench-10m"
    rm -rf "$work"
fi

echo
echo "=== bundle"
du -sh "$root"/*

if [[ -n "$dest" ]]; then
    dest="${dest%/}"
    echo
    echo "=== uploading to $dest"
    gcloud storage rsync --recursive --delete-unmatched-destination-objects \
        "$root" "$dest"
    echo "done; set the PGO_TRAINING_GCS repository variable to $dest"
else
    echo
    echo "no --dest given: bundle built but not uploaded"
fi
