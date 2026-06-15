#!/usr/bin/env bash
#
# prep-clickhouse-native.sh — set up the official ClickBench ClickHouse NATIVE
# (MergeTree) engine so `run-clickhouse.sh --native` (and benchmark.sh in native
# mode) can query it. Mirrors upstream clickhouse/{install,start,load} verbatim:
#
#   1. install clickhouse as a system service (`sudo clickhouse install`)
#   2. drop in the eager-load config (clickbench/clickhouse-official/eager_load.yaml)
#      so cold-query timing doesn't catch lazy part/primary-key loading
#   3. start the server (`sudo clickhouse start`)
#   4. create the MergeTree `hits` table (clickhouse-official/create-native.sql)
#   5. load it from a parquet source via `INSERT ... FROM file('*.parquet')`
#
# The server is left RUNNING (native ClickHouse is a persistent server, like
# pivot). Stop it with `sudo clickhouse stop` when done.
#
# Usage:
#   ./prep-clickhouse-native.sh                         # load from ~/hits_partitioned
#   ./prep-clickhouse-native.sh --source ~/hits_partitioned --binary ~/clickhouse
#
# Needs passwordless sudo (Linux). ~10 GB+ disk for the MergeTree parts.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
official_dir="$here/clickbench/clickhouse-official"

source_path="$HOME/hits_partitioned"
binary="$HOME/clickhouse"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) source_path="$2"; shift 2 ;;
        --binary) binary="$2"; shift 2 ;;
        -h|--help) sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -d "$source_path" ]] || { echo "error: --source dir not found: $source_path" >&2; exit 1; }
[[ -f "$official_dir/create-native.sql" ]] || { echo "error: missing $official_dir/create-native.sql" >&2; exit 1; }

# 1. Install clickhouse as a system service (idempotent) — upstream clickhouse/install.
if [[ ! -x /usr/bin/clickhouse ]]; then
    if [[ ! -x "$binary" ]]; then
        echo "== fetching clickhouse =="; ( cd "$HOME" && curl -s https://clickhouse.com/ | sh ); binary="$HOME/clickhouse"
    fi
    echo "== sudo clickhouse install =="
    sudo "$binary" install --noninteractive
fi

# 2. Eager-load config so the cold timer doesn't catch startup laziness.
echo "== eager-load config =="
sudo mkdir -p /etc/clickhouse-server/config.d
sudo cp "$official_dir/eager_load.yaml" /etc/clickhouse-server/config.d/eager_load.yaml

# 3. Start the server and wait until it answers. `clickhouse start` returns
#    non-zero when the server is already running, so tolerate that (the readiness
#    wait below is the real check).
echo "== clickhouse start (idempotent) =="
sudo clickhouse start 2>/dev/null || true
ready=0
for i in $(seq 1 60); do clickhouse-client --query "SELECT 1" >/dev/null 2>&1 && { ready=1; break; }; sleep 1; done
[[ $ready -eq 1 ]] || { echo "error: clickhouse server did not become ready" >&2; exit 1; }
clickhouse-client --query "SELECT 'clickhouse ' || version()"

# 4. Create the MergeTree table (CREATE OR REPLACE — idempotent).
echo "== create MergeTree hits =="
clickhouse-client < "$official_dir/create-native.sql"

# 5. Load from parquet — symlink the files into user_files (instant, vs copying
#    a 14 GB dataset) and INSERT via the file() table function, exactly upstream.
#    One ln per file (the multi-source form into a dir is fragile); ensure the
#    user_files dir exists first.
echo "== load from $source_path/*.parquet =="
sudo mkdir -p /var/lib/clickhouse/user_files
for f in "$source_path"/*.parquet; do
    sudo ln -sf "$f" /var/lib/clickhouse/user_files/"$(basename "$f")"
done
# The chown/rm globs must expand as root: user_files is clickhouse-owned (0700),
# so a glob run in this (ubuntu) shell would see nothing. Run them in a root shell.
sudo bash -c 'chown -h clickhouse:clickhouse /var/lib/clickhouse/user_files/*.parquet'
clickhouse-client --query "INSERT INTO hits SELECT * FROM file('*.parquet')" \
    --max-insert-threads "$(( $(nproc) / 4 ))"
sudo bash -c 'rm -f /var/lib/clickhouse/user_files/*.parquet'
sync

rows="$(clickhouse-client --query 'SELECT count() FROM hits')"
echo
echo "ready: clickhouse-native MergeTree hits = $rows rows (server running)"
echo "  run:  ./run-clickhouse.sh --native --query 32"
echo "  stop: sudo clickhouse stop"
