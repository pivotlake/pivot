#!/bin/bash
# Builds the input files of two more compaction shapes from table rows_1
# (28,000,000 rows sorted over one hour, 60 values of the first sort key):
#   adopt_disjoint/logs: the five files of rows_1, whose key ranges do not overlap.
#   adopt_slight/logs:   five files of 13 minutes each, each sharing one minute
#                        with the next.
set -uo pipefail
cd ~/merge-test
SQL="psql -h 127.0.0.1 -U pivot -d lake -At"
pid=$(ps -eo pid,args | grep "pivot server --config" | grep -v grep | cut -c1-8)
[ -n "$pid" ] && kill $pid && sleep 3
(COMPACT_BYTES=2g nohup ./run_server.sh > server_prepare.log 2>&1 &)
for attempt in $(seq 60); do $SQL -c "SELECT 1" >/dev/null 2>&1 && break; sleep 1; done
for part in 0 1 2 3 4; do
  first=$((part * 12))
  last=$((first + 13))
  upper="TIMESTAMP '2026-07-12 23:$(printf %02d $last):00'"
  [ $last -ge 60 ] && upper="TIMESTAMP '2026-07-13 00:00:00'"
  ./create_table.sh slight_$part > /dev/null
  $SQL -c "INSERT INTO slight_$part SELECT * FROM rows_1 WHERE round_timestamp >= TIMESTAMP '2026-07-12 23:$(printf %02d $first):00' AND round_timestamp < $upper"
  $SQL -c "COMPACT slight_$part FINAL"
  $SQL -c "SELECT '$part', count(*), min(round_timestamp), max(round_timestamp) FROM slight_$part"
  $SQL -c "SELECT path, bytes FROM system.table_files WHERE table_id = (SELECT id FROM system.tables WHERE name = 'slight_$part')"
done
python3 - <<'EOF'
import subprocess
import boto3

s3 = boto3.client("s3", region_name="eu-central-1")
bucket = "pivot-compaction-test-20261116"


def list_files(table):
    out = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-U", "pivot", "-d", "lake", "-At", "-c",
         f"SELECT path FROM system.table_files WHERE table_id = (SELECT id FROM system.tables WHERE name = '{table}')"],
        check=True, capture_output=True, text=True).stdout.split()
    return out


keys = {}
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="lake/"):
    for item in page.get("Contents", []):
        keys[item["Key"].rsplit("/", 1)[-1]] = item["Key"]


def copy(tables, prefix):
    number = 0
    for table in tables:
        for name in list_files(table):
            number += 1
            s3.copy({"Bucket": bucket, "Key": keys[name]}, bucket, f"{prefix}/input-{number}.parquet")
    print(prefix, number, "files")


copy(["rows_1"], "adopt_disjoint/logs")
copy([f"slight_{part}" for part in range(5)], "adopt_slight/logs")
EOF
pid=$(ps -eo pid,args | grep "pivot server --config" | grep -v grep | cut -c1-8)
kill $pid
echo PREPARED
