#!/bin/bash
# Unregisters the table otel_logs_unpartitioned from the datastore's catalog
# manifest without touching its files: stops the server, saves the manifest,
# removes the table's name and location from it (no tombstone, so vacuum never
# looks at the table), uploads it only if nobody changed it meanwhile, and
# starts the server again.
set -euo pipefail
cd ~/gilad-test/pivotdb
TABLE=otel_logs_unpartitioned
MANIFEST=gs://epsio-io-otel-parquet/iceberg-pivot/_pivot_manifest.json
TS=$(date -u +%Y%m%dT%H%M%S)
SAVED=manifest-before-unregister-$TS.json

PID=$(ss -ltnp | grep ":5432" | grep -o "pid=[0-9]*" | cut -d= -f2)
kill -TERM "$PID"
for i in $(seq 90); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -0 "$PID" 2>/dev/null && { echo "server did not stop"; exit 1; }
echo "server stopped"

GENERATION=$(gsutil ls -L $MANIFEST | grep "Generation:" | awk '{print $2}')
gsutil -q cp $MANIFEST $SAVED
cp $SAVED manifest-edited-$TS.json
python3 - "$SAVED" "manifest-edited-$TS.json" "$TABLE" "unregistered-$TABLE-$TS.json" <<'EOF'
import json
import sys

saved, edited, table, removed_path = sys.argv[1:5]
manifest = json.load(open(saved))
removed = {}
for schema in manifest["schemas"]:
    if table in schema["table_ids"]:
        table_id = schema["table_ids"].pop(table)
        removed = {
            "schema": schema["name"],
            "name": table,
            "id": table_id,
            "location": manifest["table_locations"].pop(table_id),
        }
assert removed, "the table is not in the manifest"
manifest["version"] += 1
json.dump(manifest, open(edited, "w"))
json.dump(removed, open(removed_path, "w"), indent=2)
print("removed:", removed)
print("edited manifest:", json.dumps(manifest))
EOF
gsutil -q -h "x-goog-if-generation-match:$GENERATION" -h "Content-Type:application/json" cp manifest-edited-$TS.json $MANIFEST
echo "uploaded; manifest now:"
gsutil cat $MANIFEST
echo
cat >> DEPLOYED-compaction-streaming-merge.txt <<EOF
$TS unregistered $TABLE by removing it from _pivot_manifest.json (its files are untouched).
  saved manifest: $SAVED; removed entries: unregistered-$TABLE-$TS.json
  to register it again: stop the server, put the name->id back under the schema's table_ids and the
  id->location back under table_locations in $MANIFEST, start the server.
EOF
echo "===== $(date -u +%FT%TZ) restarting c64f0934 after unregistering $TABLE =====" >> nohup.out
(nohup setsid ~/pivot-compaction-6ba4c334/target/release/pivot server --config conf-iceberg.yaml >> nohup.out 2>&1 &)
for i in $(seq 120); do psql -h 127.0.0.1 -U pivot -d lake -At -c "SELECT 1" >/dev/null 2>&1 && break; sleep 1; done
echo "server answering after ${i}s"
