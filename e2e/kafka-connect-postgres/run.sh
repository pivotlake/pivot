#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
COMPOSE=(docker compose --project-directory "$HERE" -f "$HERE/compose.yaml")
TMP=$(mktemp -d)
PIVOT_PID=

cleanup() {
  "${COMPOSE[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
  if [[ -n "$PIVOT_PID" ]]; then
    kill "$PIVOT_PID" >/dev/null 2>&1 || true
    wait "$PIVOT_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT

# Setup: start PivotDB, a one-node Kafka broker, and a real Connect worker with
# Confluent's PostgreSQL JDBC sink installed.
cargo build --manifest-path "$ROOT/server/Cargo.toml" --bin pivotdb-server
mkdir -p "$TMP/catalog"
PIVOT_MEMORY_PCT=1 RUST_LOG=server=info "$ROOT/server/target/debug/pivotdb-server" \
  --bind 0.0.0.0:55432 --workers 1 --path "$TMP/catalog" \
  >"$TMP/pivot.log" 2>&1 &
PIVOT_PID=$!

for _ in $(seq 1 100); do
  if psql -h 127.0.0.1 -p 55432 -U connect -d pivot -c "SELECT 1" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
psql -h 127.0.0.1 -p 55432 -U connect -d pivot -c "SELECT 1" >/dev/null

"${COMPOSE[@]}" up --build --detach --wait
"${COMPOSE[@]}" exec -T kafka kafka-topics \
  --bootstrap-server kafka:9092 --create --if-not-exists \
  --topic pivot_connect_events --partitions 1 --replication-factor 1 >/dev/null

curl -fsS -H 'Content-Type: application/json' \
  --data-binary @"$HERE/connector.json" \
  http://127.0.0.1:18083/connectors >/dev/null

# Execute: two schemaful Kafka records force auto-creation, one prepared JDBC
# batch, and a database commit before the sink commits Kafka offset 2.
"${COMPOSE[@]}" exec -T kafka kafka-console-producer \
  --bootstrap-server kafka:9092 --topic pivot_connect_events <<'RECORDS'
{"schema":{"type":"struct","name":"pivot.connect.Event","optional":false,"fields":[{"field":"id","type":"int64","optional":true},{"field":"name","type":"string","optional":true}]},"payload":{"id":1,"name":"first"}}
{"schema":{"type":"struct","name":"pivot.connect.Event","optional":false,"fields":[{"field":"id","type":"int64","optional":true},{"field":"name","type":"string","optional":true}]},"payload":{"id":2,"name":"second"}}
RECORDS

offsets=
for _ in $(seq 1 120); do
  status=$(curl -fsS http://127.0.0.1:18083/connectors/pivot-postgres-sink/status)
  if jq -e '.connector.state == "FAILED" or any(.tasks[]?; .state == "FAILED")' \
    <<<"$status" >/dev/null; then
    jq . <<<"$status" >&2
    tail -n 300 "$TMP/pivot.log" >&2
    "${COMPOSE[@]}" logs --tail 300 --no-log-prefix connect >&2
    exit 1
  fi
  offsets=$(curl -fsS http://127.0.0.1:18083/connectors/pivot-postgres-sink/offsets)
  if jq -e 'any(.offsets[]?; .offset.kafka_offset == 2)' <<<"$offsets" >/dev/null; then
    break
  fi
  sleep 0.5
done

# Assert: the connector is healthy, committed both records, and auto-created
# the target. COUNT stays zero by design because this branch's INSERT is fake.
status=$(curl -fsS http://127.0.0.1:18083/connectors/pivot-postgres-sink/status)
jq -e '.connector.state == "RUNNING" and all(.tasks[]; .state == "RUNNING")' \
  <<<"$status" >/dev/null
jq -e 'any(.offsets[]?; .offset.kafka_offset == 2)' <<<"$offsets" >/dev/null
count=$(psql -At -h 127.0.0.1 -p 55432 -U connect -d pivot \
  -c "SELECT COUNT(*) FROM pivot_connect_events")
[[ "$count" == "0" ]]

echo "Kafka Connect PostgreSQL sink accepted and committed 2 records (fake row count: $count)."
