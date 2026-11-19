#!/bin/bash
# Runs the last statement the killed server logged, once, while sampling the
# server's resident memory twice a second. If it passes 50 GB, the client is
# killed, which cancels the statement.
cd ~/gilad-test/pivotdb
SQL=$(tr -d '\033' < nohup.out | grep -a "query received sql=SELECT t.datastore, t.schema, t.name AS table_name, c.name AS column_name" | tail -1 | sed 's/.*query received sql=//')
echo "statement: $SQL"
PID=$(ss -ltnp | grep ":5432" | grep -o "pid=[0-9]*" | cut -d= -f2)
START=$(date +%s.%N)
psql -h 127.0.0.1 -U pivot -d lake -At -c "$SQL" > /tmp/try_query.out 2>&1 &
CLIENT=$!
PEAK=0
while kill -0 $CLIENT 2>/dev/null; do
  RSS=$(ps -o rss= -p $PID | awk '{printf "%d", $1 / 1024}')
  [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  if [ "$RSS" -gt 51200 ]; then
    kill $CLIENT
    echo "guard: memory passed 50 GB ($RSS MB), client killed"
    break
  fi
  sleep 0.5
done
wait $CLIENT 2>/dev/null
echo "client exit $?, seconds $(echo "$(date +%s.%N) - $START" | bc), peak rss MB $PEAK"
echo "rows returned: $(wc -l < /tmp/try_query.out)"
head -3 /tmp/try_query.out | cut -c1-200
sleep 5
echo "rss MB after: $(ps -o rss= -p $PID | awk '{printf "%d", $1 / 1024}')"
