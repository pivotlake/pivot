#!/bin/bash
# usage: query_during_compaction.sh <run number> <table to query>
# Starts the server, times the latest-row query on <table> while idle, then
# adopts the five fully overlapping input files as a new table and compacts
# it while timing the same query in a loop. Writes q<N>_idle.txt and
# q<N>_busy.txt (one "time ms" line per run) and server<N>.log.
set -uo pipefail
RUN=$1
QUERY_TABLE=$2
cd ~/merge-test
SQL="select * FROM $QUERY_TABLE ORDER BY \"timestamp\" DESC LIMIT 1"
time_query() {
  local start=$(date +%s%N)
  psql -h 127.0.0.1 -U pivot -d lake -At -c "$SQL" > /dev/null 2>&1
  echo "$(date -u +%T.%N | cut -c1-12) $(( ($(date +%s%N) - start) / 1000000 ))"
}
(LAKE=lake_gc COMPACT_BYTES=2g nohup ./run_server.sh > server$RUN.log 2>&1 &)
for attempt in $(seq 60); do psql -h 127.0.0.1 -U pivot -d lake -At -c "SELECT 1" >/dev/null 2>&1 && break; sleep 1; done
for i in $(seq 5); do time_query > /dev/null; done
for i in $(seq 40); do time_query; done > q${RUN}_idle.txt
./create_table.sh busy_$RUN "with_pre_existing_parquets = '/adopt2/logs'" > /dev/null 2>&1
rm -f compact_done_$RUN
(psql -h 127.0.0.1 -U pivot -d lake -c "COMPACT busy_$RUN FINAL" > compact$RUN.out 2>&1; touch compact_done_$RUN) &
START=$(date +%s.%N)
while [ ! -e compact_done_$RUN ]; do time_query; done > q${RUN}_busy.txt
echo "compaction seconds: $(echo "$(date +%s.%N) - $START" | bc)"
pid=$(ps -eo pid,args | grep "pivot server --config" | grep -v grep | cut -c1-8)
kill $pid
summarize() { sort -k2 -n $1 | awk '{v[NR]=$2} END {printf "runs %d median %d p90 %d max %d ms\n", NR, v[int((NR+1)/2)], v[int(NR*0.9)], v[NR]}'; }
echo "idle: $(summarize q${RUN}_idle.txt)"
echo "busy: $(summarize q${RUN}_busy.txt)"
