#!/bin/bash
# usage: run_compaction.sh <run number> <table name>
# Starts the server, adopts the five input files with a page index as a new
# table, and compacts it while recording the buffer pool once a second.
set -uo pipefail
RUN=$1
TABLE=$2
INPUTS=${3:-/adopt2/logs}
cd ~/merge-test
rm -f stop_$RUN
(nohup ./run_server.sh > server$RUN.log 2>&1 &)
for attempt in $(seq 60); do psql -h 127.0.0.1 -U pivot -d lake -At -c "SELECT 1" >/dev/null 2>&1 && break; sleep 1; done
./create_table.sh $TABLE "with_pre_existing_parquets = '$INPUTS'" 2>&1 | tail -2
psql -h 127.0.0.1 -U pivot -d lake -At -c "SELECT count(*) FROM $TABLE"
(nohup ./watch_pool.sh stop_$RUN > pool$RUN.txt 2>&1 &)
START=$(date +%s.%N)
psql -h 127.0.0.1 -U pivot -d lake -c "COMPACT $TABLE FINAL" 2>&1 | tail -3
END=$(date +%s.%N)
touch stop_$RUN
echo "compaction seconds: $(echo "$END - $START" | bc)"
psql -h 127.0.0.1 -U pivot -d lake -At -c "SELECT count(*) FROM $TABLE"
grep -ciE 'panic|error' server$RUN.log
