#!/bin/bash
# usage: time_topn.sh <runs>
# Times the latest-row query <runs> times, half a second apart. For each run
# prints: start time, milliseconds taken, and how many cores the server used
# in the second before it (from its CPU time), as a measure of how busy it was.
cd ~/gilad-test/pivotdb
PID=$(ss -ltnp | grep ":5432" | grep -o "pid=[0-9]*" | cut -d= -f2)
TICKS=$(getconf CLK_TCK)
read_cpu() { awk '{print $14 + $15}' /proc/$PID/stat; }
for run in $(seq $1); do
  CPU_BEFORE=$(read_cpu)
  sleep 1
  CPU_AFTER=$(read_cpu)
  START_NS=$(date +%s%N)
  psql -h 127.0.0.1 -U pivot -d lake -At -c 'select * FROM otel_logs ORDER BY "Timestamp" DESC LIMIT 1' > /dev/null 2>/tmp/topn.err
  END_NS=$(date +%s%N)
  echo "$(date -u -d @$((START_NS / 1000000000)) +%T) ms=$(( (END_NS - START_NS) / 1000000 )) cores_before=$(( (CPU_AFTER - CPU_BEFORE) / TICKS ))"
done
