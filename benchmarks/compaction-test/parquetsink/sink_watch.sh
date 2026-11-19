#!/bin/bash
# Records the health of the pivot server on port 5432 once a minute, into
# ~/gilad-test/pivotdb/watch-c64f0934.log, for 13 hours. One line per minute:
# time, pid, resident memory (GB), memory blocks by state, compactions and
# committed COPY batches so far, warnings and errors so far, free disk.
cd ~/gilad-test/pivotdb || exit 1
OUT=watch-c64f0934.log
END=$(( $(date +%s) + 13 * 3600 ))
while [ $(date +%s) -lt $END ]; do
  START_LINE=$(grep -a -n "starting c64f0934" nohup.out | tail -1 | cut -d: -f1)
  PID=$(ss -ltnp 2>/dev/null | grep ":5432" | grep -o "pid=[0-9]*" | cut -d= -f2 | head -1)
  if [ -z "$PID" ]; then
    echo "$(date -u +%FT%TZ) SERVER_DOWN" >> $OUT
  else
    RSS=$(ps -o rss= -p $PID | awk '{printf "%.1f", $1 / 1048576}')
    BLOCKS=$(timeout 20 psql -h 127.0.0.1 -U pivot -d lake -At -c "SELECT state || '=' || count(*) FROM system.memory_blocks GROUP BY state ORDER BY state" 2>&1 | tr '\n' ' ')
    NEW=$(tail -n +$START_LINE nohup.out | tr -d '\033')
    COMPACTIONS=$(echo "$NEW" | grep -a -c "compacted batch")
    COPIES=$(echo "$NEW" | grep -a -c "copy from stdin committed")
    PROBLEMS=$(echo "$NEW" | grep -a -E " WARN | ERROR |panicked" | grep -a -v "Using null stats" | wc -l)
    STARTS=$(echo "$NEW" | grep -a -c "listening for psql connections")
    DISK=$(df -h / | tail -1 | awk '{print $4}')
    echo "$(date -u +%FT%TZ) pid=$PID rss_gb=$RSS $BLOCKS compactions=$COMPACTIONS copies=$COPIES problems=$PROBLEMS starts=$STARTS disk_free=$DISK" >> $OUT
  fi
  sleep 60
done
