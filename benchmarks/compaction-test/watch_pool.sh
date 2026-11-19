#!/bin/bash
# Print the buffer pool's blocks by state once a second until the file named
# by $1 appears.
while [ ! -e "$1" ]; do
  echo "$(date +%T) $(psql -h 127.0.0.1 -U pivot -d lake -At -c "SELECT state || '=' || count(*) FROM system.memory_blocks GROUP BY state ORDER BY state" 2>&1 | tr '\n' ' ')"
  sleep 1
done
