#!/bin/bash
# Print the server's resident memory once a second until it exits or the
# file named by $1 appears: "<unix time> <resident MiB>".
PID=$(cat ~/merge-test/server.pid)
while [ -d /proc/$PID ] && [ ! -e "$1" ]; do
  echo "$(date +%s) $(( $(awk '/VmRSS/ {print $2}' /proc/$PID/status) / 1024 ))"
  sleep 1
done
