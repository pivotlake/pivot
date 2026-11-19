#!/bin/bash
# usage: run_shapes.sh <first run number> <input prefix>...
# Compacts each input shape in turn, each on a fresh server, numbering the
# runs from the first run number. Writes SHAPES_DONE to shapes.out at the end.
cd ~/merge-test
run=$1
shift
for shape in "$@"; do
  ./run_compaction.sh $run shape_${shape}_$run /$shape/logs > run$run.out 2>&1
  pid=$(ps -eo pid,args | grep "pivot server --config" | grep -v grep | cut -c1-8)
  kill $pid
  sleep 5
  run=$((run + 1))
done
echo SHAPES_DONE
