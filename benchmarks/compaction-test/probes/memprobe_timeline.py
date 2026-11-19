"""Print MEMPROBE samples of a server log as a timeline."""
import re
import sys

rows = []
for line in open(sys.argv[1], errors='replace'):
    if line.startswith('MEMPROBE'):
        rows.append({key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)})
start = rows[0]['t']
print('  time  pinned  planner_blocks  ready_rows  arriving_rows  in_flight_groups  in_flight_rows')
for row in rows:
    print(f"{(row['t'] - start) / 1000:5.1f}s {row['pinned']:7} {row['planner']:15} {row['ready_rows']:11} "
          f"{row['arriving_rows']:14} {row['inflight_groups']:17} {row['inflight_rows']:15}")
