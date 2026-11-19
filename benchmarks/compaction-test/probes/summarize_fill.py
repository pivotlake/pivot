"""Summarise FILL and MEMPROBE lines of a server log: how full the blocks that
hold input batches are."""
import re
import sys


def read(prefix):
    rows = []
    for line in open(sys.argv[1], errors='replace'):
        if line.startswith(prefix):
            rows.append({key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)})
    return rows[2:-1]


def median(rows, key):
    values = sorted(row[key] for row in rows)
    return values[len(values) // 2]


fill = read('FILL')
probe = read('MEMPROBE')
blocks = median(fill, 'input_blocks')
used = median(fill, 'used_mb')
print(len(fill), 'samples, typical values')
print(f'blocks holding input batches:            {blocks} = {blocks * 2} MB')
print(f'bytes of those that live batches use:    {used} MB = {100 * used // (blocks * 2)}% full')
print(f"blocks less than a tenth used:           {median(fill, 'blocks_under_tenth_used')}")
print(f"blocks less than half used:              {median(fill, 'blocks_under_half_used')}")
rows_alive = (median(probe, 'ready_rows') + median(probe, 'arriving_rows')
              + median(probe, 'collector_rows') + median(probe, 'inflight_rows'))
print(f'rows alive (ready + collector + in flight): about {rows_alive}')
print(f'bytes used per row alive:                {used * 1024 * 1024 // max(rows_alive, 1)}')
print(f"blocks referenced by input batches that are not counted as pinned: {median(probe, 'not_pinned')}")
