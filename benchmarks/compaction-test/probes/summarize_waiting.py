"""From a server log with COLLECTOR, PLANNER and MEMPROBE lines: how often the
pipeline was held back, and by what."""
import re
import sys


def read(prefix):
    rows = []
    for line in open(sys.argv[1], errors='replace'):
        if line.startswith(prefix):
            row = {}
            for key, value in re.findall(r'(\w+)=(\w+)', line):
                row[key] = int(value) if value.isdigit() else value
            rows.append(row)
    return rows


collector = read('COLLECTOR')[2:-1]
planner = read('PLANNER')[2:-1]
probe = read('MEMPROBE')[2:-1]


def share(rows, test):
    return 100 * sum(1 for row in rows if test(row)) // max(len(rows), 1)


print(f'{len(collector)} collector samples, {len(planner)} planner samples, {len(probe)} memory samples')
print()
print('Collector:')
print(f"  a finished row group is waiting for room:  {share(collector, lambda r: r['complete_groups'] > 0)}% of samples")
print(f"  no finished row group, still filling one:   {share(collector, lambda r: r['complete_groups'] == 0)}% of samples")
print(f"  finished row groups waiting, largest:       {max(r['complete_groups'] for r in collector)}")
print('Planner:')
print(f"  merge jobs held back (row group waiting):   {share(planner, lambda r: r['waiting'] == 'true')}% of samples")
print(f"  merge jobs queued and not sent:             {share(planner, lambda r: r['queued_jobs'] > 0)}% of samples")
print('Rows:')
for key in ['ready_rows', 'arriving_rows', 'collector_rows', 'inflight_rows']:
    values = sorted(r[key] for r in probe)
    print(f"  {key:>15}: median {values[len(values) // 2]:>7}, max {values[-1]:>7}")
print(f"  row groups in flight: 0 in {share(probe, lambda r: r['inflight_groups'] == 0)}%, "
      f"1 in {share(probe, lambda r: r['inflight_groups'] == 1)}%, "
      f"2 or more in {share(probe, lambda r: r['inflight_groups'] >= 2)}% of samples")
sent = [r['row_groups_sent'] for r in collector]
print(f'Row groups sent between first and last sample: {sent[-1] - sent[0]} in about {len(collector) / 2:.0f} s')
