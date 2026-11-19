"""usage: summarize_range_path.py <server log>
Summarises the range-path probe: per range, how long each step took."""
import re
import sys

release = {}
held = {}
passed = {}
first_arrive = {}
last_arrive = {}
read_issue = []
read_back = []
for line in open(sys.argv[1], errors='replace'):
    if line.startswith('RELEASE'):
        g, upto, t = map(int, re.findall(r'=(\d+)', line))
        release[(g, upto)] = t
    elif line.startswith('GATEHELD'):
        g, start, t = map(int, re.findall(r'=(\d+)', line))
        held[(g, start)] = t
    elif line.startswith('GATEPASS'):
        g, start, t = map(int, re.findall(r'=(\d+)', line))
        passed[(g, start)] = t
    elif line.startswith('ARRIVE'):
        g, first_row, rows, t = map(int, re.findall(r'=(\d+)', line))
        start = first_row // 16384 * 16384
        first_arrive.setdefault((g, start), t)
        last_arrive[(g, start)] = t
    elif line.startswith('READISSUE'):
        read_issue.append(int(re.findall(r'=(\d+)', line)[0]))
    elif line.startswith('READBACK'):
        read_back.append(int(re.findall(r'=(\d+)', line)[0]))


def show(name, values):
    values = sorted(values)
    if not values:
        print(f'{name}: none')
        return
    print(f'{name}: n={len(values)} median={values[len(values) // 2] / 1000:.1f} ms '
          f'p90={values[int(len(values) * .9)] / 1000:.1f} max={values[-1] / 1000:.1f} '
          f'sum={sum(values) / 1e6:.1f} s')


release_by_start = {}
for (g, upto), t in release.items():
    start = (upto - 1) // 16384 * 16384
    release_by_start[(g, start)] = t
keys = [k for k in release_by_start if k in passed and k in last_arrive]
show('release -> gate passes it', [passed[k] - release_by_start[k] for k in keys])
show('gate passes -> first batch arrives', [first_arrive[k] - passed[k] for k in keys])
show('gate passes -> last batch arrives', [last_arrive[k] - passed[k] for k in keys])
show('release -> last batch arrives', [last_arrive[k] - release_by_start[k] for k in keys])
late = [k for k in keys if k in held and held[k] > release_by_start[k]]
print('ranges released before they reached the gate (the merge waited for the read):', len(late), 'of', len(keys))
show('  those: release -> reached the gate', [held[k] - release_by_start[k] for k in late])
passed_at_once = [k for k in keys if k not in held]
print('ranges that passed the gate on arrival:', len(passed_at_once))
show('  those: release -> passed', [passed[k] - release_by_start[k] for k in passed_at_once])
print('reads issued', len(read_issue), 'back', len(read_back))
if read_issue:
    print('first read issued to last read back: %.1f s' % ((max(read_back) - min(read_issue)) / 1e6))
