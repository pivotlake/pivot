"""From a server log with RELEASED, GATE and DECODED lines: where each range
that was released and never decoded is."""
import collections
import re
import sys

released, held, passed, decoded = set(), set(), set(), collections.Counter()
for line in open(sys.argv[1], errors='replace'):
    match = re.search(r'row_group=(\d+) first_row=(\d+)', line)
    if not match:
        continue
    row_group, first_row = int(match.group(1)), int(match.group(2))
    if line.startswith('RELEASED'):
        released.add((row_group, first_row))
    elif line.startswith('GATE held'):
        held.add((row_group, first_row))
    elif line.startswith('GATE passed'):
        passed.add((row_group, first_row))
    elif line.startswith('DECODED'):
        # A batch belongs to the range its first row is in.
        decoded[(row_group, first_row // 16384 * 16384)] += 1

print(f'ranges released: {len(released)}, reached the gate: {len(held | passed)}, '
      f'passed the gate: {len(passed)}, with a decoded batch: {len(decoded)}')
missing = sorted(released - set(decoded))
print(f'released and not decoded: {len(missing)}')
for range_ in missing[:20]:
    where = ('passed the gate' if range_ in passed
             else 'held at the gate' if range_ in held
             else 'never reached the gate')
    print(f'  row_group={range_[0]} first_row={range_[1]}: {where}')
at_gate_unreleased = sorted((held | passed) - released)
print(f'at the gate and not released: {len(at_gate_unreleased)}')
row_groups_at_gate = sorted({range_[0] for range_ in held | passed})
print(f'row groups with ranges at the gate: {row_groups_at_gate}')
print(f'row groups released from: {sorted({range_[0] for range_ in released})}')
