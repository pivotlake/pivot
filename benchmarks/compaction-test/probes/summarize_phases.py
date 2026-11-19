"""Summarise PHASE lines: encoder first-step parts of 5 ms or more."""
import collections
import sys

by_phase = collections.defaultdict(list)
by_column = collections.defaultdict(list)
for line in open(sys.argv[1]):
    if not line.startswith('PHASE '):
        continue
    _, phase, millis, path, _ = line.split()
    by_phase[phase].append(float(millis))
    by_column[(phase, path.split('.')[0])].append(float(millis))

print(f"{'calls':>7} {'median':>8} {'max':>8} {'total s':>8}  phase")
for phase, values in sorted(by_phase.items(), key=lambda item: -sum(item[1])):
    values.sort()
    print(f"{len(values):>7} {values[len(values) // 2]:>8.1f} {values[-1]:>8.1f} {sum(values) / 1000:>8.1f}  {phase}")
print()
print(f"{'calls':>7} {'median':>8} {'max':>8}  phase, top-level column (ten slowest)")
for (phase, column), values in sorted(by_column.items(), key=lambda item: -max(item[1]))[:10]:
    values.sort()
    print(f"{len(values):>7} {values[len(values) // 2]:>8.1f} {values[-1]:>8.1f}  {phase} {column}")
