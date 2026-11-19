"""Summarise SLOW lines of a server log: calls of 5 ms or more by operator."""
import collections
import re
import sys

pattern = re.compile(r'^SLOW (run|consume) (\S+) ([0-9.]+)$')
calls = collections.defaultdict(list)
for line in open(sys.argv[1], errors='replace'):
    match = pattern.match(line.strip())
    if match:
        name = re.sub(r'[a-z_0-9]+::', '', match.group(2))
        calls[(match.group(1), name)].append(float(match.group(3)))

print(f"{'calls':>7} {'median':>8} {'p99':>8} {'max':>8} {'total s':>8}  operator")
for (what, name), values in sorted(calls.items(), key=lambda item: -max(item[1])):
    values.sort()
    count = len(values)
    print(
        f"{count:>7} {values[count // 2]:>8.1f} {values[min(count - 1, int(count * 0.99))]:>8.1f} "
        f"{values[-1]:>8.1f} {sum(values) / 1000:>8.1f}  {what} {name[:90]}"
    )
