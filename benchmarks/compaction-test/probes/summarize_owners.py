"""Summarise OWNERS lines of a server log: pinned blocks by who took them."""
import collections
import re
import sys

samples = []
for line in open(sys.argv[1], errors='replace'):
    if not line.startswith('OWNERS'):
        continue
    parts = line.rstrip().split(' ;; ')
    pinned = int(re.search(r'pinned=(\d+)', parts[0]).group(1))
    owners = {}
    for item in parts[1:]:
        count, name = item.split('\t', 1)
        name = re.sub(r'[a-z_0-9]+::', '', name)
        owners[name] = owners.get(name, 0) + int(count)
    samples.append((pinned, owners))
samples = [sample for sample in samples if sample[0] > 1500]
print(len(samples), 'samples with over 1500 blocks pinned; blocks are 2 MiB')


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


totals = collections.Counter()
for _, owners in samples:
    for name, count in owners.items():
        totals[name] += count
print(f"{'median':>6} {'max':>6}")
print(f"{median([sample[0] for sample in samples]):>6} {max(sample[0] for sample in samples):>6}  total pinned")
for name, _ in totals.most_common(16):
    values = [owners.get(name, 0) for _, owners in samples]
    print(f'{median(values):>6} {max(values):>6}  {name[:100]}')
