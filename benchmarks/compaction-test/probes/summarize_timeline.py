"""usage: summarize_timeline.py <server log>
For each short-lived dataflow (a query), how long after the first worker received it the
last worker did, how long it lived, and how spread its finish was. Then the long steps by kind."""
import collections
import re
import sys

recv = collections.defaultdict(dict)
done = collections.defaultdict(dict)
steps = collections.defaultdict(list)
for line in open(sys.argv[1], errors='replace'):
    if line.startswith('DFRECV') or line.startswith('DFDONE'):
        w, df, t = map(int, re.findall(r'=(\d+)', line))
        (recv if line.startswith('DFRECV') else done)[df][w] = t
    elif line.startswith('STEP'):
        m = re.match(r'STEP df=(\d+) ms=([\d.]+) t=(\d+) kind=(.*)', line)
        if m:
            steps[m.group(4).split('::')[-2] if '::' in m.group(4) else m.group(4)].append(
                (float(m.group(2)), int(m.group(1))))


def show(name, values):
    values = sorted(values)
    if values:
        print(f'{name}: n={len(values)} median={values[len(values) // 2]:.1f} '
              f'p90={values[int(len(values) * .9)]:.1f} max={values[-1]:.1f} ms')


lifetimes = {}
for df in recv:
    if df in done and len(done[df]) == len(recv[df]):
        first = min(recv[df].values())
        lifetimes[df] = (max(done[df].values()) - first) / 1000
long_lived = {df for df, life in lifetimes.items() if life > 5000}
queries = [df for df in lifetimes if df not in long_lived]
print('dataflows:', len(lifetimes), 'of which longer than 5 s:', len(long_lived))
show('query: first worker receives -> last worker receives', [
    (max(recv[df].values()) - min(recv[df].values())) / 1000 for df in queries])
show('query: first receive -> first worker done', [
    (min(done[df].values()) - min(recv[df].values())) / 1000 for df in queries])
show('query: first worker done -> last worker done', [
    (max(done[df].values()) - min(done[df].values())) / 1000 for df in queries])
show('query: whole life', [lifetimes[df] for df in queries])
slow = sorted(queries, key=lambda df: -lifetimes[df])[:3]
for df in slow:
    first = min(recv[df].values())
    print(f'slowest dataflow {df}: life {lifetimes[df]:.0f} ms; receive offsets ms',
          sorted(round((t - first) / 1000) for t in recv[df].values()),
          'done offsets ms', sorted(round((t - first) / 1000) for t in done[df].values()))
print('steps of 3 ms or more, by kind:')
for kind, values in sorted(steps.items(), key=lambda item: -sum(v for v, _ in item[1])):
    durations = sorted(v for v, _ in values)
    in_queries = sum(1 for _, df in values if df in set(queries))
    print(f'  {kind}: n={len(durations)} median={durations[len(durations) // 2]:.1f} '
          f'max={durations[-1]:.1f} total={sum(durations) / 1000:.1f} s, in query dataflows: {in_queries}')
