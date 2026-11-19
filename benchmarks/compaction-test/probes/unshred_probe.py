"""Machine-only: print, for every variant column folded back, how many rows it
had and how long it took.

usage: unshred_probe.py <repository root>
"""
import os
import sys

p = os.path.expanduser(sys.argv[1]) + '/parquet-engine/src/writing/shredding/mod.rs'
s = open(p).read()
count = 0
for call in ['unshred_variant(array)?', 'unshred_variant(&variant)?']:
    a = f'        let unshredded = {call};\n' if 'array' in call else f'    let unshredded = {call};\n'
    if a not in s:
        continue
    rows = 'array.len()' if 'array' in call else 'variant.len()'
    indent = a[:len(a) - len(a.lstrip())]
    b = (f'{indent}let probe_started = std::time::Instant::now();\n{a}'
         f'{indent}eprintln!("UNSHRED rows={{}} ms={{:.3}}", {rows}, '
         f'probe_started.elapsed().as_secs_f64() * 1000.0);\n')
    assert s.count(a) == 1, a
    s = s.replace(a, b)
    count += 1
assert count > 0
open(p, 'w').write(s)
print(f'unshred probe added to {count} call(s)')
