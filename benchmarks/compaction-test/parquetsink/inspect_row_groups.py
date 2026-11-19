"""Reads only the footers of a few data files of one deployment in the 17 November
partition and prints each file's row-group bounds on the sort columns, then how many
of each file's row groups lie inside each other file's Timestamp range."""
import json
import os
import subprocess
import sys

import pyarrow.parquet as pq

TABLE = 'gs://epsio-io-otel-parquet/iceberg-pivot/ab4b0685-a17f-47e1-8d4f-50299bca1706'
WORK = '/tmp/footer_probe'
os.makedirs(WORK, exist_ok=True)
rows = subprocess.run(
    ['psql', '-h', '127.0.0.1', '-U', 'pivot', '-d', 'lake', '-At', '-F', '|', '-c',
     "SELECT path, bytes, min_max_stats FROM system.table_files WHERE partition = 'Timestamp_day=2026-11-17'"],
    check=True, capture_output=True, text=True).stdout.splitlines()
chosen = []
for row in rows:
    path, size, stats = row.split('|', 2)
    stats = json.loads(stats)
    if stats['DeploymentId']['min'].startswith('52f') and stats['DeploymentId']['max'].startswith('52f'):
        chosen.append((path, int(size), stats['Timestamp']['min'][11:19], stats['Timestamp']['max'][11:19]))
chosen = sorted(chosen, key=lambda item: item[2])[:int(sys.argv[1]) if len(sys.argv) > 1 else 5]

files = []
for path, size, low, high in chosen:
    local = os.path.join(WORK, path)
    tail = subprocess.run(['gsutil', 'cat', '-r', f'{size - 8}-{size - 1}', f'{TABLE}/{path}'],
                          check=True, capture_output=True).stdout
    footer_len = int.from_bytes(tail[:4], 'little')
    footer = subprocess.run(['gsutil', 'cat', '-r', f'{size - 8 - footer_len}-{size - 1}', f'{TABLE}/{path}'],
                            check=True, capture_output=True).stdout
    with open(local, 'wb') as out:
        out.truncate(size)
        out.seek(size - len(footer))
        out.write(footer)
    metadata = pq.read_metadata(local)
    names = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
    columns = {name: names.index(name) for name in ('DeploymentId', 'ServiceName', 'Timestamp')}
    groups = []
    for index in range(metadata.num_row_groups):
        group = metadata.row_group(index)
        bounds = {}
        for name, column in columns.items():
            statistics = group.column(column).statistics
            bounds[name] = (statistics.min, statistics.max) if statistics and statistics.has_min_max else None
        groups.append((group.num_rows, bounds))
    files.append((path, low, high, groups))
    os.remove(local)
    print(f'{path[:20]}  file Timestamp {low}-{high}  row groups {len(groups)}  created_by {metadata.created_by}')
    for index, (num_rows, bounds) in enumerate(groups[:4] + groups[-2:]):
        ts = bounds['Timestamp']
        print('    rows %7d  Timestamp %s .. %s  Service %s' % (
            num_rows, str(ts[0])[11:19] if ts else None, str(ts[1])[11:19] if ts else None, bounds['ServiceName']))

print()
print('row groups of the row file that lie entirely inside the column file\'s Timestamp range (of total; 20%+1 needed):')
for path, low, high, groups in files:
    line = []
    for other_path, other_low, other_high, other_groups in files:
        if other_path == path:
            line.append('   -  ')
            continue
        ranges = [bounds['Timestamp'] for _, bounds in other_groups if bounds['Timestamp']]
        other_min = min(r[0] for r in ranges)
        other_max = max(r[1] for r in ranges)
        inside = sum(1 for _, bounds in groups
                     if bounds['Timestamp'] is None
                     or (bounds['Timestamp'][0] >= other_min and bounds['Timestamp'][1] <= other_max))
        line.append('%2d/%2d ' % (inside, len(groups)))
    print(f'{low}-{high}  ' + ' '.join(line) + f'  needs {len(groups) * 20 // 100 + 1}')
