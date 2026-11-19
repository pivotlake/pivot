"""Machine-only: print, for every output row group sent, how many parts and
chunks it has and how many rows its chunks hold.

usage: group_probe.py <repository root>
"""
import os
import sys

p = os.path.expanduser(sys.argv[1]) + '/parquet-engine/src/writing/compaction/row_group_collector.rs'
s = open(p).read()
a = '''        // Chunks that are each consecutive rows, one after the other, are
        // the row group as they stand.
'''
assert s.count(a) == 1
s = s.replace(a, '''        eprintln!(
            "GROUP rows={} parts={} chunks={} chunk_rows={} gathered={}",
            row_count,
            gather_parts.len(),
            chunks.len(),
            chunks.iter().map(|chunk| chunk.row_count()).sum::<usize>(),
            !is_in_output_order,
        );
''' + a)
open(p, 'w').write(s)
print('group probe added')
