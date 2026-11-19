"""Machine-only experiment: decode ranges of 65,536 rows and a 256 MiB budget."""
import sys

root = sys.argv[1] + '/parquet-engine/src/'
budget_mib = int(sys.argv[2]) if len(sys.argv) > 2 else 256
for name, a, b in [
    ('reading/range_cutter/row_group_pages.rs', 'pub const DECODE_RANGE_ROWS: u32 = 16 * 1024;',
     'pub const DECODE_RANGE_ROWS: u32 = 64 * 1024;'),
    ('writing/compaction/mod.rs', 'const MEMORY_BUDGET_BYTES: usize = 1 << 30;',
     f'const MEMORY_BUDGET_BYTES: usize = {budget_mib} << 20;'),
]:
    s = open(root + name).read()
    assert s.count(a) == 1, (name, a)
    open(root + name, 'w').write(s.replace(a, b))
print('big ranges applied, budget MiB', budget_mib)
