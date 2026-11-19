"""Machine-only probe: timestamps along a range's way from release to arrival, and of each read.
Lines: RELEASE g upto t, GATEHELD g start t, GATEPASS g start t, ARRIVE g first_row rows t,
READISSUE g t, READBACK g t (t in microseconds)."""
import sys

root = sys.argv[1] + '/parquet-engine/src/'
NOW = 'std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros()'


def edit(name, pairs):
    s = open(root + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(root + name, 'w').write(s)


edit('writing/compaction/merge/input_file_rows.rs', [
    ('''        releases.release_rows(row_group.group, row_group.rows_released);''',
     '''        releases.release_rows(row_group.group, row_group.rows_released);
        eprintln!("RELEASE g={} upto={} t={}", row_group.group, row_group.rows_released, %s);''' % NOW),
])
edit('writing/compaction/merge/decode_gate.rs', [
    ('''        if self.is_released(&range) {
            sender.send(range)?;
        } else {
            self.held.push(range);
        }''', '''        if self.is_released(&range) {
            eprintln!("GATEPASS g={} start={} t={}", range.metadata().row_group_index, range.rows().start, %s);
            sender.send(range)?;
        } else {
            eprintln!("GATEHELD g={} start={} t={}", range.metadata().row_group_index, range.rows().start, %s);
            self.held.push(range);
        }''' % (NOW, NOW)),
    ('''            if self.is_released(&self.held[index]) {
                sender.send(self.held.swap_remove(index))?;''', '''            if self.is_released(&self.held[index]) {
                eprintln!("GATEPASS g={} start={} t={}", self.held[index].metadata().row_group_index, self.held[index].rows().start, %s);
                sender.send(self.held.swap_remove(index))?;''' % NOW),
])
edit('writing/compaction/merge/planner.rs', [
    ('''        if decoded.batch.num_rows() > 0 {''', '''        if decoded.batch.num_rows() > 0 {
            eprintln!("ARRIVE g={} first_row={} rows={} t={}", decoded.row_group, decoded.first_row, decoded.batch.num_rows(), %s);''' % NOW),
])
edit('reading/fetching/fetcher.rs', [
    ('''            eprintln!("FETCHISSUE t_ms={}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());''',
     '''            eprintln!("READISSUE t={}", %s);''' % NOW),
    ('''        self.fetched_at.push_back(std::time::Instant::now());''',
     '''        self.fetched_at.push_back(std::time::Instant::now());
        eprintln!("READBACK t={}", %s);''' % NOW),
])
print('range path probe applied')
