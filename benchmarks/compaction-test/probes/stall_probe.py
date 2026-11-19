"""Machine-only: print the planner's and the budget's state once a second,
to see what a stalled compaction is waiting on.

usage: stall_probe.py <repository root>
"""
import os
import sys

c = os.path.expanduser(sys.argv[1]) + '/parquet-engine/src/writing/compaction/'


def edit(name, pairs):
    s = open(c + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(c + name, 'w').write(s)


edit('memory_budget.rs', [
    ('''    /// Whether an output row group is being encoded or written.''',
     '''    /// Machine-only probe.
    pub(super) fn probe_state(&self) -> (usize, usize) {
        (
            self.reserved_bytes.load(Ordering::Acquire) >> 20,
            self.row_groups_in_flight.load(Ordering::Acquire),
        )
    }

    /// Whether an output row group is being encoded or written.'''),
])

edit('merge/input_file_rows.rs', [
    ('''    /// How many of the file's rows are released to be decoded and not
    /// merged: being decoded, or ready.''', '''    /// Machine-only probe.
    pub(super) fn probe_state(&self) -> String {
        format!(
            "[unrequested={} requested={} ready_rows={} ahead={} next_range={} released_not_ready={}]",
            self.unrequested_row_groups.len(),
            self.requested.len(),
            self.ready_row_count,
            self.count_rows_ahead(),
            self.count_next_range_rows(),
            self.requested
                .iter()
                .map(|row_group| row_group.rows_released - row_group.rows_made_ready)
                .sum::<usize>(),
        )
    }

    /// How many of the file's rows are released to be decoded and not
    /// merged: being decoded, or ready.'''),
])

edit('merge/planner.rs', [
    ('''        // Memory given back since the last time may have made room.
        self.read_ahead();''', '''        // Memory given back since the last time may have made room.
        self.read_ahead();
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static LAST: AtomicU64 = AtomicU64::new(0);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            if self.row_count > 0 && LAST.swap(now, Ordering::Relaxed) != now {
                let (reserved_mb, in_flight) = self.memory_budget.probe_state();
                let files: Vec<String> = self.files.iter().map(|file| file.probe_state()).collect();
                eprintln!(
                    "STALL reserved_mb={reserved_mb} in_flight={in_flight} queued_jobs={} rows_sent={} of {} bytes_per_row={} files={}",
                    self.jobs.len(),
                    self.rows_sent,
                    self.row_count,
                    if self.decoded_rows == 0 { 0 } else { self.decoded_bytes / self.decoded_rows },
                    files.join(" "),
                );
            }
        }'''),
])

edit('merge/planner.rs', [
    ('''                self.rows_sent += job.row_count();
                sender.send(self.jobs.pop_front().unwrap())?;''', '''                self.rows_sent += job.row_count();
                eprintln!("JOBSENT sequence={} rows={}", job.sequence, job.row_count());
                sender.send(self.jobs.pop_front().unwrap())?;'''),
])

edit('merge/mod.rs', [
    ('''    fn merge_into_run(self, allocator: &mut Option<SlabAllocator>) -> UnaryResult<MergedRun> {''',
     '''    fn merge_into_run(self, allocator: &mut Option<SlabAllocator>) -> UnaryResult<MergedRun> {
        eprintln!("JOBMERGING sequence={}", self.sequence);'''),
    ('''        Ok(MergedRun {
            sequence: self.sequence,''', '''        eprintln!("JOBMERGED sequence={}", self.sequence);
        Ok(MergedRun {
            sequence: self.sequence,'''),
])

edit('merge/decode_gate.rs', [
    ('''        if self.is_released(&range) {
            sender.send(range)?;
        } else {
            self.held.push(range);
        }''', '''        let label = format!(
            "row_group={} first_row={}",
            range.metadata().metadata().row_group,
            range.rows().start
        );
        if self.is_released(&range) {
            eprintln!("GATE passed on arrival {label}");
            sender.send(range)?;
        } else {
            eprintln!("GATE held {label}");
            self.held.push(range);
        }'''),
    ('''            if self.is_released(&self.held[index]) {
                sender.send(self.held.swap_remove(index))?;''', '''            if self.is_released(&self.held[index]) {
                eprintln!(
                    "GATE passed later row_group={} first_row={}",
                    self.held[index].metadata().metadata().row_group,
                    self.held[index].rows().start
                );
                sender.send(self.held.swap_remove(index))?;'''),
    ('''        self.rows_released[row_group].store(rows.end as u32, Ordering::Release);''',
     '''        eprintln!("RELEASED row_group={row_group} first_row={}", rows.start);
        self.rows_released[row_group].store(rows.end as u32, Ordering::Release);'''),
])

edit('input_batch.rs', [
    ('''    fn attach(&self, batch: RecordBatch, rows: Range<usize>) -> InputBatch {''',
     '''    fn attach(&self, batch: RecordBatch, rows: Range<usize>) -> InputBatch {
        eprintln!("DECODED row_group={} first_row={}", self.row_group, rows.start);'''),
])

edit('row_group_collector.rs', [
    ('''        self.arrived.insert(run.sequence, run);''', '''        self.arrived.insert(run.sequence, run);
        eprintln!(
            "COLLECTED arrived={} next_sequence={} early_runs={} rows_taken={} rows_filling={} row_groups_sent={}",
            self.arrived.keys().next_back().copied().unwrap_or(0),
            self.next_sequence,
            self.arrived.len(),
            self.rows_taken,
            self.rows_filling,
            self.row_groups_sent,
        );'''),
])
print('stall probe added')
