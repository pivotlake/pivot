import os

p = os.path.expanduser('~/pivot-merge/parquet-engine/src/writing/merge.rs')
s = open(p).read()


def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)


# A once-a-second gate shared by the probes.
once('''/// How far the merge's stages have got. All of them but the writer's run on''',
     '''fn probe_due(last: &std::sync::atomic::AtomicU64) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let previous = last.load(Ordering::Relaxed);
    now > previous + 500
        && last
            .compare_exchange(previous, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
}
static LAYOUT_PROBE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static KEY_PROBE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CUTTER_PROBE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How far the merge's stages have got. All of them but the writer's run on''')

# Layout: resident input row groups and their size.
once('''    fn advance(&mut self, sender: &mut dyn Sender<GatherJob>) -> UnaryResult<bool> {
        let mut laid_out = false;''',
     '''    fn advance(&mut self, sender: &mut dyn Sender<GatherJob>) -> UnaryResult<bool> {
        if probe_due(&LAYOUT_PROBE) {
            let mut whole = 0usize;
            let mut arriving = 0usize;
            let mut bytes = 0usize;
            for group in self.resident.values() {
                match group {
                    ResidentRowGroup::Whole(fetched) => {
                        whole += 1;
                        bytes += fetched
                            .batches
                            .iter()
                            .map(|batch| batch.get_array_memory_size())
                            .sum::<usize>();
                    }
                    ResidentRowGroup::Arriving { batches, .. } => {
                        arriving += 1;
                        bytes += batches
                            .iter()
                            .map(|(_, batch)| batch.get_array_memory_size())
                            .sum::<usize>();
                    }
                }
            }
            eprintln!(
                "PROBE layout: laid_out={} written={} resident_whole={} resident_arriving={} resident_mib={} next_stretch={} order_queued={}",
                self.rows_laid_out,
                self.progress.rows_written.load(Ordering::Relaxed),
                whole,
                arriving,
                bytes >> 20,
                self.next_stretch.is_some(),
                self.order.stretches.len(),
            );
        }
        let mut laid_out = false;''')

# Key merge: keys held.
once('''    fn advance(&mut self, sender: &mut dyn Sender<KeyMergeJob>) -> UnaryResult<bool> {
        let mut released = false;''',
     '''    fn advance(&mut self, sender: &mut dyn Sender<KeyMergeJob>) -> UnaryResult<bool> {
        if probe_due(&KEY_PROBE) {
            let ready_rows: usize = self.files.iter().map(|file| file.ready_rows).sum();
            let ready_bytes: usize = self
                .files
                .iter()
                .flat_map(|file| file.ready.iter())
                .map(|batch| batch.get_array_memory_size())
                .sum();
            let arriving: usize = self.files.iter().map(|file| file.arriving.len()).sum();
            eprintln!(
                "PROBE keys: released={} ready_rows={} ready_mib={} arriving_groups={} jobs_queued={} unrequested_groups={}",
                self.rows_released,
                ready_rows,
                ready_bytes >> 20,
                arriving,
                self.jobs.len(),
                self.unrequested_groups,
            );
        }
        let mut released = false;''')

# Cutter: rows waiting to be cut and stretches out of order.
once('''        self.arrived.insert(stretch.sequence, stretch);
        while let Some(entry) = self.arrived.first_entry() {
            if *entry.key() != self.next_sequence {
                break;
            }
            let stretch = entry.remove();
            self.pending_rows += stretch.batch.num_rows();''',
     '''        self.arrived.insert(stretch.sequence, stretch);
        if probe_due(&CUTTER_PROBE) {
            eprintln!(
                "PROBE cutter: next_sequence={} out_of_order={} pending_rows={} row_groups_emitted={}",
                self.next_sequence,
                self.arrived.len(),
                self.pending_rows,
                self.row_groups_emitted,
            );
        }
        while let Some(entry) = self.arrived.first_entry() {
            if *entry.key() != self.next_sequence {
                break;
            }
            let stretch = entry.remove();
            self.pending_rows += stretch.batch.num_rows();''')

open(p, 'w').write(s)
print('instrumented')
