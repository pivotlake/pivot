"""Machine-only probe: how many decoded rows hold a reservation, and how many bytes."""
import sys

c = sys.argv[1] + '/parquet-engine/src/writing/compaction/'


def edit(name, pairs):
    s = open(c + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(c + name, 'w').write(s)


edit('memory_budget.rs', [
    ('''pub(super) struct MemoryBudget {
    reserved_bytes: AtomicUsize,''', '''pub(super) struct MemoryBudget {
    pub(super) reserved_rows: AtomicUsize,
    pub(super) reserved_bytes: AtomicUsize,'''),
    ('''        Arc::new(Self {
            reserved_bytes: AtomicUsize::new(0),''', '''        Arc::new(Self {
            reserved_rows: AtomicUsize::new(0),
            reserved_bytes: AtomicUsize::new(0),'''),
    ('''    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> Arc<MemoryReservation> {
        self.reserved_bytes.fetch_add(bytes, Ordering::Release);
        Arc::new(MemoryReservation {
            budget: self.clone(),
            bytes,''', '''    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> Arc<MemoryReservation> {
        self.reserve_rows(bytes, 0)
    }

    pub(super) fn reserve_rows(self: &Arc<Self>, bytes: usize, rows: usize) -> Arc<MemoryReservation> {
        self.reserved_bytes.fetch_add(bytes, Ordering::Release);
        self.reserved_rows.fetch_add(rows, Ordering::Release);
        Arc::new(MemoryReservation {
            budget: self.clone(),
            rows,
            bytes,'''),
    ('''pub struct MemoryReservation {
    budget: Arc<MemoryBudget>,''', '''pub struct MemoryReservation {
    rows: usize,
    budget: Arc<MemoryBudget>,'''),
    ('''        Arc::new(MemoryReservation {
            budget: self.budget.clone(),
            bytes,''', '''        Arc::new(MemoryReservation {
            rows: 0,
            budget: self.budget.clone(),
            bytes,'''),
    ('''impl Drop for MemoryReservation {
    fn drop(&mut self) {''', '''impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.budget.reserved_rows.fetch_sub(self.rows, Ordering::Release);'''),
    ('''    pub(super) fn has_row_group_in_flight(&self) -> bool {''', '''    pub(super) fn count_row_groups_in_flight(&self) -> usize {
        self.row_groups_in_flight.load(Ordering::Acquire)
    }

    pub(super) fn has_row_group_in_flight(&self) -> bool {'''),
])

edit('merge/planner.rs', [
    ('''                memory: self
                    .memory_budget
                    .reserve(decoded.batch.get_array_memory_size()),''', '''                memory: self
                    .memory_budget
                    .reserve_rows(decoded.batch.get_array_memory_size(), decoded.batch.num_rows()),'''),
    ('''            self.files[file].add_batch(batch, &self.key_columns, &self.order_by)?;
            self.read_ahead();''', '''            self.files[file].add_batch(batch, &self.key_columns, &self.order_by)?;
            self.read_ahead();
            {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static BATCHES: AtomicUsize = AtomicUsize::new(0);
                if BATCHES.fetch_add(1, Ordering::Relaxed) % 50 == 0 {
                    let rows_ahead: Vec<usize> =
                        self.files.iter().map(|file| file.count_rows_being_decoded()).collect();
                    eprintln!(
                        "ROWS reserved_rows={} reserved_mb={} in_flight={} rows_being_decoded_per_file={:?}",
                        self.memory_budget.reserved_rows.load(Ordering::Relaxed),
                        self.memory_budget.reserved_bytes.load(Ordering::Relaxed) >> 20,
                        self.memory_budget.count_row_groups_in_flight(),
                        rows_ahead,
                    );
                }
            }'''),
])
print('rows probe applied')
