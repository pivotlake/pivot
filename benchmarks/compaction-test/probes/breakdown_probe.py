"""Machine-only probe: what the compaction's memory is made of.
Prints, every 50 decoded batches: the bytes of decompressed pages alive, how many
ranges wait at the gate, the bytes the budget has reserved, row groups in flight,
and per file the rows asked for and not released, and the rows being decoded."""
import sys

root = sys.argv[1] + '/parquet-engine/src/'


def edit(name, pairs):
    s = open(root + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(root + name, 'w').write(s)


edit('reading/decompressor.rs', [
    ('''            slabs
                .into_iter()
                .map(|slab| Bytes::from_owner(slab.freeze()))
                .collect()''', '''            slabs
                .into_iter()
                .map(|slab| {
                    let frozen = slab.freeze();
                    let len = AsRef::<[u8]>::as_ref(&frozen).len();
                    LIVE_DECOMPRESSED_BYTES.fetch_add(len, std::sync::atomic::Ordering::Relaxed);
                    Bytes::from_owner(CountedPage(frozen, len))
                })
                .collect()'''),
])
s = open(root + 'reading/decompressor.rs').read()
s += '''
pub static LIVE_DECOMPRESSED_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

struct CountedPage(dispatch::memory::FrozenSlab, usize);

impl AsRef<[u8]> for CountedPage {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl Drop for CountedPage {
    fn drop(&mut self) {
        LIVE_DECOMPRESSED_BYTES.fetch_sub(self.1, std::sync::atomic::Ordering::Relaxed);
    }
}
'''
open(root + 'reading/decompressor.rs', 'w').write(s)

edit('writing/compaction/merge/decode_gate.rs', [
    ('''            self.held.push(range);''', '''            self.held.push(range);
            GATE_HELD_RANGES.store(self.held.len(), std::sync::atomic::Ordering::Relaxed);'''),
    ('''                sender.send(self.held.swap_remove(index))?;''', '''                sender.send(self.held.swap_remove(index))?;
                GATE_HELD_RANGES.store(self.held.len(), std::sync::atomic::Ordering::Relaxed);'''),
])
s = open(root + 'writing/compaction/merge/decode_gate.rs').read()
s += '''
pub(crate) static GATE_HELD_RANGES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
'''
open(root + 'writing/compaction/merge/decode_gate.rs', 'w').write(s)

edit('writing/compaction/memory_budget.rs', [
    ('''    pub(super) fn is_full(&self) -> bool {''', '''    pub(super) fn probe_state(&self) -> (usize, usize) {
        (
            self.reserved_bytes.load(Ordering::Relaxed),
            self.row_groups_in_flight.load(Ordering::Relaxed),
        )
    }

    pub(super) fn is_full(&self) -> bool {'''),
])
edit('writing/compaction/merge/input_file_rows.rs', [
    ('''    pub(super) fn has_ready_rows(&self) -> bool {''', '''    pub(super) fn probe_rows(&self) -> (usize, usize) {
        (
            self.requested
                .iter()
                .map(|row_group| row_group.row_count - row_group.rows_released)
                .sum(),
            self.ready_row_count,
        )
    }

    pub(super) fn has_ready_rows(&self) -> bool {'''),
])
edit('writing/compaction/merge/planner.rs', [
    ('''            self.files[file].add_batch(batch, &self.key_columns, &self.order_by)?;
            self.read_ahead();''', '''            self.files[file].add_batch(batch, &self.key_columns, &self.order_by)?;
            self.read_ahead();
            {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static BATCHES: AtomicUsize = AtomicUsize::new(0);
                if BATCHES.fetch_add(1, Ordering::Relaxed) % 50 == 0 {
                    let (reserved, in_flight) = self.memory_budget.probe_state();
                    let unreleased: Vec<usize> = self.files.iter().map(|f| f.probe_rows().0).collect();
                    let ready: Vec<usize> = self.files.iter().map(|f| f.probe_rows().1).collect();
                    let decoding: Vec<usize> =
                        self.files.iter().map(|f| f.count_rows_being_decoded()).collect();
                    eprintln!(
                        "BREAKDOWN t={} live_decompressed_mb={} gate_held_ranges={} reserved_mb={} in_flight={} unreleased={:?} ready={:?} decoding={:?}",
                        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
                        crate::reading::decompressor::LIVE_DECOMPRESSED_BYTES.load(Ordering::Relaxed) >> 20,
                        super::decode_gate::GATE_HELD_RANGES.load(Ordering::Relaxed),
                        reserved >> 20,
                        in_flight,
                        unreleased,
                        ready,
                        decoding,
                    );
                }
            }'''),
])
edit('reading/mod.rs', [('mod decompressor;', 'pub(crate) mod decompressor;')])
print('breakdown probe applied')
