"""Machine-only: report, twice a second, which blocks of the memory ring the
compaction's held input batches are in, by where they are held.

usage: mem_probe.py <repository root>
"""
import os
import sys

root = os.path.expanduser(sys.argv[1])
d = root + '/parquet-engine/src/writing/compaction/'


def edit(name, pairs):
    s = open(d + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(d + name, 'w').write(s)


open(d + 'probe.rs', 'w').write(r'''//! Machine-only memory probe.

use std::collections::{HashMap, HashSet};

/// The parts of ring blocks that batches' buffers are in: for each block and
/// buffer, by the buffer's address, the bytes of the buffer in that block.
pub(super) type Slots = HashMap<(u32, usize), usize>;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Array, RecordBatch};
use arrow_data::ArrayData;
use dispatch::memory::{BUFFER_SIZE, MemoryBlockState, memory_ctx};

static LAST_REPORT_MS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

pub(super) fn is_due(which: usize) -> bool {
    let now = now_ms();
    if now - LAST_REPORT_MS[which].load(Ordering::Relaxed) >= 500 {
        LAST_REPORT_MS[which].store(now, Ordering::Relaxed);
        true
    } else {
        false
    }
}

/// Add the ring slots `batch`'s buffers are in. Returns the bytes of its
/// buffers that are not in the ring.
pub(super) fn add_slots(batch: &RecordBatch, slots: &mut Slots) -> usize {
    let ring = memory_ctx().ring();
    let base = ring.slots[0].buffer as usize;
    let end = base + ring.len() * BUFFER_SIZE;
    let mut heap_bytes = 0;
    for column in batch.columns() {
        add_data_slots(&column.to_data(), base, end, slots, &mut heap_bytes);
    }
    heap_bytes
}

fn add_data_slots(
    data: &ArrayData,
    base: usize,
    end: usize,
    slots: &mut Slots,
    heap_bytes: &mut usize,
) {
    let mut add_range = |pointer: usize, len: usize| {
        if len == 0 {
            return;
        }
        if pointer < base || pointer >= end {
            *heap_bytes += len;
            return;
        }
        for slot in (pointer - base) / BUFFER_SIZE..=(pointer + len - 1 - base) / BUFFER_SIZE {
            let slot_start = base + slot * BUFFER_SIZE;
            let bytes_in_slot =
                (pointer + len).min(slot_start + BUFFER_SIZE) - pointer.max(slot_start);
            slots.insert((slot as u32, pointer), bytes_in_slot);
        }
    };
    for buffer in data.buffers() {
        add_range(buffer.as_ptr() as usize, buffer.len());
    }
    if let Some(nulls) = data.nulls() {
        add_range(nulls.buffer().as_ptr() as usize, nulls.buffer().len());
    }
    for child in data.child_data() {
        add_data_slots(child, base, end, slots, heap_bytes);
    }
}

#[derive(Default)]
struct CollectorHeld {
    held: Slots,
    held_rows: usize,
    in_flight: Slots,
    in_flight_groups: usize,
    in_flight_rows: usize,
}

static COLLECTOR: Mutex<Option<CollectorHeld>> = Mutex::new(None);

pub(super) fn set_collector(
    held: Slots,
    held_rows: usize,
    in_flight: Slots,
    in_flight_groups: usize,
    in_flight_rows: usize,
) {
    *COLLECTOR.lock().unwrap() = Some(CollectorHeld {
        held,
        held_rows,
        in_flight,
        in_flight_groups,
        in_flight_rows,
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn report(
    arriving: Slots,
    arriving_rows: usize,
    ready: Slots,
    ready_rows: usize,
    queued: Slots,
    queued_rows: usize,
    heap_bytes: usize,
) {
    let blocks = memory_ctx().read_all_blocks();
    let pinned: HashSet<u32> = blocks
        .iter()
        .filter(|block| matches!(block.state, MemoryBlockState::Pinned))
        .map(|block| block.slot as u32)
        .collect();
    let collector = COLLECTOR.lock().unwrap();
    let empty = CollectorHeld::default();
    let collector = collector.as_ref().unwrap_or(&empty);
    let mut planner: Slots = arriving.clone();
    planner.extend(&ready);
    planner.extend(&queued);
    let mut input: Slots = planner.clone();
    input.extend(&collector.held);
    input.extend(&collector.in_flight);
    let count_pinned = |slots: &Slots| {
        slots
            .keys()
            .map(|(slot, _)| *slot)
            .collect::<HashSet<u32>>()
            .iter()
            .filter(|slot| pinned.contains(slot))
            .count()
    };
    // How full the pinned blocks that input batches are in are: the bytes
    // of each that live batches' buffers use.
    let mut used_by_slot: HashMap<u32, usize> = HashMap::new();
    for ((slot, _), bytes) in &input {
        if pinned.contains(slot) {
            *used_by_slot.entry(*slot).or_default() += bytes;
        }
    }
    let used_mb = used_by_slot.values().sum::<usize>() >> 20;
    let under_tenth = used_by_slot.values().filter(|used| **used < BUFFER_SIZE / 10).count();
    let under_half = used_by_slot.values().filter(|used| **used < BUFFER_SIZE / 2).count();
    eprintln!(
        "FILL input_blocks={} used_mb={} blocks_under_tenth_used={} blocks_under_half_used={}",
        used_by_slot.len(),
        used_mb,
        under_tenth,
        under_half,
    );
    eprintln!(
        "MEMPROBE t={} pinned={} arriving_rows={} arriving={} ready_rows={} ready={} queued_rows={} queued={} planner={} collector_rows={} collector={} inflight_groups={} inflight_rows={} inflight={} input_batches={} other={} not_pinned={} heap_mb={}",
        now_ms(),
        pinned.len(),
        arriving_rows,
        count_pinned(&arriving),
        ready_rows,
        count_pinned(&ready),
        queued_rows,
        count_pinned(&queued),
        count_pinned(&planner),
        collector.held_rows,
        count_pinned(&collector.held),
        collector.in_flight_groups,
        collector.in_flight_rows,
        count_pinned(&collector.in_flight),
        count_pinned(&input),
        pinned.len() - count_pinned(&input),
        input.keys().map(|(slot, _)| *slot).collect::<HashSet<u32>>().len() - count_pinned(&input),
        heap_bytes >> 20,
    );
}
''')

edit('mod.rs', [
    ('mod memory_budget;\n', 'mod memory_budget;\nmod probe;\n'),
])

edit('merge/input_file_rows.rs', [
    ('''impl InputFileRows {
''', '''impl InputFileRows {
    /// Machine-only probe: the ring slots of the file's arriving and ready
    /// batches, and how many rows each holds.
    pub(super) fn probe(
        &self,
        arriving: &mut crate::writing::compaction::probe::Slots,
        ready: &mut crate::writing::compaction::probe::Slots,
    ) -> (usize, usize, usize) {
        let (mut arriving_rows, mut ready_rows, mut heap_bytes) = (0, 0, 0);
        for row_group in &self.requested {
            for batch in row_group.early_batches.values() {
                arriving_rows += batch.row_count();
                heap_bytes += super::super::probe::add_slots(&batch.batch, arriving);
            }
        }
        for batch in &self.ready {
            ready_rows += batch.rows.row_count();
            heap_bytes += super::super::probe::add_slots(&batch.rows.batch, ready);
        }
        (arriving_rows, ready_rows, heap_bytes)
    }

'''),
])

edit('merge/planner.rs', [
    ('''        self.plan_and_send_merge_jobs(sender)?;
        Ok(())
    }
''', '''        self.plan_and_send_merge_jobs(sender)?;
        if super::super::probe::is_due(0) {
            let mut arriving = Default::default();
            let mut ready = Default::default();
            let mut queued = Default::default();
            let (mut arriving_rows, mut ready_rows, mut heap_bytes) = (0, 0, 0);
            for file in &self.files {
                let (file_arriving, file_ready, file_heap) = file.probe(&mut arriving, &mut ready);
                arriving_rows += file_arriving;
                ready_rows += file_ready;
                heap_bytes += file_heap;
            }
            let mut queued_rows = 0;
            for job in &self.jobs {
                queued_rows += job.row_count();
                for source in job.sources.iter() {
                    heap_bytes += super::super::probe::add_slots(&source.batch, &mut queued);
                }
            }
            eprintln!(
                "PLANNER next_sequence={} queued_jobs={} rows_sent={} waiting={}",
                self.next_sequence,
                self.jobs.len(),
                self.rows_sent,
                self.memory_budget.has_row_group_waiting(),
            );
            super::super::probe::report(
                arriving,
                arriving_rows,
                ready,
                ready_rows,
                queued,
                queued_rows,
                heap_bytes,
            );
        }
        Ok(())
    }
'''),
])

edit('row_group_collector.rs', [
    ('''    leaf_counts: Vec<usize>,
    file_info: Arc<FileAssemblyInfo>,
}''', '''    leaf_counts: Vec<usize>,
    file_info: Arc<FileAssemblyInfo>,
    /// Machine-only probe: the row groups sent and not written yet, by their
    /// reservation, with the ring slots of their chunks and their rows.
    probe_in_flight: Vec<(
        std::sync::Weak<MemoryReservation>,
        crate::writing::compaction::probe::Slots,
        usize,
    )>,
}'''),
    ('''                file_info: file_info.clone(),
            })
            .collect()''', '''                file_info: file_info.clone(),
                probe_in_flight: Vec::new(),
            })
            .collect()'''),
    ('''            let parts = self.complete.pop_front().unwrap();
            self.send_column_jobs_of_row_group(parts, row_count, memory, sender)?;''',
     '''            let parts = self.complete.pop_front().unwrap();
            let mut slots = Default::default();
            for part in &parts {
                for source in part.chunks.iter() {
                    super::probe::add_slots(&source.batch, &mut slots);
                }
            }
            self.probe_in_flight
                .push((Arc::downgrade(&memory), slots, row_count));
            self.send_column_jobs_of_row_group(parts, row_count, memory, sender)?;'''),
    ('''        self.send_complete_row_groups(sender)?;
        Ok(())
    }

    fn run(''', '''        self.send_complete_row_groups(sender)?;
        if super::probe::is_due(1) {
            let mut held = Default::default();
            let mut held_rows = 0;
            for run in self.arrived.values() {
                held_rows += run.row_count();
                for source in run.sources.iter() {
                    super::probe::add_slots(&source.batch, &mut held);
                }
            }
            for part in self.filling.iter().chain(self.complete.iter().flatten()) {
                held_rows += part.row_count;
                for source in part.chunks.iter() {
                    super::probe::add_slots(&source.batch, &mut held);
                }
            }
            self.probe_in_flight
                .retain(|(reservation, _, _)| reservation.strong_count() > 0);
            let mut in_flight: crate::writing::compaction::probe::Slots = Default::default();
            let mut in_flight_rows = 0;
            for (_, slots, rows) in &self.probe_in_flight {
                in_flight.extend(slots);
                in_flight_rows += rows;
            }
            eprintln!(
                "COLLECTOR next_sequence={} early_runs={} early_rows={} lowest_early={:?} filling_rows={} complete_groups={} row_groups_sent={} waiting={}",
                self.next_sequence,
                self.arrived.len(),
                self.arrived.values().map(|run| run.row_count()).sum::<usize>(),
                self.arrived.keys().next(),
                self.rows_filling,
                self.complete.len(),
                self.row_groups_sent,
                self.memory_budget.has_row_group_waiting(),
            );
            super::probe::set_collector(
                held,
                held_rows,
                in_flight,
                self.probe_in_flight.len(),
                in_flight_rows,
            );
        }
        Ok(())
    }

    fn run('''),
])
print('memory probe added')
