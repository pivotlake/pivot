//! Writing sealed buffers to disk so their ring slots can be given up.
//!
//! An operator's [`PartitionAllocator`] hands a sealed buffer's rows to the
//! [`Spiller`] as one [`SpillWrite`] per partition with rows in it. The
//! spiller writes each to that partition's file and hands back the count of
//! writes in flight, which the allocator keeps with the buffer in a
//! [`SpilledBuffer`] until the count is zero and the ring slot can be released.
//!
//! [`PartitionAllocator`]: partition_allocator::PartitionAllocator

use crate::io::{IORequesterError, LocalFile, OpenFile};
use crate::memory::{FileBytes, FrozenSlab, WriteBuffer, memory_ctx};
use crate::worker::WORKER_IDX;
use crate::{OperatorIO, WriteRequestId};
use ahash::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) mod directory;
mod layout;
mod partition_allocator;
mod partition_evictor;
mod partition_tree;

pub use layout::{Dynamic, Fixed, Layout};
pub use partition_allocator::{PartitionAllocator, PartitionContents, PartitionSlab};
pub use partition_evictor::{PartitionEvictor, SharedPartitionEvictor, TakenSpills};
pub use partition_tree::PARTITIONS;

/// One partition's rows from a sealed buffer, going to its file as one write.
pub struct SpillWrite {
    file: OpenFile,
    rows: FileBytes,
}

/// A slab of a spilled buffer: where its rows are in memory, for as long as
/// the buffer is resident, and where they are in the partition's file.
pub struct SpilledSlab {
    pub spill_partition_id: usize,
    /// Where in the partition's file this slab's rows begin.
    pub file_offset: u64,
    pub slab: FrozenSlab,
}

/// A buffer whose rows are being, or have been, written to disk. It stays
/// resident, and its rows readable through `slabs`, until eviction takes the
/// ring slot, which may happen once `writes_in_flight` reaches zero.
pub struct SpilledBuffer {
    /// Counted down by the worker as each of the buffer's writes lands.
    writes_in_flight: Arc<AtomicUsize>,
    buffer: Arc<WriteBuffer>,
    slabs: Vec<SpilledSlab>,
    /// The tree node the buffer was allocated from, which is where the
    /// evictor keeps it.
    partition_group_slot: usize,
}

impl SpilledBuffer {
    /// Whether every write of this buffer has landed.
    pub fn is_on_disk(&self) -> bool {
        self.writes_in_flight.load(Ordering::Acquire) == 0
    }

    /// The buffer's slabs, still readable: the buffer stays resident for as
    /// long as any of them is held.
    pub fn into_slabs(self) -> Vec<SpilledSlab> {
        self.slabs
    }
}

/// What an operator hands its sealed buffers to when asked to spill. It writes
/// each partition's rows to that partition's file and records every write with
/// the worker, which counts the writes off as they land.
pub struct Spiller<'a> {
    io: OperatorIO<'a>,
    /// Every spill write in flight on this worker, by its id, pointing at the
    /// in-flight count of the buffer it belongs to.
    writes_in_flight: &'a mut HashMap<WriteRequestId, Arc<AtomicUsize>>,
    buffers_spilled: usize,
}

impl<'a> Spiller<'a> {
    pub(crate) fn new(
        io: OperatorIO<'a>,
        writes_in_flight: &'a mut HashMap<WriteRequestId, Arc<AtomicUsize>>,
    ) -> Self {
        Self {
            io,
            writes_in_flight,
            buffers_spilled: 0,
        }
    }

    /// Create the file that spilled rows of `spill_partition_id` are appended
    /// to, under this operator's spill directory (see [`directory`]). Each
    /// partition's file is created once; a second call for the same partition
    /// fails. Opened for appending, so every write goes to the end whatever
    /// offset it names.
    pub fn open_spill_file(&self, spill_partition_id: usize) -> std::io::Result<OpenFile> {
        let path = directory::partition_file_path(
            self.io.data_flow_id(),
            WORKER_IDX.get(),
            self.io.operator_idx(),
            spill_partition_id,
        );
        std::fs::create_dir_all(
            path.parent()
                .expect("a partition file lies in its operator's directory"),
        )?;
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create_new(true)
            .open(&path)?;
        Ok(OpenFile::Local(LocalFile::new(file)?))
    }

    pub(crate) fn set_operator_idx(&mut self, operator_idx: crate::Identifier) {
        self.io.set_operator_idx(operator_idx);
    }

    /// Issue one sealed buffer's writes, one per partition, and return the
    /// count of them in flight, which the worker counts down as they land.
    pub fn spill(&mut self, writes: Vec<SpillWrite>) -> Result<Arc<AtomicUsize>, IORequesterError> {
        let writes_in_flight = Arc::new(AtomicUsize::new(writes.len()));
        for SpillWrite { file, rows } in writes {
            let write_id = self.io.write_spill(file, Arc::new(rows))?;
            self.writes_in_flight
                .insert(write_id, writes_in_flight.clone());
        }
        memory_ctx().record_buffers_spilled(1);
        self.buffers_spilled += 1;
        Ok(writes_in_flight)
    }

    /// How many buffers this spiller has written out so far.
    pub fn buffers_spilled(&self) -> usize {
        self.buffers_spilled
    }

    /// Count `count` buffers as spilled without writing anything, so a test
    /// can drive the walk over a dataflow's operators without a ring.
    #[cfg(test)]
    pub(crate) fn count_buffers_spilled_without_writing(&mut self, count: usize) {
        self.buffers_spilled += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{Completion, IORequester};
    use crate::memory::{BUFFER_SIZE, Slab, init_test_free_pool};
    use crate::stats::StatsCollector;
    use std::sync::atomic::Ordering;

    /// Take the next completed spill write off the requester and count it off
    /// its buffer, as the worker does.
    fn complete_one_spill_write(
        requester: &mut IORequester,
        writes_in_flight: &mut HashMap<WriteRequestId, Arc<AtomicUsize>>,
    ) {
        loop {
            requester.wait().unwrap();
            for completion in requester.completions().unwrap() {
                if let Ok(Completion::FsWrite(write)) = completion {
                    let count = writes_in_flight.remove(&write.request.id).unwrap();
                    count.fetch_sub(1, Ordering::Release);
                    return;
                }
            }
        }
    }

    #[test]
    fn a_spilled_buffer_is_on_disk_once_each_partitions_write_has_landed() {
        init_test_free_pool(1);
        let (dataflow, operator) = (9_000_004, 2);
        let mut requester = IORequester::default();
        let mut stats = StatsCollector::disabled();
        let mut next_request_id = 0;
        let mut writes_in_flight = HashMap::default();
        let io = OperatorIO::new(
            &mut requester,
            dataflow,
            operator,
            &mut stats,
            &mut next_request_id,
        );
        let mut spiller = Spiller::new(io, &mut writes_in_flight);
        // Three 1KB slabs of one buffer: two for partition 5, one for partition 9.
        let mut buffer = memory_ctx().get_write_buffer(false);
        buffer.as_mut_slice()[..3072].fill(0xAB);
        let buffer = Arc::new(buffer);
        let slab_at = |offset: usize| {
            Slab::new(unsafe { buffer.ptr.add(offset) }, 1024, buffer.clone()).freeze()
        };
        let rows_of = |slabs: &[&FrozenSlab]| {
            let mut rows = FileBytes::new();
            slabs
                .iter()
                .for_each(|slab| rows.push_frozen_slab((*slab).clone()));
            rows
        };
        let (first, second, third) = (slab_at(0), slab_at(1024), slab_at(2048));
        let writes = vec![
            SpillWrite {
                file: spiller.open_spill_file(5).unwrap(),
                rows: rows_of(&[&first, &third]),
            },
            SpillWrite {
                file: spiller.open_spill_file(9).unwrap(),
                rows: rows_of(&[&second]),
            },
        ];

        let buffer_writes_in_flight = spiller.spill(writes).unwrap();
        let spilled = SpilledBuffer {
            writes_in_flight: buffer_writes_in_flight,
            buffer: buffer.clone(),
            partition_group_slot: 0,
            slabs: vec![first, second, third]
                .into_iter()
                .map(|slab| SpilledSlab {
                    spill_partition_id: 0,
                    file_offset: 0,
                    slab,
                })
                .collect(),
        };
        let in_flight_after_issuing = spilled.writes_in_flight.load(Ordering::Acquire);
        drop(spiller);
        complete_one_spill_write(&mut requester, &mut writes_in_flight);
        let in_flight_after_one = spilled.writes_in_flight.load(Ordering::Acquire);
        complete_one_spill_write(&mut requester, &mut writes_in_flight);
        let in_flight_after_both = spilled.writes_in_flight.load(Ordering::Acquire);

        assert_eq!(in_flight_after_issuing, 2);
        assert_eq!(in_flight_after_one, 1);
        assert_eq!(in_flight_after_both, 0);
        assert!(writes_in_flight.is_empty());
        let file_len = |partition| {
            std::fs::metadata(directory::partition_file_path(
                dataflow,
                WORKER_IDX.get(),
                operator,
                partition,
            ))
            .unwrap()
            .len()
        };
        assert_eq!(file_len(5), 2048);
        assert_eq!(file_len(9), 1024);
        // The test's handle, the spilled buffer's, and one per slab it kept.
        assert_eq!(Arc::strong_count(&spilled.buffer), 1 + 1 + 3);
        assert!(
            spilled
                .slabs
                .iter()
                .all(|s| s.slab.as_slice().iter().all(|&b| b == 0xAB))
        );
        assert!(BUFFER_SIZE >= 3072);
        directory::remove_worker_directory(dataflow, WORKER_IDX.get()).unwrap();
    }

    #[test]
    fn a_spill_file_is_created_once_under_its_operators_directory() {
        WORKER_IDX.set(0);
        let (dataflow, operator) = (9_000_003, 4);
        let mut requester = IORequester::default();
        let mut stats = StatsCollector::disabled();
        let mut next_request_id = 0;
        let io = OperatorIO::new(
            &mut requester,
            dataflow,
            operator,
            &mut stats,
            &mut next_request_id,
        );
        let mut writing = HashMap::default();
        let spiller = Spiller::new(io, &mut writing);

        let opened = spiller.open_spill_file(7).is_ok();
        let again = spiller.open_spill_file(7).err().map(|error| error.kind());
        let path = directory::partition_file_path(dataflow, 0, operator, 7);
        let created = path.exists();
        directory::remove_operator_directory(dataflow, 0, operator).unwrap();

        assert!(opened);
        assert_eq!(again, Some(std::io::ErrorKind::AlreadyExists));
        assert!(created);
        assert!(!path.exists());
    }
}
