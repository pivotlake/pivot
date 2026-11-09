//! Writing sealed buffers to disk so their ring slots can be given up.
//!
//! An operator's [`PartitionAllocator`] hands a sealed buffer's rows to the
//! [`Spiller`] as one [`SpillWrite`] per partition with rows in it. The
//! spiller writes each to that partition's file and hands back the count of
//! writes in flight, which the allocator keeps with the buffer in a
//! [`SpilledBuffer`] until the count is zero and the ring slot can be released.
//!
//! [`PartitionAllocator`]: partition_allocator::PartitionAllocator

use crate::OperatorIO;
use crate::io::{IORequesterError, LocalFile, OpenFile};
use crate::memory::{FileBytes, WeakSlab, WriteBuffer, memory_ctx};
use crate::worker::WORKER_IDX;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) mod directory;
mod layout;
mod partition_allocator;
mod partition_evictor;
mod partition_tree;

pub use layout::{Dynamic, Fixed, Layout};
pub use partition_allocator::{PartitionAllocator, PartitionContents, PartitionSlab};
pub use partition_evictor::{PartitionEvictor, SharedPartitionEvictor};
pub use partition_tree::PARTITIONS;

/// Spill files are read and written with direct IO, which takes only
/// operations whose file offset, length and memory address are all multiples
/// of the device's block size. 4 KiB is a multiple of every block size in
/// use. So each slab of rows sits at a multiple of this in its buffer and in
/// its file, and is written and read back padded out to the next multiple.
pub const SPILL_BLOCK_SIZE: usize = 4096;

/// The bytes a slab of `size` bytes takes in its buffer and in its file.
pub fn padded_to_spill_block(size: usize) -> usize {
    size.next_multiple_of(SPILL_BLOCK_SIZE)
}

/// One partition's rows from a sealed buffer, going to its file as one write.
pub struct SpillWrite {
    file: LocalFile,
    /// Where in the file the rows go: right after everything the partition
    /// spilled before.
    offset: u64,
    rows: FileBytes,
}

/// A slab of a spilled buffer: where its rows are in memory, for as long as
/// the buffer is resident, and where they are in the partition's file. The
/// handle on the slab is weak: the [`SpilledBuffer`] it belongs to is the
/// buffer's one owner, so that nothing but a reader of the slab keeps the
/// buffer from being evicted, and a reader learns from the handle whether the
/// rows are still in memory.
#[derive(Clone)]
pub struct SpilledSlab {
    pub spill_partition_id: usize,
    /// Where in the partition's file this slab's rows begin.
    pub file_offset: u64,
    pub slab: WeakSlab,
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
}

/// What an operator hands its sealed buffers to when asked to spill. It writes
/// each partition's rows to that partition's file and records every write with
/// the memory context, which counts the writes off as they land.
pub struct Spiller<'a> {
    io: OperatorIO<'a>,
    buffers_spilled: usize,
}

impl<'a> Spiller<'a> {
    pub(crate) fn new(io: OperatorIO<'a>) -> Self {
        Self {
            io,
            buffers_spilled: 0,
        }
    }

    /// Create the file that spilled rows of `spill_partition_id` are appended
    /// to, under this operator's spill directory (see [`directory`]). Each
    /// partition's file is created once; a second call for the same partition
    /// fails. Each write names where in the file it goes, so the file grows
    /// the same way on every platform.
    pub fn open_spill_file(&self, spill_partition_id: usize) -> std::io::Result<LocalFile> {
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
        crate::io::create_direct_read_write(&path)
    }

    pub(crate) fn set_operator_idx(&mut self, operator_idx: crate::Identifier) {
        self.io.set_operator_idx(operator_idx);
    }

    /// Issue one sealed buffer's writes, one per partition, and return the
    /// count of them in flight, which is counted down as they land.
    pub fn spill(&mut self, writes: Vec<SpillWrite>) -> Result<Arc<AtomicUsize>, IORequesterError> {
        let writes_in_flight = Arc::new(AtomicUsize::new(writes.len()));
        for SpillWrite { file, offset, rows } in writes {
            let write_id = self
                .io
                .write_spill(OpenFile::Local(file), offset, Arc::new(rows))?;
            memory_ctx().record_spill_write_issued(
                self.io.data_flow_id(),
                write_id,
                writes_in_flight.clone(),
            );
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
    use crate::io::IORequester;
    use crate::memory::{BUFFER_SIZE, FrozenSlab, Slab, init_test_free_pool};
    use crate::stats::StatsCollector;
    use std::cell::RefCell;

    /// Wait for every one of the buffer's spill writes to land, counting each
    /// off `buffer_writes_in_flight` as the worker does. Waiting for them one
    /// at a time is not possible: two writes can land in one wait, and a wait
    /// with nothing left in flight never returns.
    fn complete_spill_writes(
        requester: &RefCell<IORequester>,
        buffer_writes_in_flight: &AtomicUsize,
    ) {
        let mut requester = requester.borrow_mut();
        while buffer_writes_in_flight.load(Ordering::Acquire) > 0 {
            requester.wait().unwrap();
            requester.collect_completions().unwrap();
            memory_ctx().complete_landed_spill_writes(&mut requester);
        }
    }

    #[test]
    fn a_spilled_buffer_is_on_disk_once_each_partitions_write_has_landed() {
        init_test_free_pool(1);
        let (dataflow, operator) = (9_000_004, 2);
        let requester = RefCell::new(IORequester::default());
        let mut stats = StatsCollector::disabled();
        let mut next_request_id = 0;
        let io = OperatorIO::new(
            &requester,
            dataflow,
            operator,
            &mut stats,
            &mut next_request_id,
        );
        let mut spiller = Spiller::new(io);
        // Three one-block slabs of one buffer: two for partition 5, one for
        // partition 9.
        let mut buffer = memory_ctx().get_write_buffer(false);
        buffer.as_mut_slice()[..3 * SPILL_BLOCK_SIZE].fill(0xAB);
        let buffer = Arc::new(buffer);
        let slab_at = |offset: usize| {
            Slab::new(
                unsafe { buffer.ptr.add(offset) },
                SPILL_BLOCK_SIZE,
                buffer.clone(),
            )
            .freeze()
        };
        let rows_of = |slabs: &[&FrozenSlab]| {
            let mut rows = FileBytes::new();
            slabs
                .iter()
                .for_each(|slab| rows.push_frozen_slab((*slab).clone()));
            rows
        };
        let (first, second, third) = (
            slab_at(0),
            slab_at(SPILL_BLOCK_SIZE),
            slab_at(2 * SPILL_BLOCK_SIZE),
        );
        let writes = vec![
            SpillWrite {
                file: spiller.open_spill_file(5).unwrap(),
                offset: 0,
                rows: rows_of(&[&first, &third]),
            },
            SpillWrite {
                file: spiller.open_spill_file(9).unwrap(),
                offset: 0,
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
                    slab: slab.downgrade(),
                })
                .collect(),
        };
        let in_flight_after_issuing = spilled.writes_in_flight.load(Ordering::Acquire);
        drop(spiller);
        complete_spill_writes(&requester, &spilled.writes_in_flight);

        assert_eq!(in_flight_after_issuing, 2);
        assert!(!memory_ctx().has_spill_writes_in_flight());
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
        assert_eq!(file_len(5), 2 * SPILL_BLOCK_SIZE as u64);
        assert_eq!(file_len(9), SPILL_BLOCK_SIZE as u64);
        // The test's handle and the spilled buffer's: the writes let go of
        // theirs when they landed, and the slab records hold none.
        assert_eq!(Arc::strong_count(&spilled.buffer), 1 + 1);
        assert!(spilled.slabs.iter().all(|s| {
            s.slab
                .upgrade()
                .unwrap()
                .as_slice()
                .iter()
                .all(|&b| b == 0xAB)
        }));
        assert!(BUFFER_SIZE >= 3 * SPILL_BLOCK_SIZE);
        directory::remove_worker_directory(dataflow, WORKER_IDX.get()).unwrap();
    }

    #[test]
    fn a_spill_file_is_created_once_under_its_operators_directory() {
        WORKER_IDX.set(0);
        let (dataflow, operator) = (9_000_003, 4);
        let requester = RefCell::new(IORequester::default());
        let mut stats = StatsCollector::disabled();
        let mut next_request_id = 0;
        let io = OperatorIO::new(
            &requester,
            dataflow,
            operator,
            &mut stats,
            &mut next_request_id,
        );
        let spiller = Spiller::new(io);

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
