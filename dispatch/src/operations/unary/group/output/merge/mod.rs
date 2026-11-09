//! Partition-parallel merge of GROUP BY partial results.
//!
//! High hash bits select both the merge partition and the source-table slot
//! range. Each merge job therefore reads only its portion of every source:
//!
//! ```text
//! worker tables ----+
//!                   +--> partition target --> final node merge
//! scatter buffers --+
//! ```
//!
//! A partition's merge runs for as long as its rows take to arrive: the rows
//! in memory, scatter rows ([`scatter_rows`]) and in-place tables
//! ([`tables`]), go into the result when the merge starts, and rows that were
//! spilled and evicted go in as each read brings them back ([`read_rows`]).
//! On a multi-node machine the per-node results then meet in
//! [`node_tables`].

mod node_tables;
mod read_rows;
mod scatter_rows;
mod tables;
#[cfg(test)]
mod tests;

pub(super) use node_tables::merge_node_aggregated_tables;
use read_rows::merge_read_rows;
use scatter_rows::merge_scatter_rows;
use tables::merge_tables;

use crate::io::{OperatorIO, ReadData, ReadRequestId, SlabRead};
use crate::memory::{SPILL_BLOCK_SIZE, SlabAllocator, decompress_spilled_rows};
use crate::operations::unary;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, HashTable, KeyExtractor, OnDiskStretch,
};
use crate::operations::unary::group::output::partition_job::PartitionJob;
use bytes::Bytes;

/// A partition job this worker is running: its rows in memory are merged, and
/// the rows of its partition that are only on disk are merged as their reads
/// land.
pub(super) struct RunningPartitionMerge<K: KeyExtractor, V: AggregationValue + ?Sized> {
    pub(super) job: PartitionJob<K, V>,
    /// The reads still in flight, each with the stretches of a spill file it
    /// is bringing back. A read's rows are merged into `table` as it lands.
    /// A plain list: a job reads only the partition's spilled rows, a
    /// handful of reads at most, and a hash map's seeding alone would cost
    /// every job a write to a counter shared by every worker.
    pub(super) reads_in_flight: Vec<(ReadRequestId, Vec<OnDiskStretch>)>,
    /// Where the rows read back from disk land, if the job reads any. Apart
    /// from `allocator` because these rows are dropped as soon as they are
    /// merged, long before the table is.
    read_allocator: Option<SlabAllocator>,
    /// The partition's result so far. The rows in memory go in when the job
    /// starts, the rows read back from disk as each read lands. The table
    /// grows out of `allocator`, and `capacity` is its slot count as it does.
    allocator: SlabAllocator,
    pub(super) table: HashTable<K::Persisted, V>,
    capacity: usize,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> RunningPartitionMerge<K, V> {
    /// Start `job`: issue the reads for the rows of its partition that are
    /// only on disk, one per source worker's bucket that has any, then merge
    /// the rows that are in memory while those reads run.
    pub(super) fn start(job: PartitionJob<K, V>, io: &mut OperatorIO) -> unary::Result<Self> {
        // Borrowed, not cloned: the state is shared by every job of the
        // dataflow, and a clone per job is a write to its count from every
        // merging worker at once.
        let shared = &*job.shared;
        let (buffers, tables) = &shared.sources[job.node];
        let mut reads_in_flight = Vec::new();
        let mut read_allocator: Option<SlabAllocator> = None;
        for worker_buffers in buffers {
            for bucket in worker_buffers.bucket_range(job.index, shared.num_partitions) {
                worker_buffers.resolve_spilled(bucket);
                let Some(on_disk) =
                    worker_buffers.with_bucket(bucket, |rows, _| rows.on_disk().cloned())
                else {
                    continue;
                };
                let allocator = read_allocator.get_or_insert_with(|| SlabAllocator::new(false));
                let reads = on_disk
                    .stretches
                    .iter()
                    .map(|stretch| SlabRead {
                        offset: stretch.range.offset,
                        slab: allocator.get_aligned_slab(
                            stretch.range.len,
                            SPILL_BLOCK_SIZE,
                            false,
                        ),
                    })
                    .collect();
                let read = io.read_direct(on_disk.file, reads)?;
                reads_in_flight.push((read, on_disk.stretches));
            }
        }

        let partition_bits = shared.num_partitions.trailing_zeros();
        let mut allocator = SlabAllocator::new(true);
        let mut capacity = shared.partition_capacity.max(DEFAULT_CAPACITY);
        let mut table = HashTable::new(
            &mut allocator,
            capacity,
            partition_bits,
            &shared.shared_context,
        );
        let cutoff = job.cutoff();
        merge_scatter_rows::<K::Stored, V>(
            &mut table,
            &mut allocator,
            &mut capacity,
            job.index,
            buffers,
            shared.num_partitions,
            &shared.key_arena,
            &shared.shared_context,
            cutoff,
        );
        merge_tables::<K::Stored, V>(
            &mut table,
            &mut allocator,
            job.index,
            partition_bits,
            tables,
            &shared.key_arena,
            &shared.shared_context,
            cutoff,
        );
        Ok(Self {
            job,
            reads_in_flight,
            read_allocator,
            allocator,
            table,
            capacity,
        })
    }

    /// Merge the rows `read`, one of this job's reads in flight, brought back
    /// from disk as `stretches_read`, one per stretch it asked for.
    pub(super) fn merge_read(&mut self, read: ReadRequestId, stretches_read: Vec<ReadData>) {
        let position = self
            .reads_in_flight
            .iter()
            .position(|(id, _)| *id == read)
            .expect("the read is one of this job's reads in flight");
        let (_, stretches) = self.reads_in_flight.swap_remove(position);
        let read_allocator = self
            .read_allocator
            .as_mut()
            .expect("a job with a read in flight has the allocator its reads landed in");
        let rows: Vec<ReadData> = stretches
            .iter()
            .zip(stretches_read)
            .map(|(stretch, read_back)| {
                let [bytes] = read_back.bytes.as_slice() else {
                    panic!("a stretch read around the cache comes back in one piece");
                };
                let slabs = stretch
                    .cut_stored_slabs(bytes)
                    .map(|(stored, row_bytes)| {
                        let mut rows =
                            read_allocator.get_aligned_slab(row_bytes, SPILL_BLOCK_SIZE, false);
                        decompress_spilled_rows(&stored, rows.as_mut_slice());
                        Bytes::from_owner(rows.freeze())
                    })
                    .collect();
                ReadData {
                    offset: read_back.offset,
                    bytes: slabs,
                }
            })
            .collect();
        let rows = rows.as_slice();
        let shared = &*self.job.shared;
        merge_read_rows::<K::Stored, V>(
            &mut self.table,
            &mut self.allocator,
            &mut self.capacity,
            rows,
            &shared.key_arena,
            &shared.shared_context,
            self.job.cutoff(),
        );
    }

    /// Memory this job holds, in whole ring buffers: its result table, and
    /// the rows it reads back from disk, which are released as they are
    /// merged.
    pub(super) fn memory_bytes(&self) -> usize {
        self.allocator.bytes_held()
            + self
                .read_allocator
                .as_ref()
                .map_or(0, SlabAllocator::bytes_held)
    }

    /// Take a finished merge apart into its job and its result table, letting
    /// go of everything else first. That order matters for the free pool,
    /// which hands out the buffer freed last: the allocator's working buffer
    /// is one the table never reached, untouched since it was popped, while
    /// the table's own buffers are the ones this worker just wrote. Freed
    /// first, the working buffer sits under them, and the next job's table is
    /// built in the buffers still in this core's cache.
    pub(super) fn into_job_and_table(self) -> (PartitionJob<K, V>, HashTable<K::Persisted, V>) {
        let Self {
            job,
            table,
            allocator,
            reads_in_flight,
            read_allocator,
            capacity: _,
        } = self;
        debug_assert!(
            reads_in_flight.is_empty(),
            "a merge is taken apart only once every read of its rows has landed"
        );
        drop(read_allocator);
        drop(allocator);
        (job, table)
    }
}
