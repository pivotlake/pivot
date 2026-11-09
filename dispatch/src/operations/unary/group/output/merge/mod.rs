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

use crate::io::{OperatorIO, ReadData, ReadRequestId};
use crate::memory::SlabAllocator;
use crate::operations::unary;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, HashTable, KeyExtractor,
};
use crate::operations::unary::group::output::partition_job::PartitionJob;

/// A partition job this worker is running: its rows in memory are merged, and
/// the rows of its partition that are only on disk are merged as their reads
/// land.
pub(super) struct RunningPartitionMerge<K: KeyExtractor, V: AggregationValue + ?Sized> {
    pub(super) job: PartitionJob<K, V>,
    /// The reads still in flight, each with the bytes it brings. A read's
    /// rows are merged into `table` as it lands. A plain list: a job reads
    /// only the partition's spilled rows, a handful of ranges at most, and
    /// a hash map's seeding alone would cost every job a write to a counter
    /// shared by every worker.
    pub(super) reads_in_flight: Vec<(ReadRequestId, usize)>,
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
        let shared = job.shared.clone();
        let (buffers, tables) = &shared.sources[job.node];
        let mut reads_in_flight = Vec::new();
        for worker_buffers in buffers {
            for bucket in worker_buffers.bucket_range(job.index, shared.num_partitions) {
                worker_buffers.resolve_spilled(bucket);
                let Some(on_disk) =
                    worker_buffers.with_bucket(bucket, |rows, _| rows.on_disk().cloned())
                else {
                    continue;
                };
                let bytes = on_disk.ranges.iter().map(|range| range.len).sum::<usize>();
                let read = io.read(on_disk.file, on_disk.ranges)?;
                reads_in_flight.push((read, bytes));
            }
        }

        let partition_bits = shared.num_partitions.trailing_zeros();
        let mut allocator = SlabAllocator::new(true);
        let capacity = shared.partition_capacity.max(DEFAULT_CAPACITY);
        let table = HashTable::new(
            &mut allocator,
            capacity,
            partition_bits,
            &shared.shared_context,
        );
        let mut running = Self {
            job,
            reads_in_flight,
            allocator,
            table,
            capacity,
        };
        merge_scatter_rows::<K::Stored, V>(
            &mut running.table,
            &mut running.allocator,
            &mut running.capacity,
            running.job.index,
            buffers,
            shared.num_partitions,
            &shared.key_arena,
            &shared.shared_context,
            running.job.cutoff(),
        );
        merge_tables::<K::Stored, V>(
            &mut running.table,
            &mut running.allocator,
            running.job.index,
            partition_bits,
            tables,
            &shared.key_arena,
            &shared.shared_context,
            running.job.cutoff(),
        );
        Ok(running)
    }

    /// Merge the rows one read brought back from disk.
    pub(super) fn merge_read_rows(&mut self, rows: &[ReadData]) {
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

    /// Memory this job holds: its result table, in whole ring buffers, and
    /// the bytes of its reads still in flight, which land in cache slots and
    /// are released as soon as they are merged.
    pub(super) fn memory_bytes(&self) -> usize {
        self.allocator.bytes_held()
            + self
                .reads_in_flight
                .iter()
                .map(|(_, bytes)| bytes)
                .sum::<usize>()
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
            capacity: _,
        } = self;
        debug_assert!(
            reads_in_flight.is_empty(),
            "a merge is taken apart only once every read of its rows has landed"
        );
        drop(allocator);
        (job, table)
    }
}
