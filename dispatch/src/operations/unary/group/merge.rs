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
//! Tables with equal capacity are scanned together. Linear-probe entries that
//! crossed the nominal partition boundary are handled by a short overflow scan.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::PartitionBuffers;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, MAX_LOAD_FACTOR, MultiSlabTable, PersistedKey, Prober,
    StridedScatterRows,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::values::ArityBody;

/// Tables grouped by the slot count they were built at. Equal-sized tables share
/// partition boundaries, so the merge scans them together.
type TablesByCapacity<'a, S, V> = Vec<(
    usize,
    Vec<&'a MultiSlabTable<<S as StoredKey>::Persisted, V>>,
)>;

/// Receives each merged table of a partition as soon as it is complete.
type MergedTableSink<'a, S, V> = &'a mut dyn FnMut(&MultiSlabTable<<S as StoredKey>::Persisted, V>);

/// Collision-to-entry ratio at which we double the target table.
///
/// For linear probing, the cumulative collision ratio is
/// `alpha / (2 * (1 - alpha))`. A threshold of two allows dense result tables
/// without letting probe chains grow without bound.
const RESIZE_COLLISION_RATIO: f64 = 2.0;

/// Slots processed from each equal-sized source before moving to the next.
const SCAN_BATCH_SIZE: usize = 100;

const PREFETCH_DISTANCE: usize = 8;
/// Scatter rows read ahead of the row being merged, for prefetching.
const SCATTER_PREFETCH_AHEAD: usize = 16;

/// Try to resize the target if cumulative collision pressure is too high.
#[inline]
fn resize_if_needed<KP: PersistedKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    target: &mut Prober<'_, KP, V>,
) {
    // Use the integer form of `collisions / len > ratio`.
    target.resize_on_collisions(allocator, RESIZE_COLLISION_RATIO as usize);
}

/// Scan the expected slot range `[start, end)` across all source `tables`,
/// merging entries that belong to `partition` into `target`.
///
/// Equal-sized source tables are scanned in small batches so their target
/// region remains local.
#[allow(clippy::too_many_arguments)]
fn merge_within_partition_bounds<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&MultiSlabTable<S::Persisted, V>],
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
) {
    let partition_shift = u64::BITS - partition_bits;
    let start = (partition * slot_count) >> partition_bits;
    let end = ((partition + 1) * slot_count) >> partition_bits;
    let mut i = start;

    while i < end {
        let range = std::cmp::min(SCAN_BATCH_SIZE, end - i);

        for table in tables {
            // Reuse one source reader for this range.
            let source_reader = table.reader::<N>();
            for j in i..i + range {
                if j + PREFETCH_DISTANCE < slot_count {
                    // Prefetching hash zero is harmless and avoids a branch.
                    target.prefetch(source_reader.hash_at(j + PREFETCH_DISTANCE));
                }
                let entry = source_reader.view_at(j);
                let hash = entry.hash;
                // A non-short-circuiting AND combines both data-dependent tests.
                let take = (hash != 0) & ((hash >> partition_shift) as usize == partition);
                if take {
                    target.merge_from::<true, _>(
                        hash,
                        S::resolve_persisted(arena, *entry.key),
                        entry.stored,
                        context,
                    );
                    resize_if_needed::<S::Persisted, V>(allocator, target);
                }
            }
        }

        i += range;
    }
}

/// Scans the probe-chain overflow beyond a partition's nominal slot range.
#[allow(clippy::too_many_arguments)]
fn merge_past_partition_bounds<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&MultiSlabTable<S::Persisted, V>],
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
) {
    let partition_shift = u64::BITS - partition_bits;
    let end = ((partition + 1) * slot_count) >> partition_bits;
    let mask = slot_count - 1;

    for table in tables {
        let source_reader = table.reader::<N>();
        let mut i = end & mask;
        loop {
            let entry = source_reader.view_at(i);
            let hash = entry.hash;
            if hash == 0 {
                break;
            }
            if (hash >> partition_shift) as usize == partition {
                target.merge_from::<true, _>(
                    hash,
                    S::resolve_persisted(arena, *entry.key),
                    entry.stored,
                    context,
                );
                resize_if_needed::<S::Persisted, V>(allocator, target);
            }
            i = (i + 1) & mask;
        }
    }
}

/// Merge entries from `tables` that belong to `partition` into `target`.
#[allow(clippy::too_many_arguments)]
fn merge_into_partition<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: Vec<&MultiSlabTable<S::Persisted, V>>,
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
) {
    merge_within_partition_bounds::<N, S, V>(
        allocator,
        arena,
        partition,
        slot_count,
        &tables,
        target,
        partition_bits,
        context,
    );
    merge_past_partition_bounds::<N, S, V>(
        allocator,
        arena,
        partition,
        slot_count,
        &tables,
        target,
        partition_bits,
        context,
    );
}

/// Merges one partition's scatter rows and in-place tables.
///
/// Both sources use the same high hash bits, so no preliminary repartitioning
/// is required.
pub(super) fn merge_combined<S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[MultiSlabTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> MultiSlabTable<S::Persisted, V> {
    // Dispatch once so every loop in this job shares the specialized arity.
    V::dispatch_arity(
        V::storage_metadata(context),
        MergeCombined::<S, V> {
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
        },
    )
}

/// State passed through arity dispatch for one partition merge.
struct MergeCombined<'a, S: StoredKey, V: AggregationValue + ?Sized> {
    partition: usize,
    buffers: &'a [PartitionBuffers<S::Persisted, V>],
    tables: &'a [MultiSlabTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &'a SharedArena,
    context: &'a V::SharedContext,
}

impl<S: StoredKey, V: AggregationValue + ?Sized> ArityBody<MultiSlabTable<S::Persisted, V>>
    for MergeCombined<'_, S, V>
{
    #[inline(always)]
    fn run<const N: usize>(self) -> MultiSlabTable<S::Persisted, V> {
        let MergeCombined {
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
        } = self;
        merge_combined_rows::<N, S, V>(
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
        )
    }
}

/// Merges one partition after arity dispatch.
///
/// Separate reference parameters preserve alias information across raw target
/// writes, keeping shared state outside the inner loops.
#[allow(clippy::too_many_arguments)]
fn merge_combined_rows<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[MultiSlabTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> MultiSlabTable<S::Persisted, V> {
    {
        let partition_bits = num_partitions.trailing_zeros();
        let mut allocator = SlabAllocator::new(true);
        let mut capacity = partition_capacity.max(DEFAULT_CAPACITY);
        let mut result: MultiSlabTable<S::Persisted, V> = <MultiSlabTable<S::Persisted, V>>::new(
            &mut allocator,
            capacity,
            partition_bits,
            context,
        );
        // Reuse one target probe layout across the partition.
        let mut target = if N == 0 {
            result.prober()
        } else {
            result.prober_with_metadata(V::metadata_for_arity::<N>())
        };

        // Scatter rows were not aggregated during consume, so each is inserted
        // once here.
        // Prefetch strings unconditionally and fixed-width keys only when the
        // target is large enough for probes to be cold.
        const TARGET_PREFETCH_MIN_SLOTS: usize = 16384;
        let merge_prefetch =
            <S::Persisted as PersistedKey>::HAS_BLOB || capacity > TARGET_PREFETCH_MIN_SLOTS;
        // A merge partition may own several consecutive scatter buckets. Both
        // counts are powers of two, so the ranges divide evenly.
        let scatter_bucket_count = buffers
            .first()
            .map_or(num_partitions, |buffer| buffer.0.len());
        debug_assert!(
            scatter_bucket_count.is_multiple_of(num_partitions),
            "merge partitions ({num_partitions}) must evenly divide scatter buckets ({scatter_bucket_count})"
        );
        let buckets_per_partition = (scatter_bucket_count / num_partitions).max(1);
        let first_bucket = partition * buckets_per_partition;
        let end_bucket = first_bucket + buckets_per_partition;
        // All scatter buckets for this signature share one row layout.
        let scatter_layout = StridedScatterRows::<S::Persisted, V>::layout::<N>(context);
        for worker_buffers in buffers {
            for bucket in first_bucket..end_bucket {
                if merge_prefetch {
                    worker_buffers.0[bucket].for_each_prefetched::<SCATTER_PREFETCH_AHEAD>(
                        scatter_layout,
                        |hash, key, stored, ahead| {
                            if let Some((ahead_hash, ahead_key)) = ahead {
                                ahead_key.prefetch_blob(key_arena);
                                target.prefetch(ahead_hash);
                            }
                            target.grow_if_full(&mut allocator, &mut capacity);
                            let live_key = S::resolve_persisted(key_arena, *key);
                            target.merge_from::<false, _>(hash, live_key, stored, context);
                        },
                    );
                } else {
                    worker_buffers.0[bucket].for_each(scatter_layout, |hash, key, stored| {
                        target.grow_if_full(&mut allocator, &mut capacity);
                        let live_key = S::resolve_persisted(key_arena, *key);
                        target.merge_from::<false, _>(hash, live_key, stored, context);
                    });
                }
            }
        }

        // Scan equal-sized tables together so they share partition boundaries.
        let mut tables_by_capacity: TablesByCapacity<'_, S, V> = Vec::new();
        for table in tables {
            let slot_count = table.capacity();
            if let Some((_, same_size_tables)) = tables_by_capacity
                .iter_mut()
                .find(|(group_slot_count, _)| *group_slot_count == slot_count)
            {
                same_size_tables.push(table);
            } else {
                tables_by_capacity.push((slot_count, vec![table]));
            }
        }
        for (slot_count, same_size_tables) in tables_by_capacity {
            merge_into_partition::<N, S, V>(
                &mut allocator,
                key_arena,
                partition,
                slot_count,
                same_size_tables,
                &mut target,
                partition_bits,
                context,
            );
        }
        result
    }
}
/// Largest merge target the sub-partition path lets one probe loop touch.
///
/// A merge probe stalls on the target entry it loads, so the target has to
/// stay in the core's first-level cache. A partition whose target would be
/// larger is merged in sub-partitions, each into a reused table of at most
/// this size.
const SUB_PARTITION_TARGET_BYTES: usize = 32 * 1024;

/// Upper bound on sub-partitions per merge job. Every sub-partition costs a
/// table clear and an output scan, so a partition is not split finer than
/// this even when its target is very large.
const MAX_SUB_PARTITIONS: usize = 64;

/// Largest merge target that still sits in a core's second-level cache.
///
/// Splitting a partition costs one copy of every merged row. Against a
/// target that misses the second-level cache the copy always pays for
/// itself; against one that fits, it only does when most rows insert a
/// group, since a merge that mostly updates existing groups already reuses
/// the entries it touches.
const SECOND_LEVEL_TARGET_BYTES: usize = 1024 * 1024;

/// Rows per group above which a merge counts as update-dominated.
const UPDATE_DOMINATED_ROWS_PER_GROUP: usize = 2;

/// Number of sub-partitions a merge job with this target splits into, or 1
/// when the target already fits [`SUB_PARTITION_TARGET_BYTES`] or the split
/// would not pay for its row copy.
pub(super) fn sub_partition_count(
    partition_capacity: usize,
    entry_stride: usize,
    merged_rows: usize,
    estimated_groups: usize,
) -> usize {
    let target_bytes = partition_capacity * entry_stride;
    if target_bytes <= SUB_PARTITION_TARGET_BYTES {
        return 1;
    }
    let update_dominated = merged_rows > UPDATE_DOMINATED_ROWS_PER_GROUP * estimated_groups.max(1);
    if target_bytes <= SECOND_LEVEL_TARGET_BYTES && update_dominated {
        return 1;
    }
    target_bytes
        .div_ceil(SUB_PARTITION_TARGET_BYTES)
        .next_power_of_two()
        .min(MAX_SUB_PARTITIONS)
}

/// Per-worker memory the sub-partition merge refills job after job: the
/// sub-partition row buffers and the small target table. Reusing them keeps
/// the row copy inside memory that is already cached instead of streaming it
/// into fresh slabs.
pub(crate) struct SubPartitionScratch<KP: PersistedKey, V: AggregationValue + ?Sized> {
    allocator: SlabAllocator,
    sub_buffers: Vec<StridedScatterRows<KP, V>>,
    /// The target table with the capacity and hash shift it was built for.
    table: Option<(usize, u32, MultiSlabTable<KP, V>)>,
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> SubPartitionScratch<KP, V> {
    pub(crate) fn new() -> Self {
        Self {
            allocator: SlabAllocator::new(false),
            sub_buffers: Vec::new(),
            table: None,
        }
    }

    /// An empty target table of `capacity` slots hashing after `hash_left_shift`
    /// bits, reused when the previous job needed the same one.
    fn table(
        &mut self,
        capacity: usize,
        hash_left_shift: u32,
        context: &V::SharedContext,
    ) -> &mut MultiSlabTable<KP, V> {
        let matches = self
            .table
            .as_ref()
            .is_some_and(|(built_capacity, built_shift, _)| {
                *built_capacity == capacity && *built_shift == hash_left_shift
            });
        if !matches {
            let table = <MultiSlabTable<KP, V>>::new(
                &mut self.allocator,
                capacity,
                hash_left_shift,
                context,
            );
            self.table = Some((capacity, hash_left_shift, table));
        }
        &mut self.table.as_mut().unwrap().2
    }
}

/// Merges one partition and hands every merged table to `sink`.
///
/// A partition whose target fits [`SUB_PARTITION_TARGET_BYTES`] is merged
/// into one table. A larger partition is first split, by the hash bits that
/// follow the partition bits, into `sub_partitions` row buffers; each buffer
/// is then merged into one small table that is reused from sub-partition to
/// sub-partition. The table reaches `sink` once per sub-partition, so its
/// rows are still in cache when the sink reads them out.
#[allow(clippy::too_many_arguments)]
pub(super) fn merge_combined_into<S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[MultiSlabTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    sub_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    scratch: &mut SubPartitionScratch<S::Persisted, V>,
    sink: MergedTableSink<'_, S, V>,
) {
    if sub_partitions <= 1 {
        let table = merge_combined::<S, V>(
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
        );
        sink(&table);
        return;
    }
    V::dispatch_arity(
        V::storage_metadata(context),
        MergeSubPartitions::<S, V> {
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            sub_partitions,
            key_arena,
            context,
            scratch,
            sink,
        },
    )
}

/// State passed through arity dispatch for one sub-partitioned merge.
struct MergeSubPartitions<'a, S: StoredKey, V: AggregationValue + ?Sized> {
    partition: usize,
    buffers: &'a [PartitionBuffers<S::Persisted, V>],
    tables: &'a [MultiSlabTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    sub_partitions: usize,
    key_arena: &'a SharedArena,
    context: &'a V::SharedContext,
    scratch: &'a mut SubPartitionScratch<S::Persisted, V>,
    sink: MergedTableSink<'a, S, V>,
}

impl<S: StoredKey, V: AggregationValue + ?Sized> ArityBody<()> for MergeSubPartitions<'_, S, V> {
    #[inline(always)]
    fn run<const N: usize>(self) {
        let MergeSubPartitions {
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            sub_partitions,
            key_arena,
            context,
            scratch,
            sink,
        } = self;
        merge_sub_partitions::<N, S, V>(
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            sub_partitions,
            key_arena,
            context,
            scratch,
            sink,
        )
    }
}

/// Merges one partition in cache-resident sub-partitions after arity dispatch.
#[allow(clippy::too_many_arguments)]
fn merge_sub_partitions<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[MultiSlabTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    sub_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    scratch: &mut SubPartitionScratch<S::Persisted, V>,
    sink: MergedTableSink<'_, S, V>,
) {
    let partition_bits = num_partitions.trailing_zeros();
    let sub_bits = sub_partitions.trailing_zeros();
    let sub_shift = u64::BITS - sub_bits;
    let layout = StridedScatterRows::<S::Persisted, V>::layout::<N>(context);
    let requested_capacity = (partition_capacity / sub_partitions).max(DEFAULT_CAPACITY);
    // Build the table first so the row buffers can borrow the allocator.
    scratch.table(requested_capacity, partition_bits + sub_bits, context);
    let SubPartitionScratch {
        allocator,
        sub_buffers,
        table,
    } = scratch;
    let table = &mut table.as_mut().unwrap().2;
    while sub_buffers.len() < sub_partitions {
        sub_buffers.push(StridedScatterRows::new());
    }
    let sub_buffers = &mut sub_buffers[..sub_partitions];
    for sub_buffer in sub_buffers.iter_mut() {
        sub_buffer.reset();
    }

    // First pass: route every row of the partition to its sub-partition by
    // the hash bits that follow the partition bits.
    for_each_partition_row::<N, S, V>(
        partition,
        buffers,
        tables,
        num_partitions,
        context,
        |hash, key, stored| {
            let sub_partition = ((hash << partition_bits) >> sub_shift) as usize;
            sub_buffers[sub_partition]
                .push_with(layout, allocator, hash, *key, |row| row.copy_from(stored));
        },
    );

    // Second pass: merge each sub-partition into the one reused table. The
    // table hashes on the bits after the sub-partition bits, which are the
    // first bits that differ within a sub-partition.
    let mut capacity = table.capacity();
    for sub_buffer in sub_buffers.iter() {
        if sub_buffer.len() == 0 {
            continue;
        }
        {
            let mut target = if N == 0 {
                table.prober()
            } else {
                table.prober_with_metadata(V::metadata_for_arity::<N>())
            };
            // The target stays in cache; only string blobs are worth prefetching.
            if <S::Persisted as PersistedKey>::HAS_BLOB {
                sub_buffer.for_each_prefetched::<SCATTER_PREFETCH_AHEAD>(
                    layout,
                    |hash, key, stored, ahead| {
                        if let Some((_, ahead_key)) = ahead {
                            ahead_key.prefetch_blob(key_arena);
                        }
                        target.grow_if_full(allocator, &mut capacity);
                        let live_key = S::resolve_persisted(key_arena, *key);
                        target.merge_from::<false, _>(hash, live_key, stored, context);
                    },
                );
            } else {
                sub_buffer.for_each(layout, |hash, key, stored| {
                    target.grow_if_full(allocator, &mut capacity);
                    let live_key = S::resolve_persisted(key_arena, *key);
                    target.merge_from::<false, _>(hash, live_key, stored, context);
                });
            }
        }
        sink(table);
        table.clear();
    }
}

/// Visits every row that belongs to `partition`: the scatter rows of its
/// buckets from every worker, then the in-place table entries whose hash
/// routes to it.
fn for_each_partition_row<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[MultiSlabTable<S::Persisted, V>],
    num_partitions: usize,
    context: &V::SharedContext,
    mut visit: impl FnMut(u64, &S::Persisted, &V),
) {
    let partition_bits = num_partitions.trailing_zeros();
    let partition_shift = u64::BITS - partition_bits;
    let scatter_layout = StridedScatterRows::<S::Persisted, V>::layout::<N>(context);
    let scatter_bucket_count = buffers
        .first()
        .map_or(num_partitions, |buffer| buffer.0.len());
    let buckets_per_partition = (scatter_bucket_count / num_partitions).max(1);
    let first_bucket = partition * buckets_per_partition;
    for worker_buffers in buffers {
        for bucket in first_bucket..first_bucket + buckets_per_partition {
            worker_buffers.0[bucket]
                .for_each(scatter_layout, |hash, key, stored| visit(hash, key, stored));
        }
    }
    for table in tables {
        let slot_count = table.capacity();
        let reader = table.reader::<N>();
        let start = (partition * slot_count) >> partition_bits;
        let end = ((partition + 1) * slot_count) >> partition_bits;
        for slot in start..end {
            let entry = reader.view_at(slot);
            if entry.hash != 0 && (entry.hash >> partition_shift) as usize == partition {
                visit(entry.hash, entry.key, entry.stored);
            }
        }
        // Linear probing can push a partition's entries past its slot range.
        let mask = slot_count - 1;
        let mut slot = end & mask;
        loop {
            let entry = reader.view_at(slot);
            if entry.hash == 0 {
                break;
            }
            if (entry.hash >> partition_shift) as usize == partition {
                visit(entry.hash, entry.key, entry.stored);
            }
            slot = (slot + 1) & mask;
        }
    }
}

/// Merges one partition's node-local results.
///
/// The largest input is reused when it can hold the combined upper bound.
/// Otherwise a new table is allocated at the estimated partition capacity.
pub(super) fn merge_node_aggregated_tables<S: StoredKey, V: AggregationValue + ?Sized>(
    mut node_tables: Vec<MultiSlabTable<S::Persisted, V>>,
    partition_capacity: usize,
    partition_bits: u32,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> MultiSlabTable<S::Persisted, V> {
    let mut allocator = SlabAllocator::new(true);
    let total: usize = node_tables.iter().map(|table| table.len()).sum();
    let largest_table_index = node_tables
        .iter()
        .enumerate()
        .max_by_key(|(_, table)| table.len())
        .map(|(index, _)| index)
        .expect("a partition always has at least one node_table");
    let mut target = if total as f64
        <= node_tables[largest_table_index].capacity() as f64 * MAX_LOAD_FACTOR
    {
        node_tables.swap_remove(largest_table_index)
    } else {
        let capacity = partition_capacity.max(DEFAULT_CAPACITY);
        <MultiSlabTable<S::Persisted, V>>::new(&mut allocator, capacity, partition_bits, context)
    };
    let mut prober = target.prober();
    for node_table in &node_tables {
        for entry in node_table.iter(0) {
            prober.merge_from::<true, _>(
                entry.hash,
                S::resolve_persisted(key_arena, *entry.key),
                entry.stored,
                context,
            );
            resize_if_needed::<S::Persisted, V>(&mut allocator, &mut prober);
        }
    }
    // The prober's borrow of `target` ends at its last use above.
    target
}

#[cfg(test)]
mod tests {
    use super::super::PARTITIONS;
    use super::*;
    use crate::RECORD_BATCH_SIZE;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::arena::SharedArena;
    use crate::operations::unary::group::hashtables::{
        AggregatedTable, AggregatedTableOutput, RadixConfig,
    };
    use crate::operations::unary::group::keys::{InlineKey, IntKeyExtractor};
    use crate::operations::unary::group::values::{
        AggregationKind, AggregationSlot, AggregationValue, Compiled, CountSlot,
    };
    use ahash::RandomState;
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    type IntExtractor = IntKeyExtractor<Int32Type>;
    // How `IntExtractor` stores its keys. The merge is generic over this rather
    // than over the extractor, so this is what selects it.
    type IntStored = InlineKey<i32>;
    // A single `COUNT` slot. `Compiled` is numeric-only, so its `SharedContext`
    // is concretely `()`.
    type CountValue = Compiled<(CountSlot,), u8>;
    const COUNT_CFG: <CountValue as AggregationValue>::SharedContext = ();

    fn make_worker_tables(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
    ) -> Vec<MultiSlabTable<i32, CountValue>> {
        let mut agg = AggregatedTable::<IntExtractor, CountValue>::new(
            state.clone(),
            arena.clone(),
            (),
            (),
            RadixConfig::DEFAULT,
        );
        let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
        agg.consume_batch(
            &batch,
            &[0],
            &[AggregationSlot::new(
                AggregationKind::CountStar,
                0,
                DataType::Int64,
            )],
            &(),
            &COUNT_CFG,
        );
        // The test data is low-cardinality, so the worker never switches to radix:
        // `buffers` is None and the full result is in the in-place stack.
        let out = agg.flush();
        assert!(out.buffers.is_none(), "low-cardinality test stays in-place");
        out.tables
    }

    fn merge_all_partitions(
        tables: &[MultiSlabTable<i32, CountValue>],
        arena: &SharedArena,
    ) -> Vec<(i32, usize)> {
        let total_cap: usize = tables.iter().map(|t| t.capacity()).sum::<usize>() / 2;
        let partition_cap = (total_cap / PARTITIONS).max(1).next_power_of_two();

        let mut all_entries = vec![];
        for p in 0..PARTITIONS {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                &[],
                tables,
                partition_cap,
                PARTITIONS,
                arena,
                &COUNT_CFG,
            );
            for entry in result.iter(0) {
                all_entries.push((*entry.key, entry.stored.sort_key(0) as usize));
            }
        }
        all_entries.sort_by_key(|(k, _)| *k);
        all_entries
    }

    #[test]
    fn single_worker_all_entries_preserved() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let tables = make_worker_tables(&state, &arena, &[1, 2, 3, 4, 5]);

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 5);
        for (key, count) in &entries {
            assert_eq!(*count, 1, "key {} should have count 1", key);
        }
    }

    #[test]
    fn two_workers_disjoint_keys() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let mut tables = make_worker_tables(&state, &arena, &[1, 2, 3]);
        tables.extend(make_worker_tables(&state, &arena, &[4, 5, 6]));

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 6);
    }

    #[test]
    fn two_workers_overlapping_keys_merged() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let mut tables = make_worker_tables(&state, &arena, &[1, 2, 3]);
        tables.extend(make_worker_tables(&state, &arena, &[2, 3, 4]));

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 4);
        let count_for = |k: i32| entries.iter().find(|(key, _)| *key == k).unwrap().1;
        assert_eq!(count_for(1), 1);
        assert_eq!(count_for(2), 2);
        assert_eq!(count_for(3), 2);
        assert_eq!(count_for(4), 1);
    }

    #[test]
    fn empty_tables_produce_no_entries() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let tables = make_worker_tables(&state, &arena, &[]);

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn many_workers_large_overlap() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let mut tables = vec![];
        for _ in 0..8 {
            tables.extend(make_worker_tables(&state, &arena, &[10, 20, 30]));
        }

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 3);
        for (_, count) in &entries {
            assert_eq!(*count, 8);
        }
    }

    #[test]
    fn partitions_are_disjoint() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let values: Vec<i32> = (0..200).collect();
        let tables = make_worker_tables(&state, &arena, &values);

        let total_cap: usize = tables.iter().map(|t| t.capacity()).sum::<usize>() / 2;
        let partition_cap = (total_cap / PARTITIONS).max(1).next_power_of_two();
        let mut total = 0;
        for p in 0..PARTITIONS {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                &[],
                &tables,
                partition_cap,
                PARTITIONS,
                &arena,
                &COUNT_CFG,
            );
            total += result.iter(0).count();
        }

        assert_eq!(total, 200);
    }

    #[test]
    fn mixed_size_tables_merge_correctly() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let small = make_worker_tables(&state, &arena, &[1, 2]);
        let big_values: Vec<i32> = (0..500).collect();
        let big = make_worker_tables(&state, &arena, &big_values);
        let mut tables = small;
        tables.extend(big);

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 500);
        let count_for = |k: i32| entries.iter().find(|(key, _)| *key == k).unwrap().1;
        assert_eq!(count_for(1), 2);
        assert_eq!(count_for(2), 2);
        assert_eq!(count_for(499), 1);
    }

    #[test]
    fn duplicates_within_single_worker_carry_through_merge() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let tables = make_worker_tables(&state, &arena, &[5, 5, 5, 5, 5]);

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], (5, 5));
    }

    /// Node-hierarchical merges combine per-node aggregated tables (each a
    /// `merge_combined` result for the same partition) into one final table;
    /// keys shared across nodes must combine, node-disjoint keys must all
    /// survive.
    #[test]
    fn node_tables_merge_across_nodes() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let node_a = make_worker_tables(&state, &arena, &[1, 2, 3]);
        let node_b = make_worker_tables(&state, &arena, &[2, 3, 4]);

        let mut entries = vec![];
        for p in 0..PARTITIONS {
            let node_tables = vec![
                merge_combined::<IntStored, CountValue>(
                    p,
                    &[],
                    &node_a,
                    4,
                    PARTITIONS,
                    &arena,
                    &COUNT_CFG,
                ),
                merge_combined::<IntStored, CountValue>(
                    p,
                    &[],
                    &node_b,
                    4,
                    PARTITIONS,
                    &arena,
                    &COUNT_CFG,
                ),
            ];
            let folded = merge_node_aggregated_tables::<IntStored, CountValue>(
                node_tables,
                DEFAULT_CAPACITY,
                PARTITIONS.trailing_zeros(),
                &arena,
                &COUNT_CFG,
            );
            for entry in folded.iter(0) {
                entries.push((*entry.key, entry.stored.sort_key(0) as usize));
            }
        }

        entries.sort();
        assert_eq!(entries, vec![(1, 1), (2, 2), (3, 2), (4, 1)]);
    }

    /// Feed `values` through a worker in batch-sized chunks and hand back its
    /// finalized output (in-place stack + scatter buffers if it switched).
    fn make_worker_output(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
    ) -> AggregatedTableOutput<IntExtractor, CountValue> {
        let mut agg = AggregatedTable::<IntExtractor, CountValue>::new(
            state.clone(),
            arena.clone(),
            (),
            (),
            RadixConfig::DEFAULT,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        // Consume one batch at a time because hash scratch has a fixed size.
        for chunk in values.chunks(RECORD_BATCH_SIZE) {
            let array: ArrayRef = Arc::new(Int32Array::from(chunk.to_vec()));
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            agg.consume_batch(
                &batch,
                &[0],
                &[AggregationSlot::new(
                    AggregationKind::CountStar,
                    0,
                    DataType::Int64,
                )],
                &(),
                &COUNT_CFG,
            );
        }
        agg.flush()
    }

    /// A partition split into sub-partitions yields the same groups with the
    /// same counts as the whole-partition merge, from scatter rows and the
    /// pre-switch in-place stack alike.
    #[test]
    fn sub_partition_merge_matches_whole_partition_merge() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let values: Vec<i32> = (0..40_000).chain(0..20_000).collect();
        let out = make_worker_output(&state, &arena, &values);
        assert!(out.buffers.is_some(), "40000 keys switch to radix");
        let buffers: Vec<_> = out.buffers.into_iter().collect();
        let num_partitions = 256;
        let mut scratch = SubPartitionScratch::new();

        for partition in 0..num_partitions {
            let whole = merge_combined::<IntStored, CountValue>(
                partition,
                &buffers,
                &out.tables,
                DEFAULT_CAPACITY,
                num_partitions,
                &arena,
                &COUNT_CFG,
            );
            let mut expected: Vec<(i32, i128)> = whole
                .iter(0)
                .map(|entry| (*entry.key, entry.stored.sort_key(0)))
                .collect();
            let mut actual = Vec::new();
            merge_combined_into::<IntStored, CountValue>(
                partition,
                &buffers,
                &out.tables,
                DEFAULT_CAPACITY,
                num_partitions,
                4,
                &arena,
                &COUNT_CFG,
                &mut scratch,
                &mut |table| {
                    actual.extend(
                        table
                            .iter(0)
                            .map(|entry| (*entry.key, entry.stored.sort_key(0))),
                    )
                },
            );

            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected, "partition {partition}");
        }
    }

    #[test]
    fn sub_partition_count_follows_target_size_and_merge_shape() {
        assert_eq!(sub_partition_count(128, 32, 100, 100), 1);
        assert_eq!(sub_partition_count(1024, 32, 100, 100), 1);
        assert_eq!(sub_partition_count(8192, 32, 100, 100), 8);
        assert_eq!(sub_partition_count(8192, 32, 1000, 100), 1);
        assert_eq!(sub_partition_count(65536, 48, 1000, 100), 64);
        assert_eq!(sub_partition_count(1 << 20, 64, 100, 100), 64);
    }

    /// A worker with enough distinct keys to switch to radix produces scatter
    /// buffers *and* a pre-switch in-place stack. The radix merge must combine
    /// both at RADIX_PARTITIONS granularity, landing every key in exactly one
    /// partition with the right count, losing and double-counting nothing.
    /// The radix path slot-range-merges in-place stacks at RADIX_PARTITIONS, a
    /// granularity that can exceed a stack table's slot count, the regime the
    /// 64-way merge never hits. Exercise it directly: 200 distinct keys (stack
    /// tables of 128 and 512 slots) merged at `num_partitions = 256`, finer than
    /// the 128-slot table. Each key must still land in exactly one partition with
    /// the right count (the degenerate within-range plus probe-chain spillover in
    /// `merge_past_partition_bounds` has to lose and duplicate nothing).
    #[test]
    fn slot_range_merge_finer_than_table() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let n = 200i32;
        let values: Vec<i32> = (0..n).chain(0..n).collect(); // each key twice
        let out = make_worker_output(&state, &arena, &values);
        assert!(out.buffers.is_none(), "200 keys stay in-place");

        let num_partitions = 256;
        let mut occurrences = vec![0usize; n as usize];
        let mut counts = vec![0usize; n as usize];
        for p in 0..num_partitions {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                &[],
                &out.tables,
                DEFAULT_CAPACITY,
                num_partitions,
                &arena,
                &COUNT_CFG,
            );
            for entry in result.iter(0) {
                occurrences[*entry.key as usize] += 1;
                counts[*entry.key as usize] += entry.stored.sort_key(0) as usize;
            }
        }

        for k in 0..n as usize {
            assert_eq!(
                occurrences[k], 1,
                "key {k} must be in exactly one partition"
            );
            assert_eq!(counts[k], 2, "key {k} has the wrong count");
        }
    }
}
