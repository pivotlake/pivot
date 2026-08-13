//! Partition-parallel merge of GROUP BY partial results.
//!
//! High hash bits select both the merge partition and the source-table slot
//! range, so each merge job reads only its own portion of every source table
//! and folds it into one target: the partition's final result.
//!
//! Tables with equal capacity are scanned together. Linear-probe entries that
//! crossed the nominal partition boundary are handled by a short overflow scan.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, MAX_LOAD_FACTOR, MultiSlabTable, PersistedKey, Prober,
    TableReader,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::values::ArityBody;

/// Tables grouped by the slot count they were built at. Equal-sized tables share
/// partition boundaries, so the merge scans them together.
type TablesByCapacity<'a, S, V> = Vec<(
    usize,
    Vec<&'a MultiSlabTable<<S as StoredKey>::Persisted, V>>,
)>;

/// Collision-to-entry ratio at which we double the target table.
///
/// For linear probing, the cumulative collision ratio is
/// `alpha / (2 * (1 - alpha))`. A threshold of two allows dense result tables
/// without letting probe chains grow without bound.
const RESIZE_COLLISION_RATIO: f64 = 2.0;

/// Slots processed from each equal-sized source before moving to the next.
///
/// Only bites when a partition's slice of one table is longer than this: the
/// slice is then walked in batches, interleaved across tables, so the region of
/// the target they land in stays cache-warm. A slice shorter than this is
/// walked whole, one table at a time.
const SCAN_BATCH_SIZE: usize = 100;

/// How many source tables ahead to warm while scanning the current one.
const TABLE_PREFETCH_DISTANCE: usize = 8;

const PREFETCH_DISTANCE: usize = 8;

/// Try to resize the target if cumulative collision pressure is too high.
#[inline]
fn resize_if_needed<KP: PersistedKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    target: &mut Prober<'_, KP, V>,
) {
    // Use the integer form of `collisions / len > ratio`.
    target.resize_on_collisions(allocator, RESIZE_COLLISION_RATIO as usize);
}

/// Merge the slots of one source `reader` in `[from, to)` that belong to
/// `partition` into `target`.
///
/// The target prefetch reads a hash from further along this same slice. Looking
/// past its end would prefetch a target slot for a hash belonging to a
/// different partition, which no probe here will ever want: pure eviction.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn merge_slot_range<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    reader: &TableReader<'_, S::Persisted, V>,
    from: usize,
    to: usize,
    target: &mut Prober<'_, S::Persisted, V>,
    partition_shift: u32,
    context: &V::SharedContext,
) {
    // Locate this partition's entries first, then fold them. Splitting the two
    // keeps the unpredictable occupancy test out of the loop that does the
    // work, and leaves a dense list whose target slots can all be prefetched.
    let mut found: [*const u8; SCAN_BATCH_SIZE] = [std::ptr::null(); SCAN_BATCH_SIZE];
    let mut scanned = from;
    while scanned < to {
        let chunk_end = to.min(scanned + SCAN_BATCH_SIZE);
        let count = reader.collect_partition_entries(
            scanned,
            chunk_end,
            partition_shift,
            partition,
            &mut found,
        );
        for (position, &entry) in found[..count].iter().enumerate() {
            if position + PREFETCH_DISTANCE < count {
                target.prefetch(reader.hash_of(found[position + PREFETCH_DISTANCE]));
            }
            let hash = reader.hash_of(entry);
            let view = unsafe { reader.view_of(entry, hash) };
            target.merge_from::<true, _>(
                hash,
                S::resolve_persisted(arena, *view.key),
                view.stored,
                context,
            );
            resize_if_needed::<S::Persisted, V>(allocator, target);
        }
        scanned = chunk_end;
    }
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
        let batch_end = i + range;

        for (index, table) in tables.iter().enumerate() {
            // Reuse one source reader for this range.
            let source_reader = table.reader::<N>();
            // A partition owns a narrow slice of each table, so consecutive
            // tables are scattered addresses the hardware prefetcher cannot
            // predict. Warm its whole slice a few tables ahead.
            if let Some(ahead) = tables.get(index + TABLE_PREFETCH_DISTANCE) {
                ahead.reader::<N>().prefetch_entries(i, batch_end);
            }
            merge_slot_range::<N, S, V>(
                allocator,
                arena,
                partition,
                &source_reader,
                i,
                batch_end,
                target,
                partition_shift,
                context,
            );
        }

        i += range;
    }
}

/// Takes this partition's entries from one source's probe-chain overflow: the
/// run of occupied slots starting where its nominal range ends, which is where
/// linear probing can have pushed them.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn merge_overflow_tail<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    reader: &TableReader<'_, S::Persisted, V>,
    end: usize,
    mask: usize,
    target: &mut Prober<'_, S::Persisted, V>,
    partition_shift: u32,
    context: &V::SharedContext,
) {
    let mut i = end & mask;
    loop {
        let entry = reader.view_at(i);
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
        merge_overflow_tail::<N, S, V>(
            allocator,
            arena,
            partition,
            &table.reader::<N>(),
            end,
            mask,
            target,
            partition_shift,
            context,
        );
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
    let start = (partition * slot_count) >> partition_bits;
    let end = ((partition + 1) * slot_count) >> partition_bits;
    // With many source tables a partition owns only a few slots of each, and
    // the two scans below would each walk the whole table list, touching every
    // table twice from addresses far enough apart that the first visit has been
    // evicted by the time the second arrives. Short slices are therefore taken
    // whole -- nominal range and overflow tail together -- in a single visit per
    // table. Long slices keep the batched form, which interleaves tables to hold
    // the target region they land in cache-warm.
    if end - start <= SCAN_BATCH_SIZE {
        let partition_shift = u64::BITS - partition_bits;
        let mask = slot_count - 1;
        for (index, table) in tables.iter().enumerate() {
            let source_reader = table.reader::<N>();
            if let Some(ahead) = tables.get(index + TABLE_PREFETCH_DISTANCE) {
                ahead.reader::<N>().prefetch_entries(start, end);
            }
            merge_slot_range::<N, S, V>(
                allocator,
                arena,
                partition,
                &source_reader,
                start,
                end,
                target,
                partition_shift,
                context,
            );
            merge_overflow_tail::<N, S, V>(
                allocator,
                arena,
                partition,
                &source_reader,
                end,
                mask,
                target,
                partition_shift,
                context,
            );
        }
        return;
    }
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

/// Merges one partition's share of every source table.
pub(super) fn merge_combined<S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
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
            tables,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
        } = self;
        merge_combined_rows::<N, S, V>(
            partition,
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
    tables: &[MultiSlabTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> MultiSlabTable<S::Persisted, V> {
    {
        let partition_bits = num_partitions.trailing_zeros();
        let mut allocator = SlabAllocator::new(true);
        let capacity = partition_capacity.max(DEFAULT_CAPACITY);
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
        AggregatedTable, AggregatedTableOutput, SpillConfig,
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
            SpillConfig::DEFAULT,
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
        agg.flush().tables
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
                    p, &node_a, 4, PARTITIONS, &arena, &COUNT_CFG,
                ),
                merge_combined::<IntStored, CountValue>(
                    p, &node_b, 4, PARTITIONS, &arena, &COUNT_CFG,
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
            SpillConfig::DEFAULT,
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
        let num_partitions = 256;
        let mut occurrences = vec![0usize; n as usize];
        let mut counts = vec![0usize; n as usize];
        for p in 0..num_partitions {
            let result = merge_combined::<IntStored, CountValue>(
                p,
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
