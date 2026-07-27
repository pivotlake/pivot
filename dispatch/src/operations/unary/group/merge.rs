//! Partition-parallel merge of per-worker hash tables.
//!
//! During the GROUP BY consume phase each worker builds its own set of
//! hash tables. Once all workers finish, we merge those tables into a
//! single result per partition.
//!
//! ## Partitioning scheme
//!
//! The top `log2(PARTITIONS)` bits of each hash determine its partition.
//! Because the source tables use those same top bits for slot placement,
//! entries belonging to partition P are clustered in a predictable slot
//! range — roughly `[P * capacity/PARTITIONS, (P+1) * capacity/PARTITIONS)`.
//!
//! ## Merge strategy
//!
//! [`merge_combined`] is called once per partition. It combines two sources into
//! one target table: the scatter buffers of any worker that switched to radix,
//! and every worker's in-place stack. For the in-place stacks it:
//!
//! 1. Groups source tables by capacity so same-sized tables can be walked
//!    in lockstep (shared slot range → shared cache lines in the target).
//! 2. For each group, scans the expected slot range in small batches with
//!    software prefetching, then handles overflow from linear-probing
//!    chains that spill past the range boundary (including wrap-around).
//! 3. Grows the target table when the cumulative collision-to-entry ratio
//!    exceeds [`RESIZE_COLLISION_RATIO`], which corresponds to ~70% effective
//!    load (derived from Knuth's linear-probing analysis: ratio = α / 2(1-α)).
//!
//! The partition count (`num_partitions`) is the in-place `PARTITIONS` when nobody
//! switched and `RADIX_PARTITIONS` otherwise; the in-place tables slot on the same
//! top bits the scatter partitions on, so both sources land in the same partition.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::PartitionBuffers;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, KeyExtractor, MAX_LOAD_FACTOR, MultiSlabTable,
    PersistedKey, Prober,
};

/// Collision-to-entry ratio at which we double the target table.
///
/// For a linear-probing table filled from empty, the cumulative ratio of
/// total collisions to total insertions is `α / (2(1 − α))` where `α` is
/// the load factor. A threshold of 1.5 triggers resize at ~75% load. The
/// merged result table is built once then scanned sequentially for output, so
/// a higher load (more probing on insert) is a good trade for less memory to
/// allocate and zero — significant at very large group counts. 2.0 corresponds to ~80%
/// load, so a partition sized for ~78% occupancy doesn't resize (which would
/// otherwise double it back to the over-provisioned size).
const RESIZE_COLLISION_RATIO: f64 = 2.0;

/// Number of slots to process per batch in the main scan loop. Kept small
/// so that the target-table region touched by one batch stays cache-hot
/// across all source tables.
const SCAN_BATCH_SIZE: usize = 100;

const PREFETCH_DISTANCE: usize = 8;

/// Try to resize the target if cumulative collision pressure is too high.
#[inline]
fn resize_if_needed<K: KeyExtractor, V: AggregationValue>(
    allocator: &mut SlabAllocator,
    target: &mut Prober<'_, <K as KeyExtractor>::Persisted, V>,
) {
    // Integer form of `collisions / len > RESIZE_COLLISION_RATIO`. This runs on
    // the critical path after every insert, so we avoid the int→float converts
    // and the float division (~20+ cycles each) that would otherwise serialize
    // the merge's random-access inserts and cap throughput well below DRAM
    // bandwidth. `RESIZE_COLLISION_RATIO` is 2.0, so the test is `collisions > 2*len`.
    target.resize_on_collisions(allocator, RESIZE_COLLISION_RATIO as usize);
}

/// Scan the expected slot range `[start, end)` across all source `tables`,
/// merging entries that belong to `partition` into `target`.
///
/// Slots are processed in small batches ([`SCAN_BATCH_SIZE`]) so that when
/// we round-robin through the source tables, the target region stays in
/// cache. Each batch also prefetches [`PREFETCH_DISTANCE`] slots ahead.
#[allow(clippy::too_many_arguments)]
fn merge_within_partition_bounds<K: KeyExtractor, V: AggregationValue>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&MultiSlabTable<K, V>],
    target: &mut Prober<'_, <K as KeyExtractor>::Persisted, V>,
    partition_bits: u32,
    cfg: &V::SharedContext,
) {
    let partition_shift = u64::BITS - partition_bits;
    let start = (partition * slot_count) >> partition_bits;
    let end = ((partition + 1) * slot_count) >> partition_bits;
    let mut i = start;

    while i < end {
        let range = std::cmp::min(SCAN_BATCH_SIZE, end - i);

        for table in tables {
            // The scan touches `range` consecutive slots; read them through
            // one geometry snapshot so the source layout stays in registers.
            let source = table.reader();
            for j in i..i + range {
                if j + PREFETCH_DISTANCE < slot_count {
                    // Prefetch unconditionally: `slot_for(0)` is a valid slot and
                    // a prefetch is only a hint, so dropping the `!= 0` guard
                    // removes a ~70/30-biased (load-factor) branch from this hot
                    // inner loop, which dominated the merge's branch mispredicts.
                    target.prefetch(source.hash_at(j + PREFETCH_DISTANCE));
                }
                let entry = source.view_at(j);
                let h = entry.hash;
                // Single non-short-circuiting `&` so this is one branch instead of
                // two: both the empty-slot test (`h != 0`) and the partition test
                // are data-dependent on random hashes and mispredict heavily.
                let take = (h != 0) & ((h >> partition_shift) as usize == partition);
                if take {
                    target.merge_from::<true, _>(
                        h,
                        K::resolve_persisted(arena, *entry.key),
                        entry.stored,
                        cfg,
                    );
                    resize_if_needed::<K, V>(allocator, target);
                }
            }
        }

        i += range;
    }
}

/// Scan the linear-probing overflow past the partition's expected slot range.
///
/// Entries displaced by collisions can land just past `end`. For the last
/// partition this means wrapping around to slot 0. We follow each source
/// table's probe chain until hitting an empty slot (`hash == 0`).
#[allow(clippy::too_many_arguments)]
fn merge_past_partition_bounds<K: KeyExtractor, V: AggregationValue>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&MultiSlabTable<K, V>],
    target: &mut Prober<'_, <K as KeyExtractor>::Persisted, V>,
    partition_bits: u32,
    cfg: &V::SharedContext,
) {
    let partition_shift = u64::BITS - partition_bits;
    let end = ((partition + 1) * slot_count) >> partition_bits;
    let mask = slot_count - 1;

    for table in tables {
        let source = table.reader();
        let mut i = end & mask;
        loop {
            let entry = source.view_at(i);
            let h = entry.hash;
            if h == 0 {
                break;
            }
            if (h >> partition_shift) as usize == partition {
                target.merge_from::<true, _>(
                    h,
                    K::resolve_persisted(arena, *entry.key),
                    entry.stored,
                    cfg,
                );
                resize_if_needed::<K, V>(allocator, target);
            }
            i = (i + 1) & mask;
        }
    }
}

/// Merge entries from `tables` that belong to `partition` into `target`.
#[allow(clippy::too_many_arguments)]
fn merge_into_partition<K: KeyExtractor, V: AggregationValue>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: Vec<&MultiSlabTable<K, V>>,
    target: &mut Prober<'_, <K as KeyExtractor>::Persisted, V>,
    partition_bits: u32,
    cfg: &V::SharedContext,
) {
    merge_within_partition_bounds::<K, V>(
        allocator,
        arena,
        partition,
        slot_count,
        &tables,
        target,
        partition_bits,
        cfg,
    );
    merge_past_partition_bounds::<K, V>(
        allocator,
        arena,
        partition,
        slot_count,
        &tables,
        target,
        partition_bits,
        cfg,
    );
}

/// Merge one partition's sources into a result table. First aggregate every
/// switched worker's scatter rows for this partition (each inserted once into a
/// target HLL-sized to stay cache-resident), then slot-range merge the in-place
/// stacks at the same `num_partitions` granularity. Because the in-place tables
/// slot on the top hash bits — the same bits the scatter partitions on — both
/// sources fall into this partition, so one pass combines them with no pre-fold.
/// `buffers` is empty in the all-in-place case (`num_partitions == PARTITIONS`);
/// `tables` holds switched workers' pre-switch stacks and non-switched workers'
/// full stacks. `partition_bits` = log2(num_partitions) sets the target's
/// pre_shift, skipping the partition's top bits so it slots on the bits below.
pub(super) fn merge_combined<K: KeyExtractor, V: AggregationValue>(
    partition: usize,
    buffers: &[PartitionBuffers<K, V>],
    tables: &[MultiSlabTable<K, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    cfg: &V::SharedContext,
) -> MultiSlabTable<K, V> {
    let partition_bits = num_partitions.trailing_zeros();
    let mut allocator = SlabAllocator::new(true);
    let mut cap = partition_capacity.max(DEFAULT_CAPACITY);
    let mut result: MultiSlabTable<K, V> =
        <MultiSlabTable<K, V>>::new(&mut allocator, cap, partition_bits, cfg);
    // One geometry snapshot for the whole partition job: the scatter fold and
    // the slot-range merges below probe the target once per row/entry, and a
    // per-call snapshot would be paid on each. Resizes go through the prober,
    // which re-snapshots.
    let mut target = result.prober();

    // 1. Scatter buffers (present only when some worker switched). Each row is
    //    inserted once — the consume phase did no aggregation, so this is the only
    //    aggregation pass for the scattered rows. The compare inside `merge` chases
    //    the scattered row's string blob, which has no locality (rows land here in
    //    scatter order, not by key), so it misses cold. Warm it a window ahead off
    //    the handle in hand, and warm the target slot it will probe. (The target's
    //    own blob is left alone: hot keys repeat, so resident entries stay warm.)
    const SCATTER_PREFETCH_AHEAD: usize = 16;
    // The scatter-merge prefetch only pays off when its reads are genuinely cold:
    // the target slot is worth prefetching only when the per-partition target is
    // too large to stay cache-resident (high cardinality), and a string key's
    // arena blob is cold regardless. For a fixed-width key with a small (cache-hot)
    // target the prefetch is pure overhead (it chases slots already resident), so
    // scan plainly there.
    const TARGET_PREFETCH_MIN_SLOTS: usize = 16384;
    let merge_prefetch =
        <K::Persisted as PersistedKey>::HAS_BLOB || cap > TARGET_PREFETCH_MIN_SLOTS;
    // `num_partitions` may be coarser than the scatter bucket count: at moderate
    // cardinality the output sizes the merge from the exact HLL estimate so each
    // job folds a contiguous *range* of scatter buckets (a `PARTITIONS`-way merge
    // rather than a `RADIX_PARTITIONS`-way one), avoiding thousands of tiny merge
    // jobs. The range is contiguous because buckets and partitions both key on the
    // top hash bits; `stride == 1` (one bucket per partition) at high cardinality.
    let scatter_buckets = buffers.first().map_or(num_partitions, |b| b.0.len());
    // The fold below visits buckets `[0, num_partitions * stride)`; for that to be
    // every bucket, `num_partitions` must divide `scatter_buckets` evenly. Both are
    // powers of two with `num_partitions <= scatter_buckets`, so it holds. A count
    // that didn't divide would silently drop the tail buckets' rows, so guard it.
    debug_assert!(
        scatter_buckets.is_multiple_of(num_partitions),
        "merge partitions ({num_partitions}) must evenly divide scatter buckets ({scatter_buckets})"
    );
    let stride = (scatter_buckets / num_partitions).max(1);
    let bucket_lo = partition * stride;
    let bucket_hi = bucket_lo + stride;
    for wb in buffers {
        for bucket in bucket_lo..bucket_hi {
            if merge_prefetch {
                wb.0[bucket].for_each_prefetched::<SCATTER_PREFETCH_AHEAD>(
                    |(hash, key, value), ahead| {
                        if let Some((ahead_hash, ahead_key, _)) = ahead {
                            ahead_key.prefetch_blob(key_arena);
                            target.prefetch(*ahead_hash);
                        }
                        target.grow_if_full(&mut allocator, &mut cap);
                        let live = K::resolve_persisted(key_arena, key);
                        target.merge::<false, _>(hash, live, value, cfg);
                    },
                );
            } else {
                wb.0[bucket].for_each(|(hash, key, value)| {
                    target.grow_if_full(&mut allocator, &mut cap);
                    let live = K::resolve_persisted(key_arena, key);
                    target.merge::<false, _>(hash, live, value, cfg);
                });
            }
        }
    }

    // 2. In-place stacks, slot-range merged at num_partitions. Grouped by size so
    //    same-sized tables walk in lockstep (shared slot range -> cache-hot target).
    //    The within+past pair handles any size, including stacks smaller than
    //    num_partitions (degenerate within-range, past picks up the spillover).
    let mut by_size: Vec<(usize, Vec<&MultiSlabTable<K, V>>)> = Vec::new();
    for table in tables {
        let slot_count = table.capacity();
        if let Some(group) = by_size.iter_mut().find(|(s, _)| *s == slot_count) {
            group.1.push(table);
        } else {
            by_size.push((slot_count, vec![table]));
        }
    }
    for (slot_count, group) in by_size.into_iter() {
        merge_into_partition::<K, V>(
            &mut allocator,
            key_arena,
            partition,
            slot_count,
            group,
            &mut target,
            partition_bits,
            cfg,
        );
    }
    // The prober's borrow of `result` ends at its last use above.
    result
}
/// Merge one partition's per-node aggregated tables into its final table.
///
/// Each input is one NUMA node's [`merge_combined`] result for the same
/// partition, so it is already fully aggregated within its node and every
/// entry belongs to this partition. Only the keys shared *across* nodes
/// remain to be combined, which is why this final, partly-remote pass is so
/// much cheaper than merging every worker's raw tables across nodes would be.
///
/// When the largest node table can hold every entry at the table's load
/// factor (typical when nodes saw mostly the same keys), it becomes the
/// target and only the other tables are merged in. Otherwise (mostly
/// node-disjoint keys) a fresh table sized for the whole partition
/// (`partition_capacity`, derived from the global estimate) is filled from
/// all of them, so the merge doesn't resize mid-way.
pub(super) fn merge_node_aggregated_tables<K: KeyExtractor, V: AggregationValue>(
    mut node_tables: Vec<MultiSlabTable<K, V>>,
    partition_capacity: usize,
    partition_bits: u32,
    key_arena: &SharedArena,
    cfg: &V::SharedContext,
) -> MultiSlabTable<K, V> {
    let mut allocator = SlabAllocator::new(true);
    let total: usize = node_tables.iter().map(|f| f.len()).sum();
    let largest = node_tables
        .iter()
        .enumerate()
        .max_by_key(|(_, f)| f.len())
        .map(|(i, _)| i)
        .expect("a partition always has at least one node_table");
    let mut target = if total as f64 <= node_tables[largest].capacity() as f64 * MAX_LOAD_FACTOR {
        node_tables.swap_remove(largest)
    } else {
        let cap = partition_capacity.max(DEFAULT_CAPACITY);
        <MultiSlabTable<K, V>>::new(&mut allocator, cap, partition_bits, cfg)
    };
    let mut prober = target.prober();
    for node_table in &node_tables {
        for entry in node_table.iter(0) {
            prober.merge_from::<true, _>(
                entry.hash,
                K::resolve_persisted(key_arena, *entry.key),
                entry.stored,
                cfg,
            );
            resize_if_needed::<K, V>(&mut allocator, &mut prober);
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
    use crate::operations::unary::group::keys::IntKeyExtractor;
    use crate::operations::unary::group::values::{
        AggregationKind, AggregationSlot, AggregationValue, Compiled, CountSlot, OwnedValue,
    };
    use ahash::RandomState;
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    type IntExtractor = IntKeyExtractor<Int32Type>;
    // A single `COUNT` slot. `Compiled` is numeric-only, so its `SharedContext`
    // is concretely `()`.
    type CountValue = Compiled<(CountSlot,)>;
    const COUNT_CFG: <CountValue as AggregationValue>::SharedContext = ();

    fn make_worker_tables(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
    ) -> Vec<MultiSlabTable<IntExtractor, CountValue>> {
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
        tables: &[MultiSlabTable<IntExtractor, CountValue>],
        arena: &SharedArena,
    ) -> Vec<(i32, usize)> {
        let total_cap: usize = tables.iter().map(|t| t.capacity()).sum::<usize>() / 2;
        let partition_cap = (total_cap / PARTITIONS).max(1).next_power_of_two();

        let mut all_entries = vec![];
        for p in 0..PARTITIONS {
            let result = merge_combined::<IntExtractor, CountValue>(
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
            let result = merge_combined::<IntExtractor, CountValue>(
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
                merge_combined::<IntExtractor, CountValue>(
                    p,
                    &[],
                    &node_a,
                    4,
                    PARTITIONS,
                    &arena,
                    &COUNT_CFG,
                ),
                merge_combined::<IntExtractor, CountValue>(
                    p,
                    &[],
                    &node_b,
                    4,
                    PARTITIONS,
                    &arena,
                    &COUNT_CFG,
                ),
            ];
            let folded = merge_node_aggregated_tables::<IntExtractor, CountValue>(
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
        // One batch at a time — consume_batch's scratch is sized for RECORD_BATCH_SIZE.
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
    /// both — at RADIX_PARTITIONS granularity — landing every key in exactly one
    /// partition with the right count, losing and double-counting nothing.
    /// The radix path slot-range-merges in-place stacks at RADIX_PARTITIONS, a
    /// granularity that can exceed a stack table's slot count — the regime the
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
            let result = merge_combined::<IntExtractor, CountValue>(
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
