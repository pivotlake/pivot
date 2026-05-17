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
//! [`merge_partition`] is called once per partition. It:
//!
//! 1. Groups source tables by capacity so same-sized tables can be walked
//!    in lockstep (shared slot range → shared cache lines in the target).
//! 2. For each group, scans the expected slot range in small batches with
//!    software prefetching, then handles overflow from linear-probing
//!    chains that spill past the range boundary (including wrap-around).
//! 3. Grows the target table when the cumulative collision-to-entry ratio
//!    exceeds [`RESIZE_COLLISION_RATIO`], which corresponds to ~70% effective
//!    load (derived from Knuth's linear-probing analysis: ratio = α / 2(1-α)).

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    DEFAULT_CAPACITY, KeyExtractor, MultiSlabTable, Table, TableStorage,
};

use super::PARTITIONS;

/// Collision-to-entry ratio at which we double the target table.
///
/// For a linear-probing table filled from empty, the cumulative ratio of
/// total collisions to total insertions is `α / (2(1 − α))` where `α` is
/// the load factor. A threshold of 1.17 triggers resize at ~70% load,
/// matching the static `MAX_LOAD_FACTOR` used elsewhere.
const RESIZE_COLLISION_RATIO: f64 = 1.17;

const PARTITION_SHIFT: u32 = 64 - PARTITIONS.trailing_zeros();
const PARTITION_BITS: u32 = PARTITIONS.trailing_zeros();

/// Number of slots to process per batch in the main scan loop. Kept small
/// so that the target-table region touched by one batch stays cache-hot
/// across all source tables.
const SCAN_BATCH_SIZE: usize = 100;

const PREFETCH_DISTANCE: usize = 8;

/// Try to resize `target` if cumulative collision pressure is too high.
#[inline]
fn resize_if_needed<K: KeyExtractor>(
    allocator: &mut SlabAllocator,
    target: &mut MultiSlabTable<K>,
) {
    if target.collisions() as f64 / target.len() as f64 > RESIZE_COLLISION_RATIO {
        let new_size = target.capacity() << 1;
        target.resize_with(allocator.create_multi_slab_buffer(new_size, true), new_size);
    }
}

/// Scan the expected slot range `[start, end)` across all source `tables`,
/// merging entries that belong to `partition` into `target`.
///
/// Slots are processed in small batches ([`SCAN_BATCH_SIZE`]) so that when
/// we round-robin through the source tables, the target region stays in
/// cache. Each batch also prefetches [`PREFETCH_DISTANCE`] slots ahead.
fn merge_within_partition_bounds<K: KeyExtractor, S: TableStorage<K>>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&Table<K, S>],
    target: &mut MultiSlabTable<K>,
) {
    let start = (partition * slot_count) >> PARTITION_BITS;
    let end = ((partition + 1) * slot_count) >> PARTITION_BITS;
    let mut i = start;

    while i < end {
        let range = std::cmp::min(SCAN_BATCH_SIZE, end - i);

        for table in tables {
            for j in i..i + range {
                if j + PREFETCH_DISTANCE < slot_count {
                    let future_hash = table.entry_at(j + PREFETCH_DISTANCE).hash();
                    if future_hash != 0 {
                        target.prefetch(future_hash);
                    }
                }
                let entry = table.entry_at(j);
                let h = entry.hash();
                if h != 0 && (h >> PARTITION_SHIFT) as usize == partition {
                    target.merge::<true, _>(
                        h,
                        K::resolve_persisted(arena, *entry.key()),
                        *entry.value(),
                    );
                    resize_if_needed::<K>(allocator, target);
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
fn merge_past_partition_bounds<K: KeyExtractor, S: TableStorage<K>>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&Table<K, S>],
    target: &mut MultiSlabTable<K>,
) {
    let end = ((partition + 1) * slot_count) >> PARTITION_BITS;
    let mask = slot_count - 1;

    for table in tables {
        let mut i = end & mask;
        loop {
            let entry = table.entry_at(i);
            let h = entry.hash();
            if h == 0 {
                break;
            }
            if (h >> PARTITION_SHIFT) as usize == partition {
                target.merge::<true, _>(
                    h,
                    K::resolve_persisted(arena, *entry.key()),
                    *entry.value(),
                );
                resize_if_needed::<K>(allocator, target);
            }
            i = (i + 1) & mask;
        }
    }
}

/// Merge entries from `tables` that belong to `partition` into `target`.
fn merge_into_partition<K: KeyExtractor, S: TableStorage<K>>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: Vec<&Table<K, S>>,
    target: &mut MultiSlabTable<K>,
) {
    merge_within_partition_bounds::<K, S>(allocator, arena, partition, slot_count, &tables, target);
    merge_past_partition_bounds::<K, S>(allocator, arena, partition, slot_count, &tables, target);
}

/// Merge all per-worker tables for a single partition into one result table.
///
/// Source tables are grouped by capacity so that same-sized tables share
/// the same slot range and can be walked together for better locality.
pub(super) fn merge_partition<K: KeyExtractor>(
    partition: usize,
    tables: &[MultiSlabTable<K>],
    arena: &SharedArena,
    partition_capacity: usize,
) -> MultiSlabTable<K> {
    let mut allocator = SlabAllocator::new(true);
    let mut target: MultiSlabTable<K> = <MultiSlabTable<K>>::multi_slab(
        &mut allocator,
        partition_capacity.max(DEFAULT_CAPACITY),
        PARTITIONS.trailing_zeros(),
    );

    // Group source tables by capacity so same-sized tables can be walked in
    // lockstep — entries at the same slot index have similar hashes and hit
    // the same target region, keeping it cache-hot.
    let mut by_size: Vec<(usize, Vec<&MultiSlabTable<K>>)> = Vec::new();
    for table in tables {
        let slot_count = table.capacity();
        if let Some(group) = by_size.iter_mut().find(|(s, _)| *s == slot_count) {
            group.1.push(table);
        } else {
            by_size.push((slot_count, vec![table]));
        }
    }

    for (slot_count, group) in by_size.into_iter() {
        merge_into_partition::<K, _>(
            &mut allocator,
            arena,
            partition,
            slot_count,
            group,
            &mut target,
        );
    }

    target
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::arena::SharedArena;
    use crate::operations::unary::group::hashtables::AggregatedTable;
    use crate::operations::unary::group::key_extractions::IntKeyExtractor;
    use ahash::RandomState;
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array};
    use std::sync::Arc;

    type IntExtractor = IntKeyExtractor<Int32Type>;

    fn make_worker_tables(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
    ) -> Vec<MultiSlabTable<IntExtractor>> {
        let mut agg = AggregatedTable::<IntExtractor>::new(state.clone(), arena.clone());
        let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        agg.merge_array(&array);
        agg.flush()
    }

    fn merge_all_partitions(
        tables: &[MultiSlabTable<IntExtractor>],
        arena: &SharedArena,
    ) -> Vec<(i32, usize)> {
        let total_cap: usize = tables.iter().map(|t| t.capacity()).sum::<usize>() / 2;
        let partition_cap = (total_cap / PARTITIONS).max(1).next_power_of_two();

        let mut all_entries = vec![];
        for p in 0..PARTITIONS {
            let result = merge_partition::<IntExtractor>(p, tables, arena, partition_cap);
            for entry in result.iter(0) {
                all_entries.push((*entry.key(), entry.value().value));
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
            let result = merge_partition::<IntExtractor>(p, &tables, &arena, partition_cap);
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
}
