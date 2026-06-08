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

#![allow(dead_code)] // old slot-range merge kept for reference; radix uses aggregate_partition

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::PartitionBuffers;
use crate::operations::unary::group::hashtables::{
    DEFAULT_CAPACITY, KeyExtractor, MultiSlabTable, Table, TableStorage, ValueExtractor,
};

use super::PARTITIONS;
use super::RADIX_PARTITIONS;

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

const PARTITION_SHIFT: u32 = 64 - PARTITIONS.trailing_zeros();
const PARTITION_BITS: u32 = PARTITIONS.trailing_zeros();

/// Number of slots to process per batch in the main scan loop. Kept small
/// so that the target-table region touched by one batch stays cache-hot
/// across all source tables.
const SCAN_BATCH_SIZE: usize = 100;

const PREFETCH_DISTANCE: usize = 8;

/// Try to resize `target` if cumulative collision pressure is too high.
#[inline]
fn resize_if_needed<K: KeyExtractor, V: ValueExtractor>(
    allocator: &mut SlabAllocator,
    target: &mut MultiSlabTable<K, V>,
) {
    // Integer form of `collisions / len > RESIZE_COLLISION_RATIO`. This runs on
    // the critical path after every insert, so we avoid the int→float converts
    // and the float division (~20+ cycles each) that would otherwise serialize
    // the merge's random-access inserts and cap throughput well below DRAM
    // bandwidth. `RESIZE_COLLISION_RATIO` is 2.0, so the test is `collisions > 2*len`.
    if target.collisions() > target.len() * RESIZE_COLLISION_RATIO as usize {
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
fn merge_within_partition_bounds<K: KeyExtractor, V: ValueExtractor, S: TableStorage<K, V>>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&Table<K, V, S>],
    target: &mut MultiSlabTable<K, V>,
) {
    let start = (partition * slot_count) >> PARTITION_BITS;
    let end = ((partition + 1) * slot_count) >> PARTITION_BITS;
    let mut i = start;

    while i < end {
        let range = std::cmp::min(SCAN_BATCH_SIZE, end - i);

        for table in tables {
            for j in i..i + range {
                if j + PREFETCH_DISTANCE < slot_count {
                    // Prefetch unconditionally: `slot_for(0)` is a valid slot and
                    // a prefetch is only a hint, so dropping the `!= 0` guard
                    // removes a ~70/30-biased (load-factor) branch from this hot
                    // inner loop, which dominated the merge's branch mispredicts.
                    target.prefetch(table.entry_at(j + PREFETCH_DISTANCE).hash());
                }
                let entry = table.entry_at(j);
                let h = entry.hash();
                // Single non-short-circuiting `&` so this is one branch instead of
                // two: both the empty-slot test (`h != 0`) and the partition test
                // are data-dependent on random hashes and mispredict heavily.
                let take = (h != 0) & ((h >> PARTITION_SHIFT) as usize == partition);
                if take {
                    target.merge::<true, _>(
                        h,
                        K::resolve_persisted(arena, *entry.key()),
                        *entry.value(),
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
fn merge_past_partition_bounds<K: KeyExtractor, V: ValueExtractor, S: TableStorage<K, V>>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&Table<K, V, S>],
    target: &mut MultiSlabTable<K, V>,
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
                resize_if_needed::<K, V>(allocator, target);
            }
            i = (i + 1) & mask;
        }
    }
}

/// Merge entries from `tables` that belong to `partition` into `target`.
fn merge_into_partition<K: KeyExtractor, V: ValueExtractor, S: TableStorage<K, V>>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: Vec<&Table<K, V, S>>,
    target: &mut MultiSlabTable<K, V>,
) {
    merge_within_partition_bounds::<K, V, S>(
        allocator, arena, partition, slot_count, &tables, target,
    );
    merge_past_partition_bounds::<K, V, S>(
        allocator, arena, partition, slot_count, &tables, target,
    );
}

/// Merge all per-worker tables for a single partition into one result table.
///
/// Source tables are grouped by capacity so that same-sized tables share
/// the same slot range and can be walked together for better locality.
pub(super) fn merge_partition<K: KeyExtractor, V: ValueExtractor>(
    partition: usize,
    tables: &[MultiSlabTable<K, V>],
    arena: &SharedArena,
    partition_capacity: usize,
) -> MultiSlabTable<K, V> {
    let mut allocator = SlabAllocator::new(true);
    let mut target: MultiSlabTable<K, V> = <MultiSlabTable<K, V>>::multi_slab(
        &mut allocator,
        partition_capacity.max(DEFAULT_CAPACITY),
        PARTITIONS.trailing_zeros(),
    );

    // Group source tables by capacity so same-sized tables can be walked in
    // lockstep — entries at the same slot index have similar hashes and hit
    // the same target region, keeping it cache-hot.
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
        merge_into_partition::<K, V, _>(
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


/// **Single-level radix merge**: aggregate every worker's raw scatter rows for
/// `partition` into one result table. Because the consume phase does no
/// aggregation, this is the *only* aggregation pass — each row is inserted once,
/// into a table sized to the partition's distinct count so it stays cache
/// resident. `pre_shift` skips the partition's top bits so the table slots on
/// the bits below them (uses the whole table, not 1/PARTITIONS of it).
pub(super) fn aggregate_partition<K: KeyExtractor, V: ValueExtractor>(
    partition: usize,
    worker_buffers: &[PartitionBuffers<K, V>],
    inplace_tables: &[MultiSlabTable<K, V>],
    partition_capacity: usize,
    arena: &SharedArena,
) -> MultiSlabTable<K, V> {
    let pre_shift = RADIX_PARTITIONS.trailing_zeros();
    let shift = u64::BITS - RADIX_PARTITIONS.trailing_zeros();
    let mut allocator = SlabAllocator::new(true);
    let mut cap = partition_capacity;
    let mut target: MultiSlabTable<K, V> =
        <MultiSlabTable<K, V>>::multi_slab(&mut allocator, cap, pre_shift);
    for wb in worker_buffers {
        wb.0[partition].for_each(|(hash, key, value)| {
            if target.undersized() {
                cap *= 4;
                let nb = allocator.create_multi_slab_buffer(cap, true);
                target.resize_with(nb, cap);
            }
            let live = K::resolve_persisted(arena, key);
            target.merge::<false, _>(hash, live, value);
        });
    }
    // Mixed case: any worker that never switched contributes its (small) in-place
    // stack; route only the entries hashing into this partition. Empty (no cost)
    // in the common all-switched case.
    for table in inplace_tables {
        for entry in table.iter(0) {
            if (entry.hash() >> shift) as usize == partition {
                if target.undersized() {
                    cap *= 4;
                    let nb = allocator.create_multi_slab_buffer(cap, true);
                    target.resize_with(nb, cap);
                }
                let live = K::resolve_persisted(arena, *entry.key());
                target.merge::<false, _>(entry.hash(), live, *entry.value());
            }
        }
    }
    target
}
