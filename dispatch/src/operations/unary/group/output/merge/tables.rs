//! Merging a partition's entries out of the in-place hash tables.
//!
//! Tables with equal capacity are scanned together. Linear-probe entries that
//! crossed the nominal partition boundary are handled by a short overflow scan.

use crate::cpu_features::multitarget_kernel;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, HashTable, PersistedKey, Prober,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::output::topk_pruning::{TopKCutoff, topk_admits};
use crate::operations::unary::group::values::ArityBody;

/// Tables grouped by the slot count they were built at. Equal-sized tables share
/// partition boundaries, so the merge scans them together.
type TablesByCapacity<'a, S, V> = Vec<(usize, Vec<&'a HashTable<<S as StoredKey>::Persisted, V>>)>;

/// Collision-to-entry ratio at which we double the target table.
///
/// For linear probing, the cumulative collision ratio is
/// `alpha / (2 * (1 - alpha))`. A threshold of two allows dense result tables
/// without letting probe chains grow without bound.
const RESIZE_COLLISION_RATIO: f64 = 2.0;

/// Slots processed from each equal-sized source before moving to the next.
const SCAN_BATCH_SIZE: usize = 100;

const PREFETCH_DISTANCE: usize = 8;

/// Try to resize the target if cumulative collision pressure is too high.
#[inline]
pub(super) fn resize_if_needed<KP: PersistedKey, V: AggregationValue + ?Sized>(
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
    tables: &[&HashTable<S::Persisted, V>],
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
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
                if take && topk_admits(cutoff, hash) {
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
    tables: &[&HashTable<S::Persisted, V>],
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
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
            if (hash >> partition_shift) as usize == partition && topk_admits(cutoff, hash) {
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
    tables: Vec<&HashTable<S::Persisted, V>>,
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
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
        cutoff,
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
        cutoff,
    );
}

/// Merge a partition's entries of the in-place `tables` into `table`.
#[allow(clippy::too_many_arguments)]
pub(super) fn merge_tables<S: StoredKey, V: AggregationValue + ?Sized>(
    table: &mut HashTable<S::Persisted, V>,
    allocator: &mut SlabAllocator,
    partition: usize,
    partition_bits: u32,
    tables: &[HashTable<S::Persisted, V>],
    key_arena: &SharedArena,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) {
    V::dispatch_arity(
        V::storage_metadata(context),
        MergeTables::<S, V> {
            table,
            allocator,
            partition,
            partition_bits,
            tables,
            key_arena,
            context,
            cutoff,
        },
    )
}

/// State passed through arity dispatch for merging the in-place tables.
struct MergeTables<'a, S: StoredKey, V: AggregationValue + ?Sized> {
    table: &'a mut HashTable<S::Persisted, V>,
    allocator: &'a mut SlabAllocator,
    partition: usize,
    partition_bits: u32,
    tables: &'a [HashTable<S::Persisted, V>],
    key_arena: &'a SharedArena,
    context: &'a V::SharedContext,
    cutoff: Option<TopKCutoff<'a>>,
}

impl<S: StoredKey, V: AggregationValue + ?Sized> ArityBody<()> for MergeTables<'_, S, V> {
    #[inline(always)]
    fn run<const N: usize>(self) {
        let MergeTables {
            table,
            allocator,
            partition,
            partition_bits,
            tables,
            key_arena,
            context,
            cutoff,
        } = self;
        tables_kernel::<N, S, V>(
            table,
            allocator,
            partition,
            partition_bits,
            tables,
            key_arena,
            context,
            cutoff,
        )
    }
}

multitarget_kernel! {
    /// The in-place table merge, cloned per CPU tier like the scatter rows.
    #[allow(clippy::too_many_arguments)]
    fn tables_kernel[const N: usize, S: StoredKey, V: AggregationValue + ?Sized]::[N, S, V](
        table: &mut HashTable<<S as StoredKey>::Persisted, V>,
        allocator: &mut SlabAllocator,
        partition: usize,
        partition_bits: u32,
        tables: &[HashTable<<S as StoredKey>::Persisted, V>],
        key_arena: &SharedArena,
        context: &V::SharedContext,
        cutoff: Option<TopKCutoff<'_>>
    ) {
        let mut target = Prober::new::<N>(table);
        // Scan equal-sized tables together so they share partition boundaries.
        let mut tables_by_capacity: TablesByCapacity<'_, S, V> = Vec::new();
        for source in tables {
            let slot_count = source.capacity();
            if let Some((_, same_size_tables)) = tables_by_capacity
                .iter_mut()
                .find(|(group_slot_count, _)| *group_slot_count == slot_count)
            {
                same_size_tables.push(source);
            } else {
                tables_by_capacity.push((slot_count, vec![source]));
            }
        }
        for (slot_count, same_size_tables) in tables_by_capacity {
            merge_into_partition::<N, S, V>(
                allocator,
                key_arena,
                partition,
                slot_count,
                same_size_tables,
                &mut target,
                partition_bits,
                context,
                cutoff,
            );
        }
    }
}
