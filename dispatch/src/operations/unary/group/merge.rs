//! Partition-parallel streaming merge of GROUP BY partial results.
//!
//! Every retired table carries a hash-ordered run of its occupied slots, so a
//! partition's share of each table is one contiguous slice of that run. The
//! merge walks all slices at once with a loser tree keyed on the stored hash
//! and emits each group the moment its last partial has been seen: groups
//! stream straight into the output sink, and no result table is ever built.
//!
//! ```text
//! run 0:  [ h3 h9 ... ]--+
//! run 1:  [ h1 h9 ... ]--+--> loser tree --> h1, h3, h9(merged), ... --> sink
//! run 2:  [ h5 ...    ]--+
//! ```
//!
//! A group held by exactly one run (the common case at high cardinality) is
//! emitted directly from its table entry, with no copies and no value merge.
//! When several runs hold the same hash, the entries are gathered and grouped
//! by their real key, so two distinct keys sharing a 64-bit hash stay two
//! groups.

use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, BUCKET_BITS, LiveKey, PersistedKey, SealedTable, TableReader,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::values::ArityBody;

use super::Result;

/// How many entries ahead each cursor warms its run's cache lines.
const CURSOR_PREFETCH: usize = 8;

/// Loser-tree sort key: a live cursor's `hash << 1`; exhausted cursors sort
/// after every live hash, including `u64::MAX`.
const EXHAUSTED: u128 = u128::MAX;

#[inline(always)]
fn live_key(hash: u64) -> u128 {
    (hash as u128) << 1
}

/// One run's walk over its partition slice, in hash order.
struct Cursor<'t, KP, V: AggregationValue + ?Sized> {
    reader: TableReader<'t, KP, V>,
    positions: *const u32,
    pos: usize,
    end: usize,
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> Cursor<'_, KP, V> {
    /// The entry the cursor currently points at.
    #[inline(always)]
    fn entry(&self) -> *const u8 {
        let slot = unsafe { *self.positions.add(self.pos) } as usize;
        self.reader.entry_ptr(slot)
    }

    /// Steps past the current entry and returns the next sort key.
    #[inline(always)]
    fn advance(&mut self) -> u128 {
        self.pos += 1;
        if self.pos == self.end {
            return EXHAUSTED;
        }
        if self.pos + CURSOR_PREFETCH < self.end {
            let ahead = unsafe { *self.positions.add(self.pos + CURSOR_PREFETCH) } as usize;
            self.reader.prefetch_entry(ahead);
        }
        live_key(self.reader.hash_of(self.entry()))
    }
}

/// A loser tree over the cursors' current sort keys.
///
/// Internal nodes hold the loser of their subtree's play-off and the root
/// holds the overall winner, so replacing the winner's key replays exactly one
/// leaf-to-root path: log2(leaves) comparisons per emitted entry.
struct LoserTree {
    /// Current sort key per leaf, padded to a power of two with [`EXHAUSTED`].
    keys: Vec<u128>,
    /// `nodes[0]` is the winner; `nodes[1..]` hold each play-off's loser.
    nodes: Vec<u32>,
    leaves: usize,
}

impl LoserTree {
    fn new(keys: Vec<u128>) -> Self {
        let leaves = keys.len();
        debug_assert!(leaves.is_power_of_two());
        let mut nodes = vec![0u32; leaves.max(1)];
        let mut winners = vec![0u32; 2 * leaves];
        for (i, winner) in winners[leaves..].iter_mut().enumerate() {
            *winner = i as u32;
        }
        for n in (1..leaves).rev() {
            let a = winners[2 * n];
            let b = winners[2 * n + 1];
            let (winner, loser) = if keys[a as usize] <= keys[b as usize] {
                (a, b)
            } else {
                (b, a)
            };
            winners[n] = winner;
            nodes[n] = loser;
        }
        nodes[0] = winners[1.min(2 * leaves - 1)];
        Self {
            keys,
            nodes,
            leaves,
        }
    }

    /// The leaf holding the smallest key.
    #[inline(always)]
    fn winner(&self) -> usize {
        self.nodes[0] as usize
    }

    #[inline(always)]
    fn key(&self, leaf: usize) -> u128 {
        self.keys[leaf]
    }

    /// Replaces the winner leaf's key and replays its path to the root.
    #[inline(always)]
    fn replay(&mut self, leaf: usize, new_key: u128) {
        self.keys[leaf] = new_key;
        let mut winner = leaf as u32;
        let mut node = (leaf + self.leaves) >> 1;
        while node >= 1 {
            let loser = self.nodes[node];
            if self.keys[loser as usize] < self.keys[winner as usize] {
                self.nodes[node] = winner;
                winner = loser;
            }
            node >>= 1;
        }
        self.nodes[0] = winner;
    }
}

/// Heap scratch holding one merged aggregation value.
struct ScratchValue<V: AggregationValue + ?Sized> {
    buf: *mut u8,
    layout: std::alloc::Layout,
    metadata: V::StorageMetadata,
}

impl<V: AggregationValue + ?Sized> ScratchValue<V> {
    fn new(metadata: V::StorageMetadata) -> Self {
        let layout =
            std::alloc::Layout::from_size_align(V::stored_size(metadata).max(1), V::stored_align())
                .expect("value layout is valid");
        // Zeroed to match the slab-zeroed slots `copy_from` normally seeds.
        let buf = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!buf.is_null(), "scratch value allocation failed");
        Self {
            buf,
            layout,
            metadata,
        }
    }

    #[inline(always)]
    fn value_mut(&mut self) -> &mut V {
        unsafe { V::from_entry_mut(self.buf, self.metadata) }
    }

    #[inline(always)]
    fn value(&self) -> &V {
        unsafe { V::from_entry(self.buf, self.metadata) }
    }
}

impl<V: AggregationValue + ?Sized> Drop for ScratchValue<V> {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.buf, self.layout) };
    }
}

/// Streams one partition's merged groups into `emit`.
///
/// `emit` receives each group exactly once and returns whether the merge
/// should keep going (`false` stops early, for satisfied LIMIT pushdowns).
pub(super) fn merge_partition<S: StoredKey, V: AggregationValue + ?Sized, E>(
    partition: usize,
    num_partitions: usize,
    tables: &[SealedTable<S::Persisted, V>],
    key_arena: &SharedArena,
    context: &V::SharedContext,
    emit: E,
) -> Result<()>
where
    E: FnMut(&S::Persisted, &V) -> Result<bool>,
{
    // Dispatch once so every loop in this job shares the specialized arity.
    V::dispatch_arity(
        V::storage_metadata(context),
        MergePartition::<S, V, E> {
            partition,
            num_partitions,
            tables,
            key_arena,
            context,
            emit,
        },
    )
}

/// State passed through arity dispatch for one partition merge.
struct MergePartition<'a, S: StoredKey, V: AggregationValue + ?Sized, E> {
    partition: usize,
    num_partitions: usize,
    tables: &'a [SealedTable<S::Persisted, V>],
    key_arena: &'a SharedArena,
    context: &'a V::SharedContext,
    emit: E,
}

impl<S: StoredKey, V: AggregationValue + ?Sized, E> ArityBody<Result<()>>
    for MergePartition<'_, S, V, E>
where
    E: FnMut(&S::Persisted, &V) -> Result<bool>,
{
    #[inline(always)]
    fn run<const N: usize>(self) -> Result<()> {
        let MergePartition {
            partition,
            num_partitions,
            tables,
            key_arena,
            context,
            emit,
        } = self;
        merge_partition_rows::<N, S, V, E>(
            partition,
            num_partitions,
            tables,
            key_arena,
            context,
            emit,
        )
    }
}

/// Merges one partition after arity dispatch.
#[inline(never)]
fn merge_partition_rows<const N: usize, S: StoredKey, V: AggregationValue + ?Sized, E>(
    partition: usize,
    num_partitions: usize,
    tables: &[SealedTable<S::Persisted, V>],
    key_arena: &SharedArena,
    context: &V::SharedContext,
    mut emit: E,
) -> Result<()>
where
    E: FnMut(&S::Persisted, &V) -> Result<bool>,
{
    let partition_bits = num_partitions.trailing_zeros();
    debug_assert!(partition_bits <= BUCKET_BITS);
    let buckets_per_partition = BUCKET_BITS - partition_bits;
    let bucket_lo = partition << buckets_per_partition;
    let bucket_hi = (partition + 1) << buckets_per_partition;

    let mut cursors: Vec<Cursor<'_, S::Persisted, V>> = Vec::with_capacity(tables.len());
    for sealed in tables {
        let (start, end) = sealed.run.bucket_range(bucket_lo, bucket_hi);
        if start == end {
            continue;
        }
        cursors.push(Cursor {
            reader: sealed.table.reader::<N>(),
            positions: sealed.run.positions_ptr(),
            pos: start,
            end,
        });
    }
    if cursors.is_empty() {
        return Ok(());
    }
    // Every table shares one entry layout, so any reader can decode any
    // entry address; keep one copy for the emit paths.
    let layout = cursors[0].reader;

    let leaves = cursors.len().next_power_of_two();
    let mut keys = vec![EXHAUSTED; leaves];
    for (i, cursor) in cursors.iter().enumerate() {
        // Warm the head of each slice; the per-advance prefetch takes over
        // from here.
        let lookahead = (cursor.end - cursor.pos).min(CURSOR_PREFETCH);
        for k in cursor.pos..cursor.pos + lookahead {
            let slot = unsafe { *cursor.positions.add(k) } as usize;
            cursor.reader.prefetch_entry(slot);
        }
        keys[i] = live_key(cursor.reader.hash_of(cursor.entry()));
    }
    let mut tree = LoserTree::new(keys);

    let metadata = if N == 0 {
        V::storage_metadata(context)
    } else {
        V::metadata_for_arity::<N>()
    };
    let mut scratch = ScratchValue::<V>::new(metadata);
    let mut cluster: Vec<*const u8> = Vec::new();

    loop {
        let winner = tree.winner();
        let winner_key = tree.key(winner);
        if winner_key == EXHAUSTED {
            return Ok(());
        }
        let entry = cursors[winner].entry();
        let next_key = cursors[winner].advance();
        tree.replay(winner, next_key);

        let hash = (winner_key >> 1) as u64;
        if tree.key(tree.winner()) != winner_key {
            // Only one run holds this hash: emit straight from the entry.
            let view = unsafe { layout.view_of(entry, hash) };
            if !emit(view.key, view.stored)? {
                return Ok(());
            }
            continue;
        }

        // Several entries share this hash. Gather them, then group by the
        // real key: distinct keys colliding on a hash stay distinct groups.
        cluster.clear();
        cluster.push(entry);
        while tree.key(tree.winner()) == winner_key {
            let runner_up = tree.winner();
            cluster.push(cursors[runner_up].entry());
            let key = cursors[runner_up].advance();
            tree.replay(runner_up, key);
        }
        while let Some(&first) = cluster.first() {
            let first_view = unsafe { layout.view_of(first, hash) };
            let live = S::resolve_persisted(key_arena, *first_view.key);
            let mut merged = false;
            let mut i = 1;
            while i < cluster.len() {
                let other_view = unsafe { layout.view_of(cluster[i], hash) };
                if live.eq_persisted(other_view.key) {
                    if !merged {
                        scratch.value_mut().copy_from(first_view.stored);
                        merged = true;
                    }
                    scratch.value_mut().merge_from(other_view.stored, context);
                    cluster.swap_remove(i);
                } else {
                    i += 1;
                }
            }
            cluster.swap_remove(0);
            let value = if merged {
                scratch.value()
            } else {
                first_view.stored
            };
            if !emit(first_view.key, value)? {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::PARTITIONS;
    use super::*;
    use crate::RECORD_BATCH_SIZE;
    use crate::memory::{SlabAllocator, init_test_free_pool};
    use crate::operations::unary::group::arena::SharedArena;
    use crate::operations::unary::group::hashtables::{
        AggregatedTable, MultiSlabTable, SpillConfig,
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
    ) -> Vec<SealedTable<i32, CountValue>> {
        make_worker_tables_with_spill(state, arena, values, SpillConfig::DEFAULT)
    }

    fn make_worker_tables_with_spill(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
        spill: SpillConfig,
    ) -> Vec<SealedTable<i32, CountValue>> {
        let mut agg = AggregatedTable::<IntExtractor, CountValue>::new(
            state.clone(),
            arena.clone(),
            (),
            (),
            spill,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        for chunk in values.chunks(RECORD_BATCH_SIZE.max(1)) {
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
        agg.flush().tables
    }

    fn collect_partition(
        tables: &[SealedTable<i32, CountValue>],
        arena: &SharedArena,
        partition: usize,
        num_partitions: usize,
    ) -> Vec<(i32, usize)> {
        let mut groups = vec![];
        merge_partition::<IntStored, CountValue, _>(
            partition,
            num_partitions,
            tables,
            arena,
            &COUNT_CFG,
            |key, value| {
                groups.push((*key, value.sort_key(0) as usize));
                Ok(true)
            },
        )
        .unwrap();
        groups
    }

    fn merge_all_partitions(
        tables: &[SealedTable<i32, CountValue>],
        arena: &SharedArena,
    ) -> Vec<(i32, usize)> {
        let mut all_entries = vec![];
        for p in 0..PARTITIONS {
            all_entries.extend(collect_partition(tables, arena, p, PARTITIONS));
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

        let mut total = 0;
        for p in 0..PARTITIONS {
            total += collect_partition(&tables, &arena, p, PARTITIONS).len();
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

    #[test]
    fn stacked_tables_recombine_across_and_within_workers() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let spill = SpillConfig {
            spill_capacity: 256,
        };
        let values: Vec<i32> = (0..500).chain(0..500).collect();
        let mut tables = make_worker_tables_with_spill(&state, &arena, &values, spill);
        tables.extend(make_worker_tables_with_spill(
            &state, &arena, &values, spill,
        ));

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 500);
        assert!(entries.iter().all(|&(_, count)| count == 4));
    }

    /// Build a sealed table directly from `(hash, key)` pairs, bypassing the
    /// extractor so tests control hash collisions and extreme hash values.
    fn sealed_from_pairs(pairs: &[(u64, i32)]) -> SealedTable<i32, CountValue> {
        let mut allocator = SlabAllocator::new(true);
        let mut table: MultiSlabTable<i32, CountValue> =
            MultiSlabTable::new(&mut allocator, 128, 0, &());
        for &(hash, key) in pairs {
            table.prober().merge_from(hash, key, &seeded_count(), &());
        }
        let run = table.build_sorted_run();
        SealedTable { table, run }
    }

    /// A `CountValue` holding a count of one, as one consumed row seeds it.
    fn seeded_count() -> CountValue {
        let mut value = CountValue::default();
        value.seed(&((),), 0, &mut ());
        value
    }

    fn collect_all_from(tables: &[SealedTable<i32, CountValue>]) -> Vec<(i32, usize)> {
        let arena = SharedArena::new(64);
        let mut entries = vec![];
        for p in 0..PARTITIONS {
            entries.extend(collect_partition(tables, &arena, p, PARTITIONS));
        }
        entries.sort();
        entries
    }

    #[test]
    fn same_hash_distinct_keys_stay_distinct_groups() {
        init_test_free_pool(64);
        // Three runs hold hash 42: two with key 1, one with key 2. The merged
        // output must combine the key-1 partials and keep key 2 separate.
        let tables = vec![
            sealed_from_pairs(&[(42, 1)]),
            sealed_from_pairs(&[(42, 2)]),
            sealed_from_pairs(&[(42, 1)]),
        ];

        let entries = collect_all_from(&tables);

        assert_eq!(entries, vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn same_hash_distinct_keys_within_one_run() {
        init_test_free_pool(64);
        let tables = vec![sealed_from_pairs(&[(42, 1), (42, 2), (42, 3)])];

        let entries = collect_all_from(&tables);

        assert_eq!(entries, vec![(1, 1), (2, 1), (3, 1)]);
    }

    #[test]
    fn extreme_hash_values_merge_without_sentinel_confusion() {
        init_test_free_pool(64);
        // u64::MAX must not collide with the exhausted-cursor sentinel, and
        // hash 1 (the remapped zero) must land in partition 0.
        let tables = vec![
            sealed_from_pairs(&[(u64::MAX, 7), (1, 9)]),
            sealed_from_pairs(&[(u64::MAX, 7), (u64::MAX - 1, 8)]),
        ];

        let entries = collect_all_from(&tables);

        assert_eq!(entries, vec![(7, 2), (8, 1), (9, 1)]);
    }

    #[test]
    fn sorted_runs_are_hash_ordered_after_wraparound() {
        init_test_free_pool(64);
        // Hashes near u64::MAX probe past the last slot and wrap to slot 0;
        // the run must still come out in ascending hash order.
        let pairs: Vec<(u64, i32)> = (0..40)
            .map(|i| (u64::MAX - i as u64, i as i32))
            .chain((1..40).map(|i| (i as u64, 100 + i as i32)))
            .collect();
        let sealed = sealed_from_pairs(&pairs);

        let reader = sealed.table.reader::<0>();
        let mut previous = 0u64;
        for i in 0..sealed.run.len() {
            let slot = unsafe { *sealed.run.positions_ptr().add(i) } as usize;
            let hash = reader.hash_of(reader.entry_ptr(slot));
            assert!(hash >= previous, "run out of order at {}", i);
            previous = hash;
        }
        assert_eq!(sealed.run.len(), pairs.len());
    }

    #[test]
    fn early_stop_halts_partition_merge() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let values: Vec<i32> = (0..1000).collect();
        let tables = make_worker_tables(&state, &arena, &values);

        let mut seen = 0usize;
        for p in 0..PARTITIONS {
            merge_partition::<IntStored, CountValue, _>(
                p,
                PARTITIONS,
                &tables,
                &arena,
                &(),
                |_, _| {
                    seen += 1;
                    Ok(seen < 10)
                },
            )
            .unwrap();
        }

        assert!(seen < 1000, "merge stopped early");
    }
}
