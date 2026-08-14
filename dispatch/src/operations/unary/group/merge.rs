//! Partition-parallel merge of GROUP BY partial results.
//!
//! High hash bits select the merge partition, and every worker's flush
//! consolidates its table stack into one bucket-grouped [`DenseRun`], so a
//! partition's share of each worker is one contiguous byte range. Each merge
//! job reads worker-count sources sequentially and folds their entries into
//! one target table: the partition's final result. No slot is ever tested
//! for occupancy or partition membership, and the nearly hash-ordered slices
//! keep the target's probed range sweeping left to right.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, BUCKET_BITS, DEFAULT_CAPACITY, MAX_LOAD_FACTOR, MergeSource, MultiSlabTable,
    PersistedKey, Prober, TableReader,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::values::ArityBody;

/// Collision-to-entry ratio at which we double the target table.
///
/// For linear probing, the cumulative collision ratio is
/// `alpha / (2 * (1 - alpha))`. A threshold of two allows dense result tables
/// without letting probe chains grow without bound.
const RESIZE_COLLISION_RATIO: f64 = 2.0;

/// Entries gathered per fold batch. The gather pass reads each batch's
/// entries (nearly sequential in the source) and issues their target-slot
/// prefetches together, so by the time the fold pass probes a target line it
/// has had a whole batch of lead time.
const FOLD_BATCH: usize = 48;

/// Try to resize the target if cumulative collision pressure is too high.
#[inline]
fn resize_if_needed<KP: PersistedKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    target: &mut Prober<'_, KP, V>,
) {
    // Use the integer form of `collisions / len > ratio`.
    target.resize_on_collisions(allocator, RESIZE_COLLISION_RATIO as usize);
}

/// Folds one source slice `[from, to)` into `target`.
///
/// `entry_at` maps a slice index to its entry address: direct indexing for a
/// dense run, slot indirection for a sealed table. Either way the slice holds
/// only this partition's entries. Each batch is gathered first (reading the
/// source close to sequentially and prefetching every target slot it will
/// probe), then folded, so the fold's probes land on lines whose fetch
/// started a batch earlier.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn merge_run_slice<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    reader: &TableReader<'_, S::Persisted, V>,
    entry_at: impl Fn(usize) -> *const u8,
    from: usize,
    to: usize,
    target: &mut Prober<'_, S::Persisted, V>,
    context: &V::SharedContext,
) {
    let mut batch: [(*const u8, u64); FOLD_BATCH] = [(std::ptr::null(), 0); FOLD_BATCH];
    let mut i = from;
    while i < to {
        let batch_len = (to - i).min(FOLD_BATCH);
        for (j, index) in (i..i + batch_len).enumerate() {
            let entry = entry_at(index);
            let hash = reader.hash_of(entry);
            batch[j] = (entry, hash);
            target.prefetch(hash);
            if <S::Persisted as PersistedKey>::HAS_BLOB {
                let view = unsafe { reader.view_of(entry, hash) };
                view.key.prefetch_blob(arena);
            }
        }
        for &(entry, hash) in &batch[..batch_len] {
            let view = unsafe { reader.view_of(entry, hash) };
            target.merge_from::<true, _>(
                hash,
                S::resolve_persisted(arena, *view.key),
                view.stored,
                context,
            );
            resize_if_needed::<S::Persisted, V>(allocator, target);
        }
        i += batch_len;
    }
}

/// Merges one partition's share of every worker's source.
pub(super) fn merge_combined<S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    sources: &[MergeSource<S::Persisted, V>],
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
            sources,
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
    sources: &'a [MergeSource<S::Persisted, V>],
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
            sources,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
        } = self;
        merge_combined_rows::<N, S, V>(
            partition,
            sources,
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
fn merge_combined_rows<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    sources: &[MergeSource<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> MultiSlabTable<S::Persisted, V> {
    {
        let partition_bits = num_partitions.trailing_zeros();
        debug_assert!(partition_bits <= BUCKET_BITS);
        let buckets_per_partition = BUCKET_BITS - partition_bits;
        let bucket_lo = partition << buckets_per_partition;
        let bucket_hi = (partition + 1) << buckets_per_partition;

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

        // Fold bucket by bucket, every source in turn, rather than source by
        // source across the whole partition: one bucket's entries land in a
        // narrow region of the target, so all sources hit that region while
        // it is cache-warm instead of each source refetching the whole
        // target's slot range.
        for bucket in bucket_lo..bucket_hi {
            for source in sources {
                match source {
                    MergeSource::Dense(run) => {
                        let (from, to) = run.bucket_range(bucket, bucket + 1);
                        if from == to {
                            continue;
                        }
                        let reader = run.reader::<N>();
                        merge_run_slice::<N, S, V>(
                            &mut allocator,
                            key_arena,
                            &reader,
                            |index| reader.entry_ptr(index) as *const u8,
                            from,
                            to,
                            &mut target,
                            context,
                        );
                    }
                    MergeSource::Sealed(sealed) => {
                        let (from, to) = sealed.run.bucket_range(bucket, bucket + 1);
                        if from == to {
                            continue;
                        }
                        let reader = sealed.table.reader::<N>();
                        let positions = sealed.run.positions_ptr();
                        merge_run_slice::<N, S, V>(
                            &mut allocator,
                            key_arena,
                            &reader,
                            |index| {
                                let slot = unsafe { *positions.add(index) } as usize;
                                reader.entry_ptr(slot) as *const u8
                            },
                            from,
                            to,
                            &mut target,
                            context,
                        );
                    }
                }
            }
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
    use crate::operations::unary::group::hashtables::{AggregatedTable, DenseRun, SpillConfig};
    use crate::operations::unary::group::hll::Hll;
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

    fn make_worker_run(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
    ) -> MergeSource<i32, CountValue> {
        make_worker_run_with_spill(state, arena, values, SpillConfig::DEFAULT)
    }

    fn make_worker_run_with_spill(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
        spill: SpillConfig,
    ) -> MergeSource<i32, CountValue> {
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
        agg.flush().source
    }

    fn merge_all_partitions(
        runs: &[MergeSource<i32, CountValue>],
        arena: &SharedArena,
    ) -> Vec<(i32, usize)> {
        let total: usize = runs.iter().map(|run| run.len()).sum();
        let partition_cap = (total / PARTITIONS).max(1).next_power_of_two();

        let mut all_entries = vec![];
        for p in 0..PARTITIONS {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                runs,
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
        let runs = vec![make_worker_run(&state, &arena, &[1, 2, 3, 4, 5])];

        let entries = merge_all_partitions(&runs, &arena);

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
        let runs = vec![
            make_worker_run(&state, &arena, &[1, 2, 3]),
            make_worker_run(&state, &arena, &[4, 5, 6]),
        ];

        let entries = merge_all_partitions(&runs, &arena);

        assert_eq!(entries.len(), 6);
    }

    #[test]
    fn two_workers_overlapping_keys_merged() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let runs = vec![
            make_worker_run(&state, &arena, &[1, 2, 3]),
            make_worker_run(&state, &arena, &[2, 3, 4]),
        ];

        let entries = merge_all_partitions(&runs, &arena);

        assert_eq!(entries.len(), 4);
        let count_for = |k: i32| entries.iter().find(|(key, _)| *key == k).unwrap().1;
        assert_eq!(count_for(1), 1);
        assert_eq!(count_for(2), 2);
        assert_eq!(count_for(3), 2);
        assert_eq!(count_for(4), 1);
    }

    #[test]
    fn empty_runs_produce_no_entries() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let runs = vec![make_worker_run(&state, &arena, &[])];

        let entries = merge_all_partitions(&runs, &arena);

        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn many_workers_large_overlap() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let runs: Vec<_> = (0..8)
            .map(|_| make_worker_run(&state, &arena, &[10, 20, 30]))
            .collect();

        let entries = merge_all_partitions(&runs, &arena);

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
        let runs = vec![make_worker_run(&state, &arena, &values)];

        let mut total = 0;
        for p in 0..PARTITIONS {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                &runs,
                DEFAULT_CAPACITY,
                PARTITIONS,
                &arena,
                &COUNT_CFG,
            );
            total += result.iter(0).count();
        }

        assert_eq!(total, 200);
    }

    #[test]
    fn duplicates_within_single_worker_carry_through_merge() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let runs = vec![make_worker_run(&state, &arena, &[5, 5, 5, 5, 5])];

        let entries = merge_all_partitions(&runs, &arena);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], (5, 5));
    }

    #[test]
    fn consolidation_carries_a_workers_stacked_tables() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let spill = SpillConfig {
            spill_capacity: 256,
        };
        // Each key appears twice, spread across the stack the small capacity
        // forces; the merge must combine the per-table partials.
        let values: Vec<i32> = (0..500).chain(0..500).collect();
        let runs = vec![
            make_worker_run_with_spill(&state, &arena, &values, spill),
            make_worker_run_with_spill(&state, &arena, &values, spill),
        ];

        let entries = merge_all_partitions(&runs, &arena);

        assert_eq!(entries.len(), 500);
        assert!(entries.iter().all(|&(_, count)| count == 4));
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
        let node_a = vec![make_worker_run(&state, &arena, &[1, 2, 3])];
        let node_b = vec![make_worker_run(&state, &arena, &[2, 3, 4])];

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

    /// Consolidate hand-built tables of `(hash, key)` pairs into one worker
    /// run, bypassing the extractor so tests control hash collisions and
    /// extreme hash values.
    fn run_from_pair_tables(pair_tables: &[&[(u64, i32)]]) -> MergeSource<i32, CountValue> {
        let mut allocator = crate::memory::SlabAllocator::new(true);
        let mut tables = vec![];
        for pairs in pair_tables {
            let mut table: MultiSlabTable<i32, CountValue> =
                MultiSlabTable::new(&mut allocator, 128, 0, &());
            for &(hash, key) in *pairs {
                table
                    .prober()
                    .merge_from::<false, _>(hash, key, &seeded_count(), &());
            }
            tables.push(table);
        }
        MergeSource::Dense(DenseRun::consolidate(
            &tables,
            &mut allocator,
            (),
            &mut Hll::new(),
        ))
    }

    /// A `CountValue` holding a count of one, as one consumed row seeds it.
    fn seeded_count() -> CountValue {
        let mut value = CountValue::default();
        value.seed(&((),), 0, &mut ());
        value
    }

    fn collect_all_from(runs: &[MergeSource<i32, CountValue>]) -> Vec<(i32, usize)> {
        let arena = SharedArena::new(64);
        let mut entries = vec![];
        for p in 0..PARTITIONS {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                runs,
                DEFAULT_CAPACITY,
                PARTITIONS,
                &arena,
                &COUNT_CFG,
            );
            for entry in result.iter(0) {
                entries.push((*entry.key, entry.stored.sort_key(0) as usize));
            }
        }
        entries.sort();
        entries
    }

    #[test]
    fn same_hash_distinct_keys_stay_distinct_groups() {
        init_test_free_pool(64);
        // Three sources hold hash 42: two with key 1, one with key 2. The
        // merged output must combine the key-1 partials and keep key 2
        // separate.
        let runs = vec![
            run_from_pair_tables(&[&[(42, 1)]]),
            run_from_pair_tables(&[&[(42, 2)]]),
            run_from_pair_tables(&[&[(42, 1)]]),
        ];

        let entries = collect_all_from(&runs);

        assert_eq!(entries, vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn same_hash_partials_within_one_worker_combine() {
        init_test_free_pool(64);
        // One worker's stack holds the same (hash, key) twice plus a
        // colliding distinct key; consolidation keeps all three entries and
        // the fold combines exactly the matching ones.
        let runs = vec![run_from_pair_tables(&[&[(42, 1), (42, 2)], &[(42, 1)]])];

        let entries = collect_all_from(&runs);

        assert_eq!(entries, vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn extreme_hash_values_land_in_the_last_partition() {
        init_test_free_pool(64);
        let runs = vec![
            run_from_pair_tables(&[&[(u64::MAX, 7), (1, 9)]]),
            run_from_pair_tables(&[&[(u64::MAX, 7), (u64::MAX - 1, 8)]]),
        ];

        let entries = collect_all_from(&runs);

        assert_eq!(entries, vec![(7, 2), (8, 1), (9, 1)]);
    }

    #[test]
    fn runs_group_wrapped_entries_into_their_hash_buckets() {
        init_test_free_pool(64);
        // Hashes near u64::MAX probe past the last slot and wrap to slot 0;
        // the consolidated run must still deliver every entry inside its hash
        // bucket, in ascending bucket order.
        let pairs: Vec<(u64, i32)> = (0..40)
            .map(|i| (u64::MAX - i as u64, i as i32))
            .chain((1..40).map(|i| (i as u64, 100 + i as i32)))
            .collect();
        let MergeSource::Dense(run) = run_from_pair_tables(&[&pairs]) else {
            unreachable!("pair tables consolidate into a dense run");
        };

        let reader = run.reader::<0>();
        let mut previous_bucket = 0u64;
        for i in 0..run.len() {
            let bucket = reader.hash_of(reader.entry_ptr(i)) >> (64 - BUCKET_BITS);
            assert!(
                bucket >= previous_bucket,
                "run out of bucket order at {}",
                i
            );
            previous_bucket = bucket;
        }
        assert_eq!(run.len(), pairs.len());
    }

    /// The merge routes by run buckets at `num_partitions` granularity, which
    /// can be finer than a small table's slot count ever resolved. Each key
    /// must still land in exactly one partition with the right count.
    #[test]
    fn bucket_merge_finer_than_table() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let n = 200i32;
        let values: Vec<i32> = (0..n).chain(0..n).collect(); // each key twice
        let runs = vec![make_worker_run(&state, &arena, &values)];
        let num_partitions = 256;
        let mut occurrences = vec![0usize; n as usize];
        let mut counts = vec![0usize; n as usize];
        for p in 0..num_partitions {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                &runs,
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
