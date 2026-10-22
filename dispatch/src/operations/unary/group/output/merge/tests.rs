use super::*;
use crate::operations::unary::group::hashtables::PartitionBuffers;

/// Merge one partition the way a job does, from resident sources only.
fn merge_partition(
    partition: usize,
    buffers: &[PartitionBuffers<i32, CountValue>],
    tables: &[HashTable<i32, CountValue>],
    partition_capacity: usize,
    num_partitions: usize,
    arena: &SharedArena,
) -> HashTable<i32, CountValue> {
    let partition_bits = num_partitions.trailing_zeros();
    let mut allocator = SlabAllocator::new(true);
    let mut capacity = partition_capacity.max(DEFAULT_CAPACITY);
    let mut table = HashTable::new(&mut allocator, capacity, partition_bits, &COUNT_CFG);
    merge_scatter_rows::<IntStored, CountValue>(
        &mut table,
        &mut allocator,
        &mut capacity,
        partition,
        buffers,
        num_partitions,
        arena,
        &COUNT_CFG,
        None,
    );
    merge_tables::<IntStored, CountValue>(
        &mut table,
        &mut allocator,
        partition,
        partition_bits,
        tables,
        arena,
        &COUNT_CFG,
        None,
    );
    table
}
use crate::RECORD_BATCH_SIZE;
use crate::memory::{PartitionEvictor, SharedPartitionEvictor, init_test_free_pool};
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

/// The fan-out these tests merge at. Production sizes its partition count
/// per query; the merge itself only needs a power of two.
const PARTITIONS: usize = 64;

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
) -> Vec<HashTable<i32, CountValue>> {
    let mut agg = AggregatedTable::<IntExtractor, CountValue>::new(
        state.clone(),
        arena.clone(),
        (),
        (),
        RadixConfig::DEFAULT,
        None,
        fresh_evictor(),
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
    let out = agg.flush().0;
    assert!(out.buffers.is_none(), "low-cardinality test stays in-place");
    out.tables
}

fn merge_all_partitions(
    tables: &[HashTable<i32, CountValue>],
    arena: &SharedArena,
) -> Vec<(i32, usize)> {
    let total_cap: usize = tables.iter().map(|t| t.capacity()).sum::<usize>() / 2;
    let partition_cap = (total_cap / PARTITIONS).max(1).next_power_of_two();

    let mut all_entries = vec![];
    for p in 0..PARTITIONS {
        let result = merge_partition(p, &[], tables, partition_cap, PARTITIONS, arena);
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
        let result = merge_partition(p, &[], &tables, partition_cap, PARTITIONS, &arena);
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
/// `merge_partition` result for the same partition) into one final table;
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
            merge_partition(p, &[], &node_a, 4, PARTITIONS, &arena),
            merge_partition(p, &[], &node_b, 4, PARTITIONS, &arena),
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

fn fresh_evictor() -> SharedPartitionEvictor {
    std::rc::Rc::new(std::cell::RefCell::new(PartitionEvictor::new()))
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
        None,
        fresh_evictor(),
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
    agg.flush().0
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
        let result = merge_partition(
            p,
            &[],
            &out.tables,
            DEFAULT_CAPACITY,
            num_partitions,
            &arena,
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
