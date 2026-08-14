//! Partition-parallel merge of GROUP BY partial results.
//!
//! High hash bits select the merge partition, and every worker's flush hands
//! the merge one bucket-grouped source, so a partition's share of each worker
//! is one contiguous slice. Each merge job folds its slices into a table of
//! **references**: a slot names the first source entry that carried its
//! group, and only when a second partial for the same group arrives does the
//! slot get a scratch accumulator to combine values in. Groups held by one
//! source entry, the overwhelming majority at high cardinality, are never
//! copied: the output reads them straight out of the worker's run.
//!
//! ```text
//! run slices --> reference table --> output
//!                 (hash, entry)        |-- single partial: emit *entry
//!                 (hash, entry, acc)   |-- repeated:       emit *acc
//! ```
//!
//! No slot is ever tested for occupancy or partition membership, and folding
//! proceeds bucket range by bucket range across all sources so one range's
//! reference slots stay cache-warm while every source hits them.

use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, BUCKET_BITS, DEFAULT_CAPACITY, LiveKey, MergeSource, PersistedKey, Stub,
    TableReader, prefetch_l1_line,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::values::ArityBody;

use super::Result;

/// Collision-to-entry ratio at which the reference table doubles.
///
/// For linear probing, the cumulative collision ratio is
/// `alpha / (2 * (1 - alpha))`. A threshold of two allows dense tables
/// without letting probe chains grow without bound.
const RESIZE_COLLISION_RATIO: usize = 2;

/// Entries gathered per fold batch. The gather pass reads each batch's
/// entries (sequential in the source) and issues their reference-slot
/// prefetches together, so by the time the fold pass probes a slot line it
/// has had a whole batch of lead time.
const FOLD_BATCH: usize = 48;

/// One group's reference: the first source entry that carried it, and the
/// scratch accumulator combining its partials once a repeat arrived.
#[derive(Clone, Copy)]
struct RefSlot {
    /// The stored hash (never 0 for an occupied slot).
    hash: u64,
    /// First source entry for this group; key and value offsets apply.
    entry: *const u8,
    /// Scratch value storage, null until a second partial arrives.
    merged: *mut u8,
}

impl RefSlot {
    const EMPTY: RefSlot = RefSlot {
        hash: 0,
        entry: std::ptr::null(),
        merged: std::ptr::null_mut(),
    };
}

/// Open-addressing table of group references for one merge partition.
///
/// Slot placement mirrors the source tables: the hash's high bits past the
/// partition prefix pick the slot, so the bucket-ordered fold sweeps the
/// slots left to right.
struct RefTable {
    slots: Vec<RefSlot>,
    slot_mask: usize,
    slot_shift: u32,
    /// Strips the partition prefix so the following bits drive placement.
    hash_left_shift: u32,
    len: usize,
    collisions: usize,
}

impl RefTable {
    fn new(capacity: usize, hash_left_shift: u32) -> Self {
        Self {
            slots: vec![RefSlot::EMPTY; capacity],
            slot_mask: capacity - 1,
            slot_shift: u64::BITS - capacity.trailing_zeros(),
            hash_left_shift,
            len: 0,
            collisions: 0,
        }
    }

    #[inline(always)]
    fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.hash_left_shift) >> self.slot_shift) as usize
    }

    /// Warms the slot lines an entry with this hash will probe.
    #[inline(always)]
    fn prefetch(&self, hash: u64) {
        let idx = self.slot_for(hash);
        let ptr = unsafe { self.slots.as_ptr().add(idx) } as *const u8;
        prefetch_l1_line(ptr);
        prefetch_l1_line(ptr.wrapping_add(64));
    }

    /// Doubles the table once collision pressure exceeds the ratio.
    #[cold]
    fn grow(&mut self) {
        let capacity = (self.slot_mask + 1) * 2;
        let old = std::mem::replace(&mut self.slots, vec![RefSlot::EMPTY; capacity]);
        self.slot_mask = capacity - 1;
        self.slot_shift = u64::BITS - capacity.trailing_zeros();
        // Post-resize collision tracking reflects only new probing pressure.
        self.collisions = self.len;
        for slot in old {
            if slot.hash == 0 {
                continue;
            }
            let mut idx = self.slot_for(slot.hash);
            while self.slots[idx].hash != 0 {
                idx = (idx + 1) & self.slot_mask;
            }
            self.slots[idx] = slot;
        }
    }

    #[inline(always)]
    fn resize_if_needed(&mut self) {
        if self.collisions > self.len * RESIZE_COLLISION_RATIO {
            self.grow();
        }
    }

    /// Empties the table for the next hash range, keeping its allocation
    /// (and any growth the previous range forced).
    fn reset(&mut self) {
        self.slots.fill(RefSlot::EMPTY);
        self.len = 0;
        self.collisions = 0;
    }
}

/// Bump storage for merged values, with stable addresses.
///
/// Chunks are never reallocated, so a cell pointer handed out stays valid
/// for the arena's lifetime.
struct ScratchValues {
    chunks: Vec<Vec<u8>>,
    /// Chunk currently allocated from; earlier chunks are full (or retired
    /// by a reset).
    current: usize,
    cursor: usize,
    cell: usize,
    align: usize,
    chunk_bytes: usize,
}

impl ScratchValues {
    fn new(value_size: usize, align: usize) -> Self {
        let cell = value_size.max(1).next_multiple_of(align);
        Self {
            chunks: Vec::new(),
            current: 0,
            cursor: 0,
            cell,
            align,
            chunk_bytes: (cell * 1024).max(64 * 1024),
        }
    }

    /// Hands out one aligned value cell. The caller fully initialises it
    /// with [`AggregationValue::copy_from`], so reused cells need no wipe.
    fn alloc(&mut self) -> *mut u8 {
        loop {
            if let Some(chunk) = self.chunks.get_mut(self.current) {
                let base = chunk.as_mut_ptr() as usize;
                let aligned = (base + self.cursor).next_multiple_of(self.align);
                let offset = aligned - base;
                if offset + self.cell <= chunk.len() {
                    self.cursor = offset + self.cell;
                    return aligned as *mut u8;
                }
                self.current += 1;
                self.cursor = 0;
                continue;
            }
            self.chunks.push(vec![0u8; self.chunk_bytes + self.align]);
        }
    }

    /// Makes every cell reusable for the next hash range. Handed-out
    /// pointers become dangling, so the previous range must be fully
    /// emitted first.
    fn reset(&mut self) {
        self.current = 0;
        self.cursor = 0;
    }
}

/// Streams one partition's merged groups into `emit`.
///
/// `emit` receives each group exactly once, as a key reference and the
/// group's value (in a source run for single partials, in scratch for
/// combined groups), and returns whether the merge should keep going
/// (`false` stops early, for satisfied LIMIT pushdowns).
pub(super) fn merge_partition<S: StoredKey, V: AggregationValue + ?Sized, E>(
    partition: usize,
    num_partitions: usize,
    sources: &[MergeSource<S::Persisted, V>],
    partition_capacity: usize,
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
            sources,
            partition_capacity,
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
    sources: &'a [MergeSource<S::Persisted, V>],
    partition_capacity: usize,
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
            sources,
            partition_capacity,
            key_arena,
            context,
            emit,
        } = self;
        merge_partition_rows::<N, S, V, E>(
            partition,
            num_partitions,
            sources,
            partition_capacity,
            key_arena,
            context,
            emit,
        )
    }
}

/// Folds one source slice `[from, to)` into the reference table.
///
/// `entry_at` maps a slice index to its entry address: direct indexing for a
/// dense run, slot indirection for a sealed table. Each batch is gathered
/// first (reading the source close to sequentially and prefetching every
/// reference slot it will probe), then folded, so the fold's probes land on
/// lines whose fetch started a batch earlier.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fold_slice<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    reader: &TableReader<'_, S::Persisted, V>,
    entry_at: impl Fn(usize) -> *const u8,
    from: usize,
    to: usize,
    table: &mut RefTable,
    scratch: &mut ScratchValues,
    key_arena: &SharedArena,
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
            table.prefetch(hash);
            if <S::Persisted as PersistedKey>::HAS_BLOB {
                unsafe { reader.key_of(entry) }.prefetch_blob(key_arena);
            }
        }
        for &(entry, hash) in &batch[..batch_len] {
            fold_entry::<N, S, V>(reader, entry, hash, table, scratch, key_arena, context);
            table.resize_if_needed();
        }
        i += batch_len;
    }
}

/// Folds one contiguous run of routing stubs into the reference table.
///
/// The stub stream already carries each entry's hash, so nothing reads the
/// source tables here; each batch first warms the reference slots it will
/// probe and the entries it references (for the rare key compare, and for
/// the emit that follows this range), then folds.
#[inline(always)]
fn fold_stubs<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    reader: &TableReader<'_, S::Persisted, V>,
    stubs: &[Stub],
    table: &mut RefTable,
    scratch: &mut ScratchValues,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) {
    let mut i = 0;
    while i < stubs.len() {
        let batch_len = (stubs.len() - i).min(FOLD_BATCH);
        for stub in &stubs[i..i + batch_len] {
            table.prefetch(stub.hash);
            prefetch_l1_line(stub.entry);
        }
        for stub in &stubs[i..i + batch_len] {
            fold_entry::<N, S, V>(
                reader, stub.entry, stub.hash, table, scratch, key_arena, context,
            );
            table.resize_if_needed();
        }
        i += batch_len;
    }
}

/// Inserts one source entry: seeds an empty slot with a reference, or folds
/// the value into the matching group's scratch accumulator.
#[inline(always)]
fn fold_entry<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    reader: &TableReader<'_, S::Persisted, V>,
    entry: *const u8,
    hash: u64,
    table: &mut RefTable,
    scratch: &mut ScratchValues,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) {
    let metadata = reader.metadata();
    let mut idx = table.slot_for(hash);
    loop {
        let slot = table.slots[idx];
        if slot.hash == 0 {
            table.slots[idx] = RefSlot {
                hash,
                entry,
                merged: std::ptr::null_mut(),
            };
            table.len += 1;
            return;
        }
        if slot.hash == hash {
            let incoming_key = unsafe { reader.key_of(entry) };
            let stored_key = unsafe { reader.key_of(slot.entry) };
            let live = S::resolve_persisted(key_arena, *incoming_key);
            if live.eq_persisted(stored_key) {
                let merged = if slot.merged.is_null() {
                    let cell = scratch.alloc();
                    let first = unsafe { reader.value_of(slot.entry) };
                    unsafe { V::from_entry_mut(cell, metadata) }.copy_from(first);
                    table.slots[idx].merged = cell;
                    cell
                } else {
                    slot.merged
                };
                let incoming_value = unsafe { reader.value_of(entry) };
                unsafe { V::from_entry_mut(merged, metadata) }.merge_from(incoming_value, context);
                return;
            }
        }
        table.collisions += 1;
        idx = (idx + 1) & table.slot_mask;
    }
}

/// Merges one partition after arity dispatch.
///
/// Separate reference parameters preserve alias information across raw slot
/// writes, keeping shared state outside the inner loops.
#[allow(clippy::too_many_arguments)]
fn merge_partition_rows<const N: usize, S: StoredKey, V: AggregationValue + ?Sized, E>(
    partition: usize,
    num_partitions: usize,
    sources: &[MergeSource<S::Persisted, V>],
    partition_capacity: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    mut emit: E,
) -> Result<()>
where
    E: FnMut(&S::Persisted, &V) -> Result<bool>,
{
    let partition_bits = num_partitions.trailing_zeros();
    let min_source_bits = sources
        .iter()
        .map(|source| source.bucket_bits())
        .min()
        .unwrap_or(BUCKET_BITS);
    debug_assert!(partition_bits <= min_source_bits);
    // Global bucket units, at the finest resolution any run can have.
    let partition_width = 1usize << (BUCKET_BITS - partition_bits);
    let bucket_lo = partition * partition_width;

    let metadata = if N == 0 {
        V::storage_metadata(context)
    } else {
        V::metadata_for_arity::<N>()
    };
    let capacity = partition_capacity.max(DEFAULT_CAPACITY);
    // The partition is folded and emitted one hash range at a time: a
    // range's groups are complete once every source folded its slice, so
    // they are emitted immediately, while the reference slots and the
    // entries they point at are still cache-warm. One range-sized table is
    // reused (reset, not reallocated) across ranges, so the merge's working
    // table stays cache-resident by construction however large the
    // partition is. The step count follows the partition's footprint (small
    // partitions take one pass) and never splits finer than any source
    // resolves.
    const STEP_TARGET_BYTES: usize = 32 * 1024;
    let target_bytes = capacity * size_of::<RefSlot>();
    let desired_steps = (target_bytes / STEP_TARGET_BYTES)
        .max(1)
        .next_power_of_two();
    let max_steps = 1usize << (min_source_bits - partition_bits);
    let steps = desired_steps.min(max_steps);
    let step_width = partition_width / steps;
    let step_capacity = (capacity / steps).max(DEFAULT_CAPACITY);
    let step_bits = partition_bits + steps.trailing_zeros();
    let mut table = RefTable::new(step_capacity, step_bits);
    let mut scratch = ScratchValues::new(V::stored_size(metadata), V::stored_align());
    let Some(layout) = sources.first().map(|source| match source {
        MergeSource::Stubs { tables, .. } => tables[0].reader::<N>(),
        MergeSource::Sealed(sealed) => sealed.table.reader::<N>(),
    }) else {
        return Ok(());
    };
    for step in 0..steps {
        let lo = bucket_lo + step * step_width;
        let hi = lo + step_width;
        for source in sources {
            match source {
                MergeSource::Stubs { tables, run } => {
                    let (from, to) = run.bucket_range(lo, hi);
                    if from == to {
                        continue;
                    }
                    let reader = tables[0].reader::<N>();
                    run.slices(from, to, |stubs| {
                        fold_stubs::<N, S, V>(
                            &reader,
                            stubs,
                            &mut table,
                            &mut scratch,
                            key_arena,
                            context,
                        );
                    });
                }
                MergeSource::Sealed(sealed) => {
                    let (from, to) = sealed.run.bucket_range(lo, hi);
                    if from == to {
                        continue;
                    }
                    let reader = sealed.table.reader::<N>();
                    let positions = sealed.run.positions_ptr();
                    fold_slice::<N, S, V>(
                        &reader,
                        |index| {
                            let slot = unsafe { *positions.add(index) } as usize;
                            reader.entry_ptr(slot) as *const u8
                        },
                        from,
                        to,
                        &mut table,
                        &mut scratch,
                        key_arena,
                        context,
                    );
                }
            }
        }
        // Emit this range's groups while everything is hot: single partials
        // straight from their source entry, combined groups from their
        // scratch accumulator. Any source reader can decode any entry
        // address; they all share one layout.
        for slot in &table.slots {
            if slot.hash == 0 {
                continue;
            }
            let key = unsafe { layout.key_of(slot.entry) };
            let value = if slot.merged.is_null() {
                unsafe { layout.value_of(slot.entry) }
            } else {
                unsafe { V::from_entry(slot.merged, metadata) }
            };
            if !emit(key, value)? {
                return Ok(());
            }
        }
        table.reset();
        scratch.reset();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::PARTITIONS;
    use super::*;
    use crate::RECORD_BATCH_SIZE;
    use crate::memory::{SlabAllocator, init_test_free_pool};
    use crate::operations::unary::group::arena::SharedArena;
    use crate::operations::unary::group::hashtables::{
        AggregatedTable, MultiSlabTable, SpillConfig, StubRun,
    };
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

    fn collect_partition(
        sources: &[MergeSource<i32, CountValue>],
        arena: &SharedArena,
        partition: usize,
        num_partitions: usize,
    ) -> Vec<(i32, usize)> {
        let mut groups = vec![];
        merge_partition::<IntStored, CountValue, _>(
            partition,
            num_partitions,
            sources,
            DEFAULT_CAPACITY,
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
        sources: &[MergeSource<i32, CountValue>],
        arena: &SharedArena,
    ) -> Vec<(i32, usize)> {
        let mut all_entries = vec![];
        for p in 0..PARTITIONS {
            all_entries.extend(collect_partition(sources, arena, p, PARTITIONS));
        }
        all_entries.sort_by_key(|(k, _)| *k);
        all_entries
    }

    #[test]
    fn single_worker_all_entries_preserved() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let sources = vec![make_worker_run(&state, &arena, &[1, 2, 3, 4, 5])];

        let entries = merge_all_partitions(&sources, &arena);

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
        let sources = vec![
            make_worker_run(&state, &arena, &[1, 2, 3]),
            make_worker_run(&state, &arena, &[4, 5, 6]),
        ];

        let entries = merge_all_partitions(&sources, &arena);

        assert_eq!(entries.len(), 6);
    }

    #[test]
    fn two_workers_overlapping_keys_merged() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let sources = vec![
            make_worker_run(&state, &arena, &[1, 2, 3]),
            make_worker_run(&state, &arena, &[2, 3, 4]),
        ];

        let entries = merge_all_partitions(&sources, &arena);

        assert_eq!(entries.len(), 4);
        let count_for = |k: i32| entries.iter().find(|(key, _)| *key == k).unwrap().1;
        assert_eq!(count_for(1), 1);
        assert_eq!(count_for(2), 2);
        assert_eq!(count_for(3), 2);
        assert_eq!(count_for(4), 1);
    }

    #[test]
    fn three_workers_same_key_combine_into_one_group() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let sources = vec![
            make_worker_run(&state, &arena, &[7, 7]),
            make_worker_run(&state, &arena, &[7]),
            make_worker_run(&state, &arena, &[7, 7, 7]),
        ];

        let entries = merge_all_partitions(&sources, &arena);

        assert_eq!(entries, vec![(7, 6)]);
    }

    #[test]
    fn empty_sources_produce_no_entries() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let sources = vec![make_worker_run(&state, &arena, &[])];

        let entries = merge_all_partitions(&sources, &arena);

        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn many_workers_large_overlap() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let sources: Vec<_> = (0..8)
            .map(|_| make_worker_run(&state, &arena, &[10, 20, 30]))
            .collect();

        let entries = merge_all_partitions(&sources, &arena);

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
        let sources = vec![make_worker_run(&state, &arena, &values)];

        let mut total = 0;
        for p in 0..PARTITIONS {
            total += collect_partition(&sources, &arena, p, PARTITIONS).len();
        }

        assert_eq!(total, 200);
    }

    #[test]
    fn duplicates_within_single_worker_carry_through_merge() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let sources = vec![make_worker_run(&state, &arena, &[5, 5, 5, 5, 5])];

        let entries = merge_all_partitions(&sources, &arena);

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
        let sources = vec![
            make_worker_run_with_spill(&state, &arena, &values, spill),
            make_worker_run_with_spill(&state, &arena, &values, spill),
        ];

        let entries = merge_all_partitions(&sources, &arena);

        assert_eq!(entries.len(), 500);
        assert!(entries.iter().all(|&(_, count)| count == 4));
    }

    /// Consolidate hand-built tables of `(hash, key)` pairs into one worker
    /// run, bypassing the extractor so tests control hash collisions and
    /// extreme hash values.
    fn run_from_pair_tables(pair_tables: &[&[(u64, i32)]]) -> MergeSource<i32, CountValue> {
        let mut allocator = SlabAllocator::new(true);
        let mut tables = vec![];
        for pairs in pair_tables {
            let mut table: MultiSlabTable<i32, CountValue> =
                MultiSlabTable::new(&mut allocator, 128, 0, &());
            for &(hash, key) in *pairs {
                table.prober().merge_from(hash, key, &seeded_count(), &());
            }
            tables.push(table);
        }
        let run = StubRun::consolidate(&tables, &mut allocator, &mut Hll::new());
        MergeSource::Stubs { tables, run }
    }

    /// A `CountValue` holding a count of one, as one consumed row seeds it.
    fn seeded_count() -> CountValue {
        let mut value = CountValue::default();
        value.seed(&((),), 0, &mut ());
        value
    }

    fn collect_all_from(sources: &[MergeSource<i32, CountValue>]) -> Vec<(i32, usize)> {
        let arena = SharedArena::new(64);
        let mut entries = vec![];
        for p in 0..PARTITIONS {
            entries.extend(collect_partition(sources, &arena, p, PARTITIONS));
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
        let sources = vec![
            run_from_pair_tables(&[&[(42, 1)]]),
            run_from_pair_tables(&[&[(42, 2)]]),
            run_from_pair_tables(&[&[(42, 1)]]),
        ];

        let entries = collect_all_from(&sources);

        assert_eq!(entries, vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn same_hash_partials_within_one_worker_combine() {
        init_test_free_pool(64);
        // One worker's stack holds the same (hash, key) twice plus a
        // colliding distinct key; consolidation keeps all three entries and
        // the fold combines exactly the matching ones.
        let sources = vec![run_from_pair_tables(&[&[(42, 1), (42, 2)], &[(42, 1)]])];

        let entries = collect_all_from(&sources);

        assert_eq!(entries, vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn extreme_hash_values_land_in_the_last_partition() {
        init_test_free_pool(64);
        let sources = vec![
            run_from_pair_tables(&[&[(u64::MAX, 7), (1, 9)]]),
            run_from_pair_tables(&[&[(u64::MAX, 7), (u64::MAX - 1, 8)]]),
        ];

        let entries = collect_all_from(&sources);

        assert_eq!(entries, vec![(7, 2), (8, 1), (9, 1)]);
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
        let sources = vec![make_worker_run(&state, &arena, &values)];
        let num_partitions = 256;
        let mut occurrences = vec![0usize; n as usize];
        let mut counts = vec![0usize; n as usize];
        for p in 0..num_partitions {
            for (key, count) in collect_partition(&sources, &arena, p, num_partitions) {
                occurrences[key as usize] += 1;
                counts[key as usize] += count;
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
