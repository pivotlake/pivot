//! Two-level streaming merge of GROUP BY partial results.
//!
//! The merge's cost is governed by its fan-in: every emitted entry pops a
//! loser tree and touches the winning stream's current cache line, so a merge
//! over thousands of streams misses on almost every pop. Bounding the fan-in
//! at both levels keeps every pop inside the cache hierarchy:
//!
//! 1. **Worker consolidation** ([`consolidate_worker_runs`]): at flush, each
//!    worker merges its own stack of sealed tables (however tall the input's
//!    cardinality made it) into one [`DenseRun`], deduplicating its own
//!    repeats. The tree and all stream heads are worker-local and
//!    cache-resident, reads follow each table in slot order, and the output
//!    is written sequentially.
//! 2. **Partition merge** ([`merge_partition`]): each partition job merges
//!    one run per worker, so its fan-in is the worker count. Groups stream
//!    straight into the output sink; no result table is ever built.
//!
//! ```text
//! worker tables ==(flush)==> dense run 0 --+
//!                            dense run 1 --+--> loser tree --> sink
//!                            dense run 2 --+
//! ```
//!
//! A group held by exactly one stream (the common case at high cardinality)
//! is emitted directly from its entry, with no copies and no value merge.
//! When several streams hold the same hash, the entries are gathered and
//! grouped by their real key, so two distinct keys sharing a 64-bit hash stay
//! two groups.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, BUCKET_BITS, DenseRun, DenseRunBuilder, LiveKey, PersistedKey, SealedTable,
    TableReader,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::values::ArityBody;

use super::Result;

/// How many entries ahead each cursor warms its stream's cache lines.
const CURSOR_PREFETCH: usize = 8;

/// An exhausted cursor's tree key. A live entry can carry this same hash, so
/// reaching a `u64::MAX` winner falls into the tail drain, which collects the
/// real entries the spent cursors are shadowing.
const EXHAUSTED: u64 = u64::MAX;

/// One hash-ordered stream of entries feeding the loser tree.
///
/// Every method is monomorphized into the merge loop; the trait only exists
/// so tables (slot indirection through a sorted run) and dense runs (direct
/// indexing) share the merge core.
trait MergeCursor {
    /// Address of the entry the cursor currently points at.
    fn current(&self) -> *const u8;

    /// The current entry's stored hash.
    fn current_hash(&self) -> u64;

    /// Steps past the current entry; returns the next hash, or [`EXHAUSTED`].
    fn advance(&mut self) -> u64;

    /// Warms the first few entries, before the merge starts pulling.
    fn prefetch_head(&self);

    /// Hands every not-yet-consumed entry to `take`, in order.
    fn for_each_remaining(&self, take: impl FnMut(*const u8));
}

/// A cursor over one sealed table's sorted-run slice.
struct TableCursor<'t, KP, V: AggregationValue + ?Sized> {
    reader: TableReader<'t, KP, V>,
    positions: *const u32,
    current: *const u8,
    pos: usize,
    end: usize,
}

impl<'t, KP: PersistedKey, V: AggregationValue + ?Sized> TableCursor<'t, KP, V> {
    fn new(reader: TableReader<'t, KP, V>, positions: *const u32, pos: usize, end: usize) -> Self {
        let slot = unsafe { *positions.add(pos) } as usize;
        let current = reader.entry_ptr(slot);
        Self {
            reader,
            positions,
            current,
            pos,
            end,
        }
    }
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> MergeCursor for TableCursor<'_, KP, V> {
    #[inline(always)]
    fn current(&self) -> *const u8 {
        self.current
    }

    #[inline(always)]
    fn current_hash(&self) -> u64 {
        self.reader.hash_of(self.current)
    }

    #[inline(always)]
    fn advance(&mut self) -> u64 {
        self.pos += 1;
        if self.pos == self.end {
            return EXHAUSTED;
        }
        if self.pos + CURSOR_PREFETCH < self.end {
            let ahead = unsafe { *self.positions.add(self.pos + CURSOR_PREFETCH) } as usize;
            self.reader.prefetch_entry(ahead);
        }
        let slot = unsafe { *self.positions.add(self.pos) } as usize;
        self.current = self.reader.entry_ptr(slot);
        self.reader.hash_of(self.current)
    }

    #[inline(always)]
    fn prefetch_head(&self) {
        let lookahead = (self.end - self.pos).min(CURSOR_PREFETCH);
        for k in self.pos..self.pos + lookahead {
            let slot = unsafe { *self.positions.add(k) } as usize;
            self.reader.prefetch_entry(slot);
        }
    }

    fn for_each_remaining(&self, mut take: impl FnMut(*const u8)) {
        for pos in self.pos..self.end {
            let slot = unsafe { *self.positions.add(pos) } as usize;
            take(self.reader.entry_ptr(slot));
        }
    }
}

/// A cursor over one dense run's partition slice.
struct DenseCursor<'t, KP, V: AggregationValue + ?Sized> {
    reader: TableReader<'t, KP, V>,
    current: *const u8,
    pos: usize,
    end: usize,
}

impl<'t, KP: PersistedKey, V: AggregationValue + ?Sized> DenseCursor<'t, KP, V> {
    fn new(reader: TableReader<'t, KP, V>, pos: usize, end: usize) -> Self {
        let current = reader.entry_ptr(pos);
        Self {
            reader,
            current,
            pos,
            end,
        }
    }
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> MergeCursor for DenseCursor<'_, KP, V> {
    #[inline(always)]
    fn current(&self) -> *const u8 {
        self.current
    }

    #[inline(always)]
    fn current_hash(&self) -> u64 {
        self.reader.hash_of(self.current)
    }

    #[inline(always)]
    fn advance(&mut self) -> u64 {
        self.pos += 1;
        if self.pos == self.end {
            return EXHAUSTED;
        }
        if self.pos + CURSOR_PREFETCH < self.end {
            self.reader.prefetch_entry(self.pos + CURSOR_PREFETCH);
        }
        self.current = self.reader.entry_ptr(self.pos);
        self.reader.hash_of(self.current)
    }

    #[inline(always)]
    fn prefetch_head(&self) {
        let lookahead = (self.end - self.pos).min(CURSOR_PREFETCH);
        for pos in self.pos..self.pos + lookahead {
            self.reader.prefetch_entry(pos);
        }
    }

    fn for_each_remaining(&self, mut take: impl FnMut(*const u8)) {
        for pos in self.pos..self.end {
            take(self.reader.entry_ptr(pos));
        }
    }
}

/// A loser tree over the cursors' current hashes.
///
/// Internal nodes hold the loser of their subtree's play-off and the root
/// holds the overall winner, so replacing the winner's key replays exactly one
/// leaf-to-root path: log2(leaves) comparisons per emitted entry.
struct LoserTree {
    /// Current hash per leaf, padded to a power of two with [`EXHAUSTED`].
    keys: Vec<u64>,
    /// `nodes[0]` is the winner; `nodes[1..]` hold each play-off's loser.
    nodes: Vec<u32>,
    leaves: usize,
}

impl LoserTree {
    fn new(keys: Vec<u64>) -> Self {
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
    fn key(&self, leaf: usize) -> u64 {
        self.keys[leaf]
    }

    /// Replaces the winner leaf's key and replays its path to the root.
    #[inline(always)]
    fn replay(&mut self, leaf: usize, new_key: u64) {
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

/// Receives each merged group exactly once.
///
/// Both methods return whether the merge should keep going (`false` stops
/// early, for satisfied LIMIT pushdowns).
trait MergeSink<KP, V: AggregationValue + ?Sized> {
    /// A group whose only partial is the entry at `entry`.
    fn singleton(&mut self, hash: u64, entry: *const u8) -> Result<bool>;

    /// A group combined from several partials into `value`.
    fn merged(&mut self, hash: u64, key: &KP, value: &V) -> Result<bool>;
}

/// Streams the merge of `cursors` into `sink`.
///
/// The shared core behind worker consolidation and the partition merge:
/// cursors must produce entries in ascending hash order, and every group is
/// handed to the sink exactly once.
fn merge_streams<C, S, V>(
    mut cursors: Vec<C>,
    layout: &TableReader<'_, S::Persisted, V>,
    metadata: V::StorageMetadata,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    sink: &mut impl MergeSink<S::Persisted, V>,
) -> Result<()>
where
    C: MergeCursor,
    S: StoredKey,
    V: AggregationValue + ?Sized,
{
    if cursors.is_empty() {
        return Ok(());
    }
    let leaves = cursors.len().next_power_of_two();
    let mut keys = vec![EXHAUSTED; leaves];
    for (i, cursor) in cursors.iter().enumerate() {
        // Warm the head of each stream; the per-advance prefetch takes over
        // from here.
        cursor.prefetch_head();
        keys[i] = cursor.current_hash();
    }
    let mut tree = LoserTree::new(keys);

    let mut scratch = ScratchValue::<V>::new(metadata);
    let mut cluster: Vec<*const u8> = Vec::new();

    loop {
        let winner = tree.winner();
        let hash = tree.key(winner);
        if hash == EXHAUSTED {
            // Spent cursors and real `u64::MAX` hashes share this key, and
            // the tree cannot tell them apart. The minimum being `u64::MAX`
            // means every remaining live entry carries that hash, so one tail
            // walk over the cursors finishes the merge.
            cluster.clear();
            for cursor in &cursors {
                cursor.for_each_remaining(|entry| cluster.push(entry));
            }
            if cluster.is_empty() {
                return Ok(());
            }
            emit_cluster::<S, V>(
                &mut cluster,
                EXHAUSTED,
                layout,
                &mut scratch,
                key_arena,
                context,
                sink,
            )?;
            return Ok(());
        }
        let entry = cursors[winner].current();
        let next_key = cursors[winner].advance();
        tree.replay(winner, next_key);

        if tree.key(tree.winner()) != hash {
            // Only one stream holds this hash: emit straight from the entry.
            if !sink.singleton(hash, entry)? {
                return Ok(());
            }
            continue;
        }

        // Several entries share this hash. Gather them, then group by the
        // real key: distinct keys colliding on a hash stay distinct groups.
        // Every cursor whose key matches is live: `hash` is below
        // `EXHAUSTED` here.
        cluster.clear();
        cluster.push(entry);
        while tree.key(tree.winner()) == hash {
            let runner_up = tree.winner();
            cluster.push(cursors[runner_up].current());
            let key = cursors[runner_up].advance();
            tree.replay(runner_up, key);
        }
        if !emit_cluster::<S, V>(
            &mut cluster,
            hash,
            layout,
            &mut scratch,
            key_arena,
            context,
            sink,
        )? {
            return Ok(());
        }
    }
}

/// Groups one hash cluster's entries by their real key and emits each group.
///
/// Returns whether the caller should keep merging.
fn emit_cluster<S: StoredKey, V: AggregationValue + ?Sized>(
    cluster: &mut Vec<*const u8>,
    hash: u64,
    layout: &TableReader<'_, S::Persisted, V>,
    scratch: &mut ScratchValue<V>,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    sink: &mut impl MergeSink<S::Persisted, V>,
) -> Result<bool> {
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
        let keep_going = if merged {
            sink.merged(hash, first_view.key, scratch.value())?
        } else {
            sink.singleton(hash, first)?
        };
        if !keep_going {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Sink adapter handing each group to a caller closure as `(key, value)`.
struct EmitSink<'a, KP, V: AggregationValue + ?Sized, E> {
    layout: TableReader<'a, KP, V>,
    emit: E,
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized, E> MergeSink<KP, V> for EmitSink<'_, KP, V, E>
where
    E: FnMut(&KP, &V) -> Result<bool>,
{
    #[inline(always)]
    fn singleton(&mut self, hash: u64, entry: *const u8) -> Result<bool> {
        let view = unsafe { self.layout.view_of(entry, hash) };
        (self.emit)(view.key, view.stored)
    }

    #[inline(always)]
    fn merged(&mut self, _hash: u64, key: &KP, value: &V) -> Result<bool> {
        (self.emit)(key, value)
    }
}

/// Sink adapter appending each group to a worker's dense run.
struct AppendSink<'a, KP: PersistedKey, V: AggregationValue + ?Sized> {
    builder: &'a mut DenseRunBuilder<KP, V>,
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> MergeSink<KP, V> for AppendSink<'_, KP, V> {
    #[inline(always)]
    fn singleton(&mut self, hash: u64, entry: *const u8) -> Result<bool> {
        unsafe { self.builder.append_raw(hash, entry) };
        Ok(true)
    }

    #[inline(always)]
    fn merged(&mut self, hash: u64, key: &KP, value: &V) -> Result<bool> {
        self.builder
            .append_merged(hash, *key, |stored| stored.copy_from(value));
        Ok(true)
    }
}

/// Merges one worker's sealed tables into a single hash-ordered dense run.
///
/// The worker's tables are dropped as their entries are copied out, so the
/// peak holds one worker's tables plus its run.
pub(super) fn consolidate_worker_runs<S: StoredKey, V: AggregationValue + ?Sized>(
    tables: Vec<SealedTable<S::Persisted, V>>,
    allocator: &mut SlabAllocator,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> DenseRun<S::Persisted, V> {
    V::dispatch_arity(
        V::storage_metadata(context),
        ConsolidateRuns::<S, V> {
            tables,
            allocator,
            key_arena,
            context,
        },
    )
}

/// State passed through arity dispatch for one worker consolidation.
struct ConsolidateRuns<'a, S: StoredKey, V: AggregationValue + ?Sized> {
    tables: Vec<SealedTable<S::Persisted, V>>,
    allocator: &'a mut SlabAllocator,
    key_arena: &'a SharedArena,
    context: &'a V::SharedContext,
}

impl<S: StoredKey, V: AggregationValue + ?Sized> ArityBody<DenseRun<S::Persisted, V>>
    for ConsolidateRuns<'_, S, V>
{
    #[inline(always)]
    fn run<const N: usize>(self) -> DenseRun<S::Persisted, V> {
        let ConsolidateRuns {
            tables,
            allocator,
            key_arena,
            context,
        } = self;
        consolidate_worker_runs_body::<N, S, V>(tables, allocator, key_arena, context)
    }
}

#[inline(never)]
fn consolidate_worker_runs_body<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    tables: Vec<SealedTable<S::Persisted, V>>,
    allocator: &mut SlabAllocator,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> DenseRun<S::Persisted, V> {
    let metadata = if N == 0 {
        V::storage_metadata(context)
    } else {
        V::metadata_for_arity::<N>()
    };
    let capacity: usize = tables.iter().map(|sealed| sealed.run.len()).sum();
    let mut builder =
        DenseRunBuilder::<S::Persisted, V>::with_capacity(allocator, capacity, metadata);
    let mut cursors = Vec::with_capacity(tables.len());
    for sealed in &tables {
        if sealed.run.len() == 0 {
            continue;
        }
        cursors.push(TableCursor::new(
            sealed.table.reader::<N>(),
            sealed.run.positions_ptr(),
            0,
            sealed.run.len(),
        ));
    }
    if let Some(first) = cursors.first() {
        let layout = first.reader;
        let mut sink = AppendSink {
            builder: &mut builder,
        };
        merge_streams::<_, S, V>(cursors, &layout, metadata, key_arena, context, &mut sink)
            .expect("appending to a dense run cannot fail");
    }
    builder.finish()
}

/// Streams one partition's merged groups into `emit`.
///
/// `emit` receives each group exactly once and returns whether the merge
/// should keep going (`false` stops early, for satisfied LIMIT pushdowns).
pub(super) fn merge_partition<S: StoredKey, V: AggregationValue + ?Sized, E>(
    partition: usize,
    num_partitions: usize,
    runs: &[DenseRun<S::Persisted, V>],
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
            runs,
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
    runs: &'a [DenseRun<S::Persisted, V>],
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
            runs,
            key_arena,
            context,
            emit,
        } = self;
        merge_partition_rows::<N, S, V, E>(
            partition,
            num_partitions,
            runs,
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
    runs: &[DenseRun<S::Persisted, V>],
    key_arena: &SharedArena,
    context: &V::SharedContext,
    emit: E,
) -> Result<()>
where
    E: FnMut(&S::Persisted, &V) -> Result<bool>,
{
    let partition_bits = num_partitions.trailing_zeros();
    debug_assert!(partition_bits <= BUCKET_BITS);
    let buckets_per_partition = BUCKET_BITS - partition_bits;
    let bucket_lo = partition << buckets_per_partition;
    let bucket_hi = (partition + 1) << buckets_per_partition;

    let mut cursors: Vec<DenseCursor<'_, S::Persisted, V>> = Vec::with_capacity(runs.len());
    for run in runs {
        let (start, end) = run.bucket_range(bucket_lo, bucket_hi);
        if start == end {
            continue;
        }
        cursors.push(DenseCursor::new(run.reader::<N>(), start, end));
    }
    let Some(first) = cursors.first() else {
        return Ok(());
    };
    let layout = first.reader;
    let metadata = if N == 0 {
        V::storage_metadata(context)
    } else {
        V::metadata_for_arity::<N>()
    };
    let mut sink = EmitSink { layout, emit };
    merge_streams::<_, S, V>(cursors, &layout, metadata, key_arena, context, &mut sink)
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

    fn make_worker_run(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
    ) -> DenseRun<i32, CountValue> {
        make_worker_run_with_spill(state, arena, values, SpillConfig::DEFAULT)
    }

    fn make_worker_run_with_spill(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
        spill: SpillConfig,
    ) -> DenseRun<i32, CountValue> {
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
        agg.flush().run
    }

    fn collect_partition(
        runs: &[DenseRun<i32, CountValue>],
        arena: &SharedArena,
        partition: usize,
        num_partitions: usize,
    ) -> Vec<(i32, usize)> {
        let mut groups = vec![];
        merge_partition::<IntStored, CountValue, _>(
            partition,
            num_partitions,
            runs,
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
        runs: &[DenseRun<i32, CountValue>],
        arena: &SharedArena,
    ) -> Vec<(i32, usize)> {
        let mut all_entries = vec![];
        for p in 0..PARTITIONS {
            all_entries.extend(collect_partition(runs, arena, p, PARTITIONS));
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
            total += collect_partition(&runs, &arena, p, PARTITIONS).len();
        }

        assert_eq!(total, 200);
    }

    #[test]
    fn consolidation_recombines_a_workers_stacked_tables() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let spill = SpillConfig {
            spill_capacity: 256,
        };
        // Each key appears twice, spread across the stack the small capacity
        // forces; the worker's run must combine them.
        let values: Vec<i32> = (0..500).chain(0..500).collect();
        let run = make_worker_run_with_spill(&state, &arena, &values, spill);

        assert_eq!(run.len(), 500);
        let entries = merge_all_partitions(&[run], &arena);
        assert_eq!(entries.len(), 500);
        assert!(entries.iter().all(|&(_, count)| count == 2));
    }

    #[test]
    fn stacked_workers_recombine_across_workers() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let spill = SpillConfig {
            spill_capacity: 256,
        };
        let values: Vec<i32> = (0..500).chain(0..500).collect();
        let runs = vec![
            make_worker_run_with_spill(&state, &arena, &values, spill),
            make_worker_run_with_spill(&state, &arena, &values, spill),
        ];

        let entries = merge_all_partitions(&runs, &arena);

        assert_eq!(entries.len(), 500);
        assert!(entries.iter().all(|&(_, count)| count == 4));
    }

    /// Consolidate one hand-built table of `(hash, key)` pairs into a run,
    /// bypassing the extractor so tests control hash collisions and extreme
    /// hash values.
    fn run_from_pairs(pairs: &[(u64, i32)]) -> DenseRun<i32, CountValue> {
        let mut allocator = SlabAllocator::new(true);
        let mut table: MultiSlabTable<i32, CountValue> =
            MultiSlabTable::new(&mut allocator, 128, 0, &());
        for &(hash, key) in pairs {
            table.prober().merge_from(hash, key, &seeded_count(), &());
        }
        let run = table.build_sorted_run();
        let arena = SharedArena::new(64);
        consolidate_worker_runs::<IntStored, CountValue>(
            vec![SealedTable { table, run }],
            &mut allocator,
            &arena,
            &COUNT_CFG,
        )
    }

    /// A `CountValue` holding a count of one, as one consumed row seeds it.
    fn seeded_count() -> CountValue {
        let mut value = CountValue::default();
        value.seed(&((),), 0, &mut ());
        value
    }

    fn collect_all_from(runs: &[DenseRun<i32, CountValue>]) -> Vec<(i32, usize)> {
        let arena = SharedArena::new(64);
        let mut entries = vec![];
        for p in 0..PARTITIONS {
            entries.extend(collect_partition(runs, &arena, p, PARTITIONS));
        }
        entries.sort();
        entries
    }

    #[test]
    fn same_hash_distinct_keys_stay_distinct_groups() {
        init_test_free_pool(64);
        // Three runs hold hash 42: two with key 1, one with key 2. The merged
        // output must combine the key-1 partials and keep key 2 separate.
        let runs = vec![
            run_from_pairs(&[(42, 1)]),
            run_from_pairs(&[(42, 2)]),
            run_from_pairs(&[(42, 1)]),
        ];

        let entries = collect_all_from(&runs);

        assert_eq!(entries, vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn same_hash_distinct_keys_within_one_run() {
        init_test_free_pool(64);
        let runs = vec![run_from_pairs(&[(42, 1), (42, 2), (42, 3)])];

        let entries = collect_all_from(&runs);

        assert_eq!(entries, vec![(1, 1), (2, 1), (3, 1)]);
    }

    #[test]
    fn extreme_hash_values_merge_without_sentinel_confusion() {
        init_test_free_pool(64);
        // u64::MAX must not collide with the exhausted-cursor sentinel, and
        // hash 1 (the remapped zero) must land in partition 0.
        let runs = vec![
            run_from_pairs(&[(u64::MAX, 7), (1, 9)]),
            run_from_pairs(&[(u64::MAX, 7), (u64::MAX - 1, 8)]),
        ];

        let entries = collect_all_from(&runs);

        assert_eq!(entries, vec![(7, 2), (8, 1), (9, 1)]);
    }

    #[test]
    fn consolidation_merges_max_hash_partials_across_tables() {
        init_test_free_pool(64);
        // Two tables in ONE worker both hold (u64::MAX, 7): the tail drain in
        // the consolidation merge must combine them, not duplicate them.
        let mut allocator = SlabAllocator::new(true);
        let mut sealed = vec![];
        for _ in 0..2 {
            let mut table: MultiSlabTable<i32, CountValue> =
                MultiSlabTable::new(&mut allocator, 128, 0, &());
            table.prober().merge_from(u64::MAX, 7, &seeded_count(), &());
            let run = table.build_sorted_run();
            sealed.push(SealedTable { table, run });
        }
        let arena = SharedArena::new(64);
        let run = consolidate_worker_runs::<IntStored, CountValue>(
            sealed,
            &mut allocator,
            &arena,
            &COUNT_CFG,
        );

        assert_eq!(run.len(), 1);
        let entries = collect_all_from(&[run]);
        assert_eq!(entries, vec![(7, 2)]);
    }

    #[test]
    fn sorted_runs_are_hash_ordered_after_wraparound() {
        init_test_free_pool(64);
        // Hashes near u64::MAX probe past the last slot and wrap to slot 0;
        // the consolidated run must come out in ascending hash order.
        let pairs: Vec<(u64, i32)> = (0..40)
            .map(|i| (u64::MAX - i as u64, i as i32))
            .chain((1..40).map(|i| (i as u64, 100 + i as i32)))
            .collect();
        let run = run_from_pairs(&pairs);

        let reader = run.reader::<0>();
        let mut previous = 0u64;
        for i in 0..run.len() {
            let hash = reader.hash_of(reader.entry_ptr(i));
            assert!(hash >= previous, "run out of order at {}", i);
            previous = hash;
        }
        assert_eq!(run.len(), pairs.len());
    }

    #[test]
    fn early_stop_halts_partition_merge() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let values: Vec<i32> = (0..1000).collect();
        let runs = vec![make_worker_run(&state, &arena, &values)];

        let mut seen = 0usize;
        for p in 0..PARTITIONS {
            merge_partition::<IntStored, CountValue, _>(
                p,
                PARTITIONS,
                &runs,
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
