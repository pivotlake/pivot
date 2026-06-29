//! Assembles a merged partition table into output [`RecordBatch`]es.
//!
//! This is the one place the key and value sides meet on the output path: the
//! [`KeyExtractor`] emits the leading key column(s) and the [`AggregationValue`]
//! the trailing value column(s), and a single combinator zips them — so neither
//! extractor has to know about the other.
//!
//! Columns are built into engine slab memory (see [`crate::arrays`]) via the
//! per-column builders, so output buffers stay on our pre-faulted, accounted
//! memory. Rows are accumulated across partition tables into
//! [`OUTPUT_CHUNK_ROWS`]-row `RecordBatch`es (a slab is at most 2MB), so the
//! emitted batch count tracks the row count, not the partition count.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field as ArrowField, Schema};

use crate::memory::{
    BUFFER_SIZE, HeapBuffer, MultiTopK, Ranked, SingleTopK, SlabAllocator, SlabTopK, slots_per_slab,
};
use crate::operations::channels::Sender;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{Table, TableStorage};
use crate::operations::unary::group::keys::{KeyColumns, KeyExtractor};
use crate::operations::unary::group::values::{AggregationValue, ValueColumns, cast_value_column};

use super::{GroupLimit, Result};

/// Rows per output batch. Each batch's columns are built into single 2MB slabs,
/// so the widest column (a `u128` key = 16 bytes) bounds this: `rows * 16 <= 2MB`.
/// Kept at the maximum so each emitted batch is as full as one slab allows,
/// minimising the per-batch build and downstream-ingest overhead. (A smaller
/// value would also re-wrap the whole arena once per batch for string keys.)
const OUTPUT_CHUNK_ROWS: usize = BUFFER_SIZE / 16;

/// A worker-wide top-`limit` heap for a pushed `ORDER BY <slot> DESC LIMIT`, kept
/// across *every* partition a worker runs.
///
/// The per-partition pushdown can only prune a partition larger than `limit`; the
/// radix path's many small buckets (smaller than `limit`) would each emit all
/// their rows, defeating the LIMIT. Selecting top-`limit` once across the worker's
/// partitions prunes to `limit` rows per worker regardless of how finely the
/// groups were partitioned. Sound for the same reason the per-partition pushdown
/// is: the downstream operator re-applies the global LIMIT over every worker's
/// output.
///
/// Backed by a slab-resident [`SlabTopK`] rather than a `BinaryHeap`, so the
/// worker's top-k lives on the engine's accounted buffers; its buffer grows
/// lazily toward `limit`, so a generous user `LIMIT` reserves no slab until that
/// many rows actually survive.
///
/// The backing is chosen once, here, by `limit`: a `limit` that fits one slab
/// (the near-universal case) takes the contiguous [`SingleTopK`], a larger one the
/// [`MultiTopK`]. Picking it up front rather than per row keeps the per-group
/// `offer` monomorphised, with no runtime dispatch over the storage kind (the
/// caller matches the enum once per partition table, not once per group).
enum TopKHeap<K: KeyExtractor, V: AggregationValue> {
    Single {
        slot: usize,
        heap: SingleTopK<V::SortKey, (K::Persisted, V)>,
    },
    Multi {
        slot: usize,
        heap: MultiTopK<V::SortKey, (K::Persisted, V)>,
    },
}

impl<K: KeyExtractor, V: AggregationValue> TopKHeap<K, V> {
    /// A heap that keeps the top `limit` groups by `sort_key(slot)`.
    fn new(allocator: &mut SlabAllocator, slot: usize, limit: usize) -> Self {
        if limit <= slots_per_slab::<V::SortKey, (K::Persisted, V)>() {
            Self::Single {
                slot,
                heap: SlabTopK::single(allocator, limit),
            }
        } else {
            Self::Multi {
                slot,
                heap: SlabTopK::multi(allocator, limit),
            }
        }
    }
}

/// Offer every group in `table` into `heap`, keyed by the sort key of aggregate
/// `slot`. The heap retains the top rows by that key.
fn offer_all<K, V, A, S>(
    heap: &mut SlabTopK<V::SortKey, (K::Persisted, V), A>,
    slot: usize,
    table: &Table<K, V, S>,
    allocator: &mut SlabAllocator,
) where
    K: KeyExtractor,
    V: AggregationValue,
    A: HeapBuffer<Ranked<V::SortKey, (K::Persisted, V)>>,
    S: TableStorage<K, V>,
{
    for entry in table.iter(0) {
        let sort = entry.value().sort_key(slot);
        heap.offer(allocator, sort, (*entry.key(), *entry.value()));
    }
}

/// How a pushed LIMIT prunes this worker's output. Exactly one mode is live for
/// the accumulator's lifetime, so a single field rules out the impossible "both a
/// top-k heap and a plain-limit budget" state that two `Option`s would allow.
enum OutputMode<K: KeyExtractor, V: AggregationValue> {
    /// `ORDER BY <slot> DESC LIMIT`: keep the worker-wide top-`limit`, drained into
    /// the column builders at [`flush`](OutputAccumulator::flush).
    TopK(TopKHeap<K, V>),
    /// Plain `LIMIT` (no order): rows this worker may still take before it stops
    /// (any `limit` per worker satisfies it; the downstream re-applies the LIMIT).
    First { remaining: usize },
    /// No pushdown: stream every group.
    Unlimited,
}

impl<K: KeyExtractor, V: AggregationValue> OutputMode<K, V> {
    fn new(allocator: &mut SlabAllocator, output_limit: Option<GroupLimit>) -> Self {
        match output_limit {
            Some(GroupLimit::TopK { slot, limit }) => {
                Self::TopK(TopKHeap::new(allocator, slot, limit))
            }
            Some(GroupLimit::First { limit }) => Self::First { remaining: limit },
            None => Self::Unlimited,
        }
    }
}

/// Accumulates output rows across many partition tables into full
/// [`OUTPUT_CHUNK_ROWS`]-row `RecordBatch`es, so the number of emitted batches
/// tracks the total row count rather than the partition count.
///
/// One worker holds one accumulator and feeds it every partition job it runs
/// (see [`extend_from_table`](Self::extend_from_table)); a final
/// [`flush`](Self::flush) emits the remainder. This keeps the radix path, which
/// merges at [`RADIX_PARTITIONS`](super::RADIX_PARTITIONS) granularity, from
/// emitting one tiny batch per radix bucket at moderate cardinality, where the
/// fixed per-batch build and downstream-ingest cost would otherwise dominate.
///
/// A pushed LIMIT is applied **per worker** here rather than per partition, so the
/// radix path's small buckets still prune (see [`TopKHeap`]).
///
/// The per-output-phase constants (`key_arena`, `output_buffers`, `key_config`,
/// `shared_context`) are captured once at construction, so the final flush needs
/// no live job in hand.
pub(crate) struct OutputAccumulator<K: KeyExtractor, V: AggregationValue> {
    /// The key and value column builders for the batch currently filling. Sized to
    /// `builder_cap` at construction and re-made at the same size after every
    /// [`flush`](Self::flush). The accumulator is itself only created once a worker
    /// has a row to add, so these are never allocated for an empty result.
    keys: K::Columns,
    values: V::Columns,
    /// Rows a builder is sized for: `min(limit, OUTPUT_CHUNK_ROWS)` when a LIMIT is
    /// pushed (the result never exceeds it), else a full chunk.
    builder_cap: usize,
    len: usize,
    /// Which pushed LIMIT, if any, prunes this worker's output.
    mode: OutputMode<K, V>,
    key_arena: Arc<SharedArena>,
    output_buffers: Arc<[Buffer]>,
    key_config: K::Config,
    shared_context: V::SharedContext,
    /// Declared output type per value column, in slot order; each finished value
    /// column is cast to its type (see [`emit`](Self::emit)).
    value_output_types: Arc<[DataType]>,
}

impl<K: KeyExtractor, V: AggregationValue> OutputAccumulator<K, V> {
    pub(crate) fn new(
        allocator: &mut SlabAllocator,
        output_limit: Option<GroupLimit>,
        key_arena: Arc<SharedArena>,
        output_buffers: Arc<[Buffer]>,
        key_config: K::Config,
        shared_context: V::SharedContext,
        value_output_types: Arc<[DataType]>,
    ) -> Self {
        // A pushed LIMIT caps the rows this worker ever emits, so its builders never
        // need a full chunk; an unlimited group-by streams full chunks.
        let builder_cap = output_limit.map_or(OUTPUT_CHUNK_ROWS, |limit| {
            limit.row_limit().clamp(1, OUTPUT_CHUNK_ROWS)
        });
        Self {
            keys: K::Columns::with_capacity(allocator, builder_cap, &key_config),
            values: V::Columns::with_capacity(allocator, builder_cap),
            builder_cap,
            len: 0,
            mode: OutputMode::new(allocator, output_limit),
            key_arena,
            output_buffers,
            key_config,
            shared_context,
            value_output_types,
        }
    }

    #[inline]
    fn push(&mut self, key: &K::Persisted, value: V) {
        self.keys.push(key);
        self.values.push(&value);
        self.len += 1;
    }

    /// Append one finished partition table's rows. With a pushed `TopK`/`First`
    /// LIMIT, only the rows that could survive the worker-wide limit are kept;
    /// otherwise every row streams into the column builders, flushing a full batch
    /// whenever the chunk fills.
    pub(crate) fn extend_from_table<S, Snd>(
        &mut self,
        table: Table<K, V, S>,
        allocator: &mut SlabAllocator,
        sender: &mut Snd,
    ) -> Result<()>
    where
        S: TableStorage<K, V>,
        Snd: Sender<RecordBatch>,
    {
        match &mut self.mode {
            // `ORDER BY … DESC LIMIT`: keep only the worker-wide top-`limit`
            // (emitted at flush). No mid-stream flush: the heap is bounded by
            // `limit`. The backing kind is matched here, once, so each row's
            // `offer` runs monomorphised.
            OutputMode::TopK(TopKHeap::Single { slot, heap }) => {
                offer_all(heap, *slot, &table, allocator)
            }
            OutputMode::TopK(TopKHeap::Multi { slot, heap }) => {
                offer_all(heap, *slot, &table, allocator)
            }
            // Plain `LIMIT`: take rows until this worker's budget is spent, then
            // write the budget back so the next partition resumes from it.
            OutputMode::First { remaining } => {
                let mut remaining = *remaining;
                for entry in table.iter(0) {
                    if remaining == 0 {
                        break;
                    }
                    remaining -= 1;
                    self.push(entry.key(), *entry.value());
                    self.flush_if_full(allocator, sender)?;
                }
                self.mode = OutputMode::First { remaining };
            }
            // No pushdown: every group streams into the builders.
            OutputMode::Unlimited => {
                for entry in table.iter(0) {
                    self.push(entry.key(), *entry.value());
                    self.flush_if_full(allocator, sender)?;
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn flush_if_full<Snd: Sender<RecordBatch>>(
        &mut self,
        allocator: &mut SlabAllocator,
        sender: &mut Snd,
    ) -> Result<()> {
        if self.len >= OUTPUT_CHUNK_ROWS {
            self.flush(allocator, sender)
        } else {
            Ok(())
        }
    }

    /// Emit the accumulated rows as `RecordBatch`es and reset the builders. For a
    /// `TopK` limit, the worker-wide heap is materialised into the column builders
    /// first. A no-op when nothing was accumulated.
    pub(crate) fn flush<Snd: Sender<RecordBatch>>(
        &mut self,
        allocator: &mut SlabAllocator,
        sender: &mut Snd,
    ) -> Result<()> {
        if let Some(topk) = self.take_topk() {
            match topk {
                TopKHeap::Single { mut heap, .. } => {
                    for (key, value) in heap.values() {
                        self.push(&key, value);
                        self.flush_if_full(allocator, sender)?;
                    }
                }
                TopKHeap::Multi { mut heap, .. } => {
                    for (key, value) in heap.values() {
                        self.push(&key, value);
                        self.flush_if_full(allocator, sender)?;
                    }
                }
            }
        }
        if self.len == 0 {
            return Ok(());
        }
        // Swap in fresh builders and emit the filled ones. (`emit` consumes the
        // builders to finish them, so they have to be replaced, not borrowed.)
        let keys = std::mem::replace(
            &mut self.keys,
            K::Columns::with_capacity(allocator, self.builder_cap, &self.key_config),
        );
        let values = std::mem::replace(
            &mut self.values,
            V::Columns::with_capacity(allocator, self.builder_cap),
        );
        self.len = 0;
        self.emit(keys, values, allocator, sender)
    }

    /// Build the accumulated key and value columns into one `RecordBatch` and send
    /// it. Reads the per-output-phase state (`key_arena`, `output_buffers`,
    /// `shared_context`) straight off `self`; the caller hands over the filled
    /// builders to finish.
    fn emit<Snd: Sender<RecordBatch>>(
        &self,
        keys: K::Columns,
        values: V::Columns,
        allocator: &mut SlabAllocator,
        sender: &mut Snd,
    ) -> Result<()> {
        let (mut fields, mut columns) =
            keys.finish(&self.key_arena, &self.output_buffers, allocator);
        let (value_fields, value_columns) = values.finish(&self.shared_context);
        // The accumulator renders each value at its storage width; cast it to the
        // slot's declared output type (zero-cost when they already match).
        for ((field, column), output_type) in value_fields
            .into_iter()
            .zip(value_columns)
            .zip(self.value_output_types.iter())
        {
            let (field, column) = cast_value_column(field, column, output_type);
            fields.push(field);
            columns.push(column);
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        sender.send(batch)?;
        Ok(())
    }

    /// Take the top-k heap if in `TopK` mode, switching to `Unlimited` so the heap
    /// is drained exactly once. Leaves `First`/`Unlimited` modes untouched (a
    /// mid-stream flush there keeps its budget).
    fn take_topk(&mut self) -> Option<TopKHeap<K, V>> {
        if matches!(self.mode, OutputMode::TopK(_)) {
            match std::mem::replace(&mut self.mode, OutputMode::Unlimited) {
                OutputMode::TopK(heap) => Some(heap),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }
}

/// Global `COUNT(DISTINCT)`: emit a partition's distinct-key count as a single
/// `Int64` row instead of materialising every key column (hundreds of MB of pure
/// waste at high cardinality); a downstream `SUM` over the per-partition counts
/// gives the total (partitions are hash-disjoint). Callers skip `count == 0`
/// partitions, so every emitted count is non-zero.
pub(crate) fn emit_count<Snd: Sender<RecordBatch>>(count: usize, sender: &mut Snd) -> Result<()> {
    let arr = Arc::new(Int64Array::from(vec![count as i64]));
    let field = ArrowField::new("v0", DataType::Int64, false);
    let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![arr])?;
    sender.send(batch)?;
    Ok(())
}
