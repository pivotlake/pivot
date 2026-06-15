//! Row-encoded multi-column GROUP BY keys.
//!
//! The general-purpose key extractor: it covers every shape the specialised
//! extractors don't — three or more keys, strings mixed with integers, and
//! integer pairs outside the packed [`IntPairKeyExtractor`](super::IntPairKeyExtractor)
//! set. Each row's key columns are serialised, in GROUP BY order, into one
//! contiguous byte string:
//!
//! - fixed-width integers as their little-endian bytes,
//! - strings as a `u32` length prefix followed by the raw bytes — except a
//!   *trailing* string, whose bytes simply run to the end of the blob (its
//!   length is the blob's remaining length, so the prefix is redundant). This
//!   shaves 4 bytes off every such row and, more importantly, keeps many more
//!   tuples within the 12-byte inline budget of [`ArenaKey`].
//!
//! Because the layout is canonical, key-tuple equality is exactly byte equality
//! and one hash covers the whole tuple — so the hash table needs no per-shape
//! logic. The persisted form is an [`ArenaKey`], identical to a single string
//! key: a tuple of ≤ 12 encoded bytes inlines into the table entry, a longer one
//! lives in the shared arena. Output decodes the blobs back into typed arrow
//! columns, emitting string sub-keys as zero-copy views into the same arena
//! bytes (nothing is copied on the string output path).
//!
//! The encoding is driven by a [`RowKeySchema`] (the key columns' arrow types),
//! which the planner supplies as the extractor's [`KeyExtractor::Config`]. It is
//! the one thing neither the per-batch reader nor the output decode can recover
//! from the data alone — the reader needs it to know each column's width, and
//! the decode needs it to rebuild typed columns.

use crate::arrays::{ArrayBuilder, PrimitiveBuilder, SlabColumn};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::LiveKey;
use crate::operations::unary::group::keys::string::ResolvedKey;
use crate::operations::unary::group::keys::{ArenaKey, KeyColumns, KeyExtractor};
use ahash::RandomState;
use arrow_array::builder::make_view;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The key columns' arrow types, in GROUP BY order — the row extractor's
/// [`KeyExtractor::Config`]. Supports the eight fixed-width integer types and
/// `Utf8View`; the planner must only route those shapes here.
#[derive(Clone)]
pub struct RowKeySchema(Arc<[DataType]>);

impl RowKeySchema {
    /// Build a schema from the key columns' arrow types, panicking on an
    /// unsupported type (the planner is responsible for only routing supported
    /// shapes to the row extractor).
    pub fn new(types: impl Into<Arc<[DataType]>>) -> Self {
        let types = types.into();
        for t in types.iter() {
            assert!(
                encoded_width(t).is_some() || *t == DataType::Utf8View,
                "row key column type not supported: {t}"
            );
        }
        Self(types)
    }

    fn types(&self) -> &[DataType] {
        &self.0
    }
}

/// Encoded byte width of a fixed-width integer type; `None` for variable-width
/// (string) columns.
fn encoded_width(dt: &DataType) -> Option<usize> {
    Some(match dt {
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 => 4,
        DataType::Int64 | DataType::UInt64 => 8,
        _ => return None,
    })
}

/// One key column made encodable for a single batch — a downcast primitive
/// array per integer type, or a string array.
///
/// Per-type (rather than a single width-parameterised path) so each integer
/// encodes a *const*-width little-endian copy via `to_le_bytes`, which the
/// compiler lowers to a fixed-size `memcpy` from an unchecked load. On the
/// per-row hot path that's measurably tighter than a runtime-width slice copy.
enum KeyCol<'b> {
    I8(&'b PrimitiveArray<Int8Type>),
    I16(&'b PrimitiveArray<Int16Type>),
    I32(&'b PrimitiveArray<Int32Type>),
    I64(&'b PrimitiveArray<Int64Type>),
    U8(&'b PrimitiveArray<UInt8Type>),
    U16(&'b PrimitiveArray<UInt16Type>),
    U32(&'b PrimitiveArray<UInt32Type>),
    U64(&'b PrimitiveArray<UInt64Type>),
    Str(&'b StringViewArray),
}

impl<'b> KeyCol<'b> {
    fn new(array: &'b ArrayRef) -> Self {
        match array.data_type() {
            DataType::Int8 => KeyCol::I8(array.as_primitive()),
            DataType::Int16 => KeyCol::I16(array.as_primitive()),
            DataType::Int32 => KeyCol::I32(array.as_primitive()),
            DataType::Int64 => KeyCol::I64(array.as_primitive()),
            DataType::UInt8 => KeyCol::U8(array.as_primitive()),
            DataType::UInt16 => KeyCol::U16(array.as_primitive()),
            DataType::UInt32 => KeyCol::U32(array.as_primitive()),
            DataType::UInt64 => KeyCol::U64(array.as_primitive()),
            DataType::Utf8View => KeyCol::Str(array.as_string_view()),
            dt => panic!("row key column type not supported: {dt}"),
        }
    }

    /// Append row `idx`'s encoded bytes to `out`. Safety: `idx` is always within
    /// the batch row count, so the unchecked reads are sound.
    #[inline(always)]
    fn encode(&self, idx: usize, out: &mut Vec<u8>) {
        unsafe {
            match self {
                KeyCol::I8(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                KeyCol::I16(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                KeyCol::I32(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                KeyCol::I64(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                KeyCol::U8(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                KeyCol::U16(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                KeyCol::U32(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                KeyCol::U64(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                KeyCol::Str(a) => {
                    let s = a.value_unchecked(idx).as_bytes();
                    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                    out.extend_from_slice(s);
                }
            }
        }
    }
}

/// Per-worker reusable encode buffers. Cleared and refilled by
/// [`prepare_and_hash`](RowKeyExtractor::prepare_and_hash) each batch; the
/// capacity persists across batches, so nothing is reallocated per batch.
#[derive(Default)]
pub struct RowScratch {
    /// Encoded key tuples, back to back.
    bytes: Vec<u8>,
    /// Row `i` occupies `bytes[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<u32>,
}

/// Per-batch reader: the (possibly cast) key columns plus a borrow of the worker
/// scratch the keys encode into. [`make_reader`](RowKeyExtractor::make_reader)
/// only binds these; [`prepare_and_hash`](RowKeyExtractor::prepare_and_hash)
/// fills the scratch, after which `row(i)` slices it.
pub struct RowReader<'b> {
    casted: Vec<ArrayRef>,
    scratch: &'b mut RowScratch,
}

impl RowReader<'_> {
    #[inline(always)]
    fn row(&self, idx: usize) -> &[u8] {
        let off = &self.scratch.offsets;
        &self.scratch.bytes[off[idx] as usize..off[idx + 1] as usize]
    }
}

/// `GROUP BY (k0, k1, …)` over any mix of integer and string key columns.
pub struct RowKeyExtractor;

impl KeyExtractor for RowKeyExtractor {
    // Stays in-place like a single string key. The radix scatter persists every
    // row's key before deduping, so a > 12-byte tuple would push one arena blob
    // per *occurrence* rather than per distinct key — ballooning the arena at
    // exactly the high cardinality the scatter is meant to relieve.
    type Config = RowKeySchema;
    type Persisted = ArenaKey;
    type LiveKey<'a, 'b> = RowKey<'a, 'b>;
    type PersistedLiveKey<'a> = ResolvedKey<'a>;
    type Reader<'b> = RowReader<'b>;
    type Columns = RowKeyColumns;
    type Scratch = RowScratch;

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        config: &RowKeySchema,
        scratch: &'b mut RowScratch,
    ) -> RowReader<'b> {
        // A column whose runtime type differs from the schema (e.g. a DATE that
        // arrives with its parquet-physical type) is cast once per batch; the
        // owned results live in the reader until `prepare_and_hash` encodes them.
        let casted: Vec<ArrayRef> = key_cols
            .iter()
            .zip(config.types())
            .map(|(&col, want)| {
                let array = batch.column(col);
                if array.data_type() == want {
                    array.clone()
                } else {
                    arrow::compute::cast(array, want).expect("row key column cast failed")
                }
            })
            .collect();
        RowReader { casted, scratch }
    }

    fn prepare_and_hash(reader: &mut RowReader<'_>, state: &RandomState, hashes: &mut [u64]) {
        let cols: Vec<KeyCol> = reader.casted.iter().map(KeyCol::new).collect();
        let scratch = &mut *reader.scratch;
        scratch.bytes.clear();
        scratch.offsets.clear();
        // Reserve once; from the second batch on the cleared buffers already have
        // the capacity, so this is a no-op and nothing reallocates.
        scratch.bytes.reserve(hashes.len() * 16);
        scratch.offsets.reserve(hashes.len() + 1);
        scratch.offsets.push(0);
        // A *trailing* string field needs no length prefix: its bytes run to the
        // end of the row blob, whose length we recover from `offsets` (and, once
        // persisted, from the key's own length). Encoding all but that last field
        // normally and the tail raw shrinks every such row by 4 bytes — and, more
        // importantly, lets many more tuples inline into the 12-byte `ArenaKey`
        // instead of spilling to the arena, which speeds the probe and decode too.
        let head = match cols.last() {
            Some(KeyCol::Str(_)) => cols.len() - 1,
            _ => cols.len(),
        };
        let mut start = 0usize;
        for (i, slot) in hashes.iter_mut().enumerate() {
            for col in &cols[..head] {
                col.encode(i, &mut scratch.bytes);
            }
            if let Some(KeyCol::Str(a)) = cols.get(head) {
                scratch
                    .bytes
                    .extend_from_slice(unsafe { a.value_unchecked(i) }.as_bytes());
            }
            let end = scratch.bytes.len();
            // Hash each blob the moment it is written — still hot from encoding —
            // so the probe never re-reads the buffer just to hash. Identical to
            // hashing the slice `row(i)` returns, so it matches the table's hasher.
            *slot = state.hash_one(&scratch.bytes[start..end]);
            scratch.offsets.push(end as u32);
            start = end;
        }
    }

    #[inline(always)]
    fn live_key<'a, 'b>(
        reader: &Self::Reader<'b>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b> {
        // The row slice borrows the worker scratch (via `&reader`); we extend it
        // to the trait's `'b`. Sound: the scratch outlives the batch, the consume
        // loop holds the reader (unmutated after `prepare_and_hash`) for as long
        // as any live key it produced is alive, and a live key is always consumed
        // — `eq_persisted` or `persist` — before the next row is read.
        let value: &'b [u8] = unsafe { std::mem::transmute(reader.row(idx)) };
        RowKey { arena, value }
    }

    fn resolve_persisted(arena: &SharedArena, persisted: ArenaKey) -> ResolvedKey<'_> {
        ResolvedKey::new(persisted, arena)
    }
}

/// A live row key: the encoded tuple bytes plus the worker arena to compare
/// against / persist into. The byte-blob analogue of `StringKey`.
pub struct RowKey<'a, 'b> {
    arena: &'a mut WorkerArena,
    value: &'b [u8],
}

impl Hash for RowKey<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl LiveKey for RowKey<'_, '_> {
    type Persisted = ArenaKey;

    #[inline(always)]
    fn eq_persisted(&self, other: &ArenaKey) -> bool {
        self.value == other.resolve(self.arena.shared())
    }

    #[inline(always)]
    fn persist(self) -> ArenaKey {
        self.arena.push_bytes(self.value)
    }
}

/// One output key column under construction. Integers decode into a typed
/// primitive builder; strings into view headers pointing back into the arena.
enum FieldBuilder {
    I8(PrimitiveBuilder<Int8Type>),
    I16(PrimitiveBuilder<Int16Type>),
    I32(PrimitiveBuilder<Int32Type>),
    I64(PrimitiveBuilder<Int64Type>),
    U8(PrimitiveBuilder<UInt8Type>),
    U16(PrimitiveBuilder<UInt16Type>),
    U32(PrimitiveBuilder<UInt32Type>),
    U64(PrimitiveBuilder<UInt64Type>),
    Str(SlabColumn<u128>),
}

impl FieldBuilder {
    fn new(dt: &DataType, allocator: &mut SlabAllocator, rows: usize) -> Self {
        match dt {
            DataType::Int8 => FieldBuilder::I8(PrimitiveBuilder::with_capacity(allocator, rows)),
            DataType::Int16 => FieldBuilder::I16(PrimitiveBuilder::with_capacity(allocator, rows)),
            DataType::Int32 => FieldBuilder::I32(PrimitiveBuilder::with_capacity(allocator, rows)),
            DataType::Int64 => FieldBuilder::I64(PrimitiveBuilder::with_capacity(allocator, rows)),
            DataType::UInt8 => FieldBuilder::U8(PrimitiveBuilder::with_capacity(allocator, rows)),
            DataType::UInt16 => FieldBuilder::U16(PrimitiveBuilder::with_capacity(allocator, rows)),
            DataType::UInt32 => FieldBuilder::U32(PrimitiveBuilder::with_capacity(allocator, rows)),
            DataType::UInt64 => FieldBuilder::U64(PrimitiveBuilder::with_capacity(allocator, rows)),
            DataType::Utf8View => FieldBuilder::Str(SlabColumn::with_capacity(allocator, rows)),
            dt => unreachable!("row key column type not supported: {dt}"),
        }
    }
}

/// Pops a little-endian integer off the front of a key blob, advancing it.
macro_rules! pop_le {
    ($blob:expr, $t:ty) => {{
        let (head, rest) = $blob.split_at(std::mem::size_of::<$t>());
        *$blob = rest;
        <$t>::from_le_bytes(head.try_into().unwrap())
    }};
}

/// Emits the decoded key columns of a row-key GROUP BY result.
///
/// Persisted keys are buffered raw (decoding needs the arena, which only
/// [`finish`](KeyColumns::finish) receives). At `finish` each blob is walked
/// once: fixed-width fields scatter into primitive builders, string fields are
/// emitted as views into the blob's own arena bytes — zero-copy.
pub struct RowKeyColumns {
    schema: RowKeySchema,
    keys: SlabColumn<u128>,
}

impl KeyColumns for RowKeyColumns {
    type Key = ArenaKey;
    type Config = RowKeySchema;

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, config: &RowKeySchema) -> Self {
        Self {
            schema: config.clone(),
            keys: SlabColumn::with_capacity(allocator, rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, key: &ArenaKey) {
        self.keys.push(key.as_u128());
    }

    fn finish(
        self,
        arena: &Arc<SharedArena>,
        allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let rows = self.keys.len();
        let mut builders: Vec<FieldBuilder> = self
            .schema
            .types()
            .iter()
            .map(|t| FieldBuilder::new(t, allocator, rows))
            .collect();

        let raw_keys = ScalarBuffer::<u128>::new(self.keys.into_buffer(), 0, rows);
        let last = builders.len() - 1;
        let trailing_str = matches!(self.schema.types().last(), Some(DataType::Utf8View));
        // Keys are walked here in (hash) slot order, but their arena blobs were
        // written in insertion order, so every non-inline `resolve` is a scattered
        // cache miss. Prefetch the blob a few keys ahead to hide that latency; for
        // an all-inline schema the `is_inline` guard skips it (no arena access).
        const DECODE_PREFETCH: usize = 16;
        for i in 0..rows {
            if i + DECODE_PREFETCH < rows {
                let ahead = ArenaKey::from_raw(raw_keys[i + DECODE_PREFETCH]);
                if !ahead.is_inline() {
                    arena.prefetch(ahead.buffer_index(), ahead.offset());
                }
            }
            let key = ArenaKey::from_raw(raw_keys[i]);
            let full = key.resolve(arena);
            let mut blob = full;
            for (j, builder) in builders.iter_mut().enumerate() {
                match builder {
                    FieldBuilder::I8(b) => b.col.push(pop_le!(&mut blob, i8)),
                    FieldBuilder::I16(b) => b.col.push(pop_le!(&mut blob, i16)),
                    FieldBuilder::I32(b) => b.col.push(pop_le!(&mut blob, i32)),
                    FieldBuilder::I64(b) => b.col.push(pop_le!(&mut blob, i64)),
                    FieldBuilder::U8(b) => b.col.push(pop_le!(&mut blob, u8)),
                    FieldBuilder::U16(b) => b.col.push(pop_le!(&mut blob, u16)),
                    FieldBuilder::U32(b) => b.col.push(pop_le!(&mut blob, u32)),
                    FieldBuilder::U64(b) => b.col.push(pop_le!(&mut blob, u64)),
                    FieldBuilder::Str(views) => {
                        // A trailing string carries no length prefix — it is the
                        // rest of the blob; others are u32-length-prefixed.
                        let s = if trailing_str && j == last {
                            std::mem::take(&mut blob)
                        } else {
                            let len = pop_le!(&mut blob, u32) as usize;
                            let (s, rest) = blob.split_at(len);
                            blob = rest;
                            s
                        };
                        // Byte offset of `s` within the key's arena buffer. Strings
                        // ≤ 12 bytes inline into the view header, so the buffer/
                        // offset args only matter for longer ones — and those live
                        // in a blob longer than the string, hence a non-inline blob
                        // with valid arena coordinates.
                        let pos_in_blob = (full.len() - blob.len() - s.len()) as u32;
                        views.push(make_view(s, key.buffer_index(), key.offset() + pos_in_blob));
                    }
                }
            }
        }

        let mut fields = Vec::with_capacity(builders.len());
        let mut columns = Vec::with_capacity(builders.len());
        for (i, builder) in builders.into_iter().enumerate() {
            let name = format!("k{i}");
            let (dt, array): (DataType, ArrayRef) = match builder {
                FieldBuilder::I8(b) => (DataType::Int8, b.into_array(None)),
                FieldBuilder::I16(b) => (DataType::Int16, b.into_array(None)),
                FieldBuilder::I32(b) => (DataType::Int32, b.into_array(None)),
                FieldBuilder::I64(b) => (DataType::Int64, b.into_array(None)),
                FieldBuilder::U8(b) => (DataType::UInt8, b.into_array(None)),
                FieldBuilder::U16(b) => (DataType::UInt16, b.into_array(None)),
                FieldBuilder::U32(b) => (DataType::UInt32, b.into_array(None)),
                FieldBuilder::U64(b) => (DataType::UInt64, b.into_array(None)),
                FieldBuilder::Str(views) => {
                    let len = views.len();
                    let views = ScalarBuffer::<u128>::new(views.into_buffer(), 0, len);
                    let buffers = arena.to_arrow_buffers();
                    // Safety: views built from valid blob slices; the Arc'd arena
                    // keeps the ring memory alive as long as the array exists.
                    let array: ArrayRef =
                        Arc::new(unsafe { StringViewArray::new_unchecked(views, buffers, None) });
                    (DataType::Utf8View, array)
                }
            };
            fields.push(Field::new(name, dt, false));
            columns.push(array);
        }
        (fields, columns)
    }
}
