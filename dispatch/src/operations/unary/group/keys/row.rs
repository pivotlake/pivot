//! N-column mixed-type GROUP BY keys, row-encoded into arena-backed byte blobs.
//!
//! Covers every key shape the specialised extractors don't: three or more
//! keys, strings mixed with integers, and type pairs outside the packed
//! `IntPairKeyExtractor` set. Each row's key columns are encoded into one
//! contiguous byte string — fixed-width integers as little-endian bytes,
//! strings length-prefixed — so key-tuple equality is exactly byte equality
//! and a single hash covers the whole tuple. The persisted form is an
//! [`ArenaKey`]: tuples up to 12 encoded bytes are inlined in the entry, longer
//! ones live in the shared arena, exactly like single string keys.
//!
//! Output decodes the blobs back into typed arrow columns. String sub-keys are
//! emitted as zero-copy `StringViewArray` views: their bytes already sit in the
//! arena ring buffers (or fit a view's 12-byte inline form), so no string data
//! is copied on output.
//!
//! The per-batch [`Reader`](RowReader) pre-encodes every row once into a single
//! buffer (one sequential pass over the key columns), so the hash and probe
//! steps just slice it — encoding never runs twice for a row.

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

/// Arrow types of the key columns, in GROUP BY order — the row extractor's
/// [`KeyExtractor::Config`]. The reader uses it to drive encoding (casting a
/// mismatched input column once per batch) and the output columns use it to
/// decode blobs back into typed arrays.
///
/// Supported: all eight fixed-width integer types and `Utf8View`.
#[derive(Clone)]
pub struct RowKeySchema(Arc<[DataType]>);

impl RowKeySchema {
    /// Build a schema from the key columns' arrow types. Panics on a type the
    /// row encoding doesn't support — the planner must only route supported
    /// shapes here.
    pub fn new(types: impl Into<Arc<[DataType]>>) -> Self {
        let types = types.into();
        for t in types.iter() {
            assert!(
                fixed_width(t).is_some() || *t == DataType::Utf8View,
                "unsupported row-key column type: {t}"
            );
        }
        Self(types)
    }

    pub fn types(&self) -> &[DataType] {
        &self.0
    }
}

/// Encoded byte width of a fixed-width key column, `None` for strings.
fn fixed_width(dt: &DataType) -> Option<usize> {
    Some(match dt {
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 => 4,
        DataType::Int64 | DataType::UInt64 => 8,
        _ => return None,
    })
}

/// One key column made encodable: a downcast primitive array, or a string
/// array. Built per batch by [`RowKeyExtractor::make_reader`]; a column whose
/// runtime type differs from the schema (e.g. a date stored as `UInt16` keyed
/// as `Int32`) is cast once per batch and the owned result encoded instead.
enum Col<'b> {
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

impl<'b> Col<'b> {
    fn new(array: &'b ArrayRef) -> Self {
        match array.data_type() {
            DataType::Int8 => Col::I8(array.as_primitive()),
            DataType::Int16 => Col::I16(array.as_primitive()),
            DataType::Int32 => Col::I32(array.as_primitive()),
            DataType::Int64 => Col::I64(array.as_primitive()),
            DataType::UInt8 => Col::U8(array.as_primitive()),
            DataType::UInt16 => Col::U16(array.as_primitive()),
            DataType::UInt32 => Col::U32(array.as_primitive()),
            DataType::UInt64 => Col::U64(array.as_primitive()),
            DataType::Utf8View => Col::Str(array.as_string_view()),
            dt => panic!("unsupported row-key column type: {dt}"),
        }
    }

    /// Append row `idx`'s encoded bytes.
    #[inline(always)]
    fn encode(&self, idx: usize, out: &mut Vec<u8>) {
        // Safety: idx < batch row count for every column.
        unsafe {
            match self {
                Col::I8(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                Col::I16(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                Col::I32(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                Col::I64(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                Col::U8(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                Col::U16(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                Col::U32(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                Col::U64(a) => out.extend_from_slice(&a.value_unchecked(idx).to_le_bytes()),
                Col::Str(a) => {
                    let s = a.value_unchecked(idx).as_bytes();
                    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                    out.extend_from_slice(s);
                }
            }
        }
    }
}

/// Per-batch reader: every row's key tuple pre-encoded into one buffer.
pub struct RowReader {
    /// Encoded rows, back to back.
    bytes: Vec<u8>,
    /// Row `i` spans `bytes[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<u32>,
}

impl RowReader {
    #[inline(always)]
    fn row(&self, idx: usize) -> &[u8] {
        &self.bytes[self.offsets[idx] as usize..self.offsets[idx + 1] as usize]
    }
}

/// `GROUP BY (k0, k1, …)` over any mix of integer and string key columns.
pub struct RowKeyExtractor;

impl KeyExtractor for RowKeyExtractor {
    // Stays in-place like strings: radix scatter persists every row's key, and
    // a >12-byte tuple would push one arena blob per *occurrence* (not per
    // distinct key), ballooning the arena at exactly the high cardinality the
    // scatter is meant to help.
    type Config = RowKeySchema;
    type Persisted = ArenaKey;
    type LiveKey<'a, 'b> = RowKey<'a, 'b>;
    type PersistedLiveKey<'a> = ResolvedKey<'a>;
    type Reader<'b> = RowReader;
    type Columns = RowKeyColumns;

    fn make_reader(batch: &RecordBatch, key_cols: &[usize], config: &RowKeySchema) -> RowReader {
        let rows = batch.num_rows();
        // Columns whose runtime type differs from the schema are cast once per
        // batch (e.g. a DATE column arrives with its parquet-physical type).
        // The cast results must outlive the encode loop, hence the holder vec.
        let casted: Vec<ArrayRef> = key_cols
            .iter()
            .zip(config.types())
            .map(|(&c, want)| {
                let col = batch.column(c);
                if col.data_type() == want {
                    col.clone()
                } else {
                    arrow::compute::cast(col, want).expect("row-key column cast failed")
                }
            })
            .collect();
        let cols: Vec<Col> = casted.iter().map(Col::new).collect();

        // Pre-encode every row. Fixed widths are known; strings grow the buffer
        // as needed (extend_from_slice amortises).
        let fixed: usize = config.types().iter().filter_map(fixed_width).sum();
        let mut bytes = Vec::with_capacity(rows * (fixed + 16));
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(0u32);
        for i in 0..rows {
            for col in &cols {
                col.encode(i, &mut bytes);
            }
            offsets.push(bytes.len() as u32);
        }
        RowReader { bytes, offsets }
    }

    #[inline(always)]
    fn hash(reader: &RowReader, idx: usize, state: &RandomState) -> u64 {
        state.hash_one(reader.row(idx))
    }

    #[inline(always)]
    fn live_key<'a, 'b>(
        reader: &RowReader,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> RowKey<'a, 'b> {
        // The trait's `'b` is the input batch's lifetime, but this reader owns
        // its encoded rows rather than borrowing the batch, so the row slice
        // carries the reader borrow instead. Extending it to `'b` is sound
        // because the consume path keeps the reader alive (and never mutates
        // it after `make_reader`) for as long as any live key it produced is
        // in use — a live key is always consumed (`eq_persisted`/`persist`)
        // before the next row is read.
        let value: &'b [u8] = unsafe { std::mem::transmute(reader.row(idx)) };
        RowKey { arena, value }
    }

    fn resolve_persisted(arena: &SharedArena, persisted: ArenaKey) -> ResolvedKey<'_> {
        ResolvedKey {
            key: persisted,
            arena,
        }
    }
}

/// A live row key: the encoded tuple bytes plus the worker arena to compare
/// against / persist into. The byte-blob twin of `StringKey`.
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

/// Decoder for one output key column.
enum ColBuilder {
    I8(PrimitiveBuilder<Int8Type>),
    I16(PrimitiveBuilder<Int16Type>),
    I32(PrimitiveBuilder<Int32Type>),
    I64(PrimitiveBuilder<Int64Type>),
    U8(PrimitiveBuilder<UInt8Type>),
    U16(PrimitiveBuilder<UInt16Type>),
    U32(PrimitiveBuilder<UInt32Type>),
    U64(PrimitiveBuilder<UInt64Type>),
    /// String view headers; the bytes stay in the arena (or inline in the view).
    Str(SlabColumn<u128>),
}

/// Reads a little-endian primitive from the front of `blob` and advances it.
macro_rules! take_le {
    ($blob:expr, $t:ty) => {{
        let (head, rest) = $blob.split_at(std::mem::size_of::<$t>());
        *$blob = rest;
        <$t>::from_le_bytes(head.try_into().unwrap())
    }};
}

/// Emits the decoded key columns of a row-key GROUP BY result.
///
/// Pushed keys are buffered raw (decoding needs the arena, which only
/// [`finish`](KeyColumns::finish) receives); `finish` walks each blob once,
/// scattering fixed-width fields into primitive builders and emitting string
/// fields as views into the blob's own arena bytes.
pub struct RowKeyColumns {
    schema: RowKeySchema,
    /// Raw [`ArenaKey`] bit patterns, in push order.
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
        let mut builders: Vec<ColBuilder> = self
            .schema
            .types()
            .iter()
            .map(|t| match t {
                DataType::Int8 => ColBuilder::I8(PrimitiveBuilder::with_capacity(allocator, rows)),
                DataType::Int16 => {
                    ColBuilder::I16(PrimitiveBuilder::with_capacity(allocator, rows))
                }
                DataType::Int32 => {
                    ColBuilder::I32(PrimitiveBuilder::with_capacity(allocator, rows))
                }
                DataType::Int64 => {
                    ColBuilder::I64(PrimitiveBuilder::with_capacity(allocator, rows))
                }
                DataType::UInt8 => ColBuilder::U8(PrimitiveBuilder::with_capacity(allocator, rows)),
                DataType::UInt16 => {
                    ColBuilder::U16(PrimitiveBuilder::with_capacity(allocator, rows))
                }
                DataType::UInt32 => {
                    ColBuilder::U32(PrimitiveBuilder::with_capacity(allocator, rows))
                }
                DataType::UInt64 => {
                    ColBuilder::U64(PrimitiveBuilder::with_capacity(allocator, rows))
                }
                DataType::Utf8View => ColBuilder::Str(SlabColumn::with_capacity(allocator, rows)),
                dt => unreachable!("unsupported row-key column type: {dt}"),
            })
            .collect();

        let raw_keys = ScalarBuffer::<u128>::new(self.keys.into_buffer(), 0, rows);
        for &raw in raw_keys.iter() {
            let key = ArenaKey::from_raw(raw);
            let full = key.resolve(arena);
            let mut blob = full;
            for b in builders.iter_mut() {
                match b {
                    ColBuilder::I8(p) => p.push(&take_le!(&mut blob, i8), 1),
                    ColBuilder::I16(p) => p.push(&take_le!(&mut blob, i16), 1),
                    ColBuilder::I32(p) => p.push(&take_le!(&mut blob, i32), 1),
                    ColBuilder::I64(p) => p.push(&take_le!(&mut blob, i64), 1),
                    ColBuilder::U8(p) => p.push(&take_le!(&mut blob, u8), 1),
                    ColBuilder::U16(p) => p.push(&take_le!(&mut blob, u16), 1),
                    ColBuilder::U32(p) => p.push(&take_le!(&mut blob, u32), 1),
                    ColBuilder::U64(p) => p.push(&take_le!(&mut blob, u64), 1),
                    ColBuilder::Str(views) => {
                        let len = take_le!(&mut blob, u32) as usize;
                        let (s, rest) = blob.split_at(len);
                        blob = rest;
                        // The string's byte offset within the key's arena
                        // buffer: how far we've consumed into the blob, plus
                        // the blob's own offset. ≤ 12-byte strings inline into
                        // the view, so the buffer args only matter for longer
                        // strings — whose blobs are necessarily non-inline
                        // (blob length > string length > 12), making
                        // `key.offset()` valid arena coordinates.
                        let pos_in_blob = (full.len() - blob.len() - s.len()) as u32;
                        views.push(make_view(s, key.buffer_index(), key.offset() + pos_in_blob));
                    }
                }
            }
        }

        let fields_and_arrays: Vec<(Field, ArrayRef)> = builders
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                let name = format!("k{i}");
                match b {
                    ColBuilder::I8(p) => {
                        (Field::new(name, DataType::Int8, false), p.into_array(None))
                    }
                    ColBuilder::I16(p) => {
                        (Field::new(name, DataType::Int16, false), p.into_array(None))
                    }
                    ColBuilder::I32(p) => {
                        (Field::new(name, DataType::Int32, false), p.into_array(None))
                    }
                    ColBuilder::I64(p) => {
                        (Field::new(name, DataType::Int64, false), p.into_array(None))
                    }
                    ColBuilder::U8(p) => {
                        (Field::new(name, DataType::UInt8, false), p.into_array(None))
                    }
                    ColBuilder::U16(p) => (
                        Field::new(name, DataType::UInt16, false),
                        p.into_array(None),
                    ),
                    ColBuilder::U32(p) => (
                        Field::new(name, DataType::UInt32, false),
                        p.into_array(None),
                    ),
                    ColBuilder::U64(p) => (
                        Field::new(name, DataType::UInt64, false),
                        p.into_array(None),
                    ),
                    ColBuilder::Str(views) => {
                        let len = views.len();
                        let views = ScalarBuffer::<u128>::new(views.into_buffer(), 0, len);
                        let buffers = arena.to_arrow_buffers();
                        // Safety: views built from valid blob slices; the Arc'd
                        // arena keeps the ring memory alive with the array.
                        let arr: ArrayRef = Arc::new(unsafe {
                            StringViewArray::new_unchecked(views, buffers, None)
                        });
                        (Field::new(name, DataType::Utf8View, false), arr)
                    }
                }
            })
            .collect();
        fields_and_arrays.into_iter().unzip()
    }
}
