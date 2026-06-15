//! The encode side: turn a batch's key columns into per-row hashes plus a
//! contiguous blob buffer the probe slices.

use super::schema::RowKeySchema;
use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_schema::DataType;

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
/// [`RowReader::encode_and_hash`] each batch; the capacity persists across
/// batches, so nothing is reallocated per batch.
#[derive(Default)]
pub struct RowScratch {
    /// Encoded key tuples, back to back.
    bytes: Vec<u8>,
    /// Row `i` occupies `bytes[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<u32>,
}

/// Per-batch reader: the (possibly cast) key columns plus a borrow of the worker
/// scratch the keys encode into. [`new`](RowReader::new) only binds these;
/// [`encode_and_hash`](RowReader::encode_and_hash) fills the scratch, after which
/// [`row`](RowReader::row) slices it.
pub struct RowReader<'b> {
    casted: Vec<ArrayRef>,
    scratch: &'b mut RowScratch,
}

impl<'b> RowReader<'b> {
    /// Bind the key columns, casting any whose runtime type differs from the
    /// schema (e.g. a DATE arriving as its parquet-physical type). The owned
    /// cast results live in the reader until `encode_and_hash` consumes them.
    pub(super) fn new(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        config: &RowKeySchema,
        scratch: &'b mut RowScratch,
    ) -> Self {
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

    /// Encode every row's key tuple into the scratch buffer and write its hash
    /// into `hashes` (sized to the batch length by the caller).
    pub(super) fn encode_and_hash(&mut self, state: &RandomState, hashes: &mut [u64]) {
        let cols: Vec<KeyCol> = self.casted.iter().map(KeyCol::new).collect();
        let scratch = &mut *self.scratch;
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

    /// Row `idx`'s encoded blob, borrowed from the scratch.
    #[inline(always)]
    pub(super) fn row(&self, idx: usize) -> &[u8] {
        let off = &self.scratch.offsets;
        &self.scratch.bytes[off[idx] as usize..off[idx + 1] as usize]
    }
}
