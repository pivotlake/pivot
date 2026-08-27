//! The encode side: turn a batch's key columns into per-row hashes plus a
//! contiguous blob buffer the probe slices.

use super::schema::RowKeySchema;
use crate::cpu_features::multitarget_kernel;
use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal64Type, Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_schema::DataType;

/// Generates [`ColumnEncoder`] (the encode-side per-column binder) and its
/// methods from the shared `int_key_types!` list, so its integer arms stay in
/// lockstep with the decode side. The `Str` arm is spelled out because it is
/// genuinely different (a `u32` length prefix, not a fixed-width value); the
/// `Dec64`/`Dec128` arms because a decimal carries a precision/scale the
/// unit-variant pattern can't name (its raw bytes encode exactly like an
/// integer of the same width).
///
/// Per-type (rather than a single width-parameterised path) so each integer
/// encodes a *const*-width little-endian copy via `to_le_bytes`, which the
/// compiler lowers to a fixed-size `memcpy` from an unchecked load. On the
/// per-row hot path that's measurably tighter than a runtime-width slice copy.
macro_rules! define_column_encoder {
    ( $( ($variant:ident, $dt:ident, $arrow:ty, $native:ty) ),+ $(,)? ) => {
        /// Encodes one key column into the row blob: a downcast primitive array per
        /// fixed-width type, or a string array, bound once per batch. [`encode`]
        /// appends a cell's canonical bytes.
        ///
        /// [`encode`]: ColumnEncoder::encode
        enum ColumnEncoder<'b> {
            $( $variant(&'b PrimitiveArray<$arrow>), )+
            Dec64(&'b PrimitiveArray<Decimal64Type>),
            Dec128(&'b PrimitiveArray<Decimal128Type>),
            Str(&'b StringViewArray),
        }

        impl<'b> ColumnEncoder<'b> {
            fn new(array: &'b ArrayRef) -> Self {
                match array.data_type() {
                    $( DataType::$dt => ColumnEncoder::$variant(array.as_primitive()), )+
                    DataType::Decimal64(_, _) => ColumnEncoder::Dec64(array.as_primitive()),
                    DataType::Decimal128(_, _) => ColumnEncoder::Dec128(array.as_primitive()),
                    DataType::Utf8View => ColumnEncoder::Str(array.as_string_view()),
                    dt => panic!("row key column type not supported: {dt}"),
                }
            }

            /// Append row `idx`'s encoded bytes to `out`. A nullable field leads
            /// with a validity byte, and a NULL row is that byte alone. Safety:
            /// `idx` is always within the batch row count, so the unchecked reads
            /// are sound.
            #[inline(always)]
            fn encode(&self, idx: usize, nullable: bool, out: &mut Vec<u8>) {
                if nullable {
                    let valid = self.is_valid(idx);
                    out.push(valid as u8);
                    if !valid {
                        return;
                    }
                }
                // Every fixed-width arm is the same: append the value's little-endian
                // bytes (a const-width copy, see the type doc). A decimal appends its
                // raw unscaled integer's bytes at its width; every value of a column
                // shares its scale, so byte equality is value equality.
                macro_rules! le {
                    ($a:expr) => {
                        out.extend_from_slice(&$a.value_unchecked(idx).to_le_bytes())
                    };
                }
                unsafe {
                    match self {
                        $( ColumnEncoder::$variant(a) => le!(a), )+
                        ColumnEncoder::Dec64(a) => le!(a),
                        ColumnEncoder::Dec128(a) => le!(a),
                        ColumnEncoder::Str(a) => {
                            let s = a.value_unchecked(idx).as_bytes();
                            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                            out.extend_from_slice(s);
                        }
                    }
                }
            }

            #[inline(always)]
            fn is_valid(&self, idx: usize) -> bool {
                match self {
                    $( ColumnEncoder::$variant(a) => a.is_valid(idx), )+
                    ColumnEncoder::Dec64(a) => a.is_valid(idx),
                    ColumnEncoder::Dec128(a) => a.is_valid(idx),
                    ColumnEncoder::Str(a) => a.is_valid(idx),
                }
            }
        }
    };
}

int_key_types!(define_column_encoder);

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
    nullable: Vec<bool>,
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
                    // A decimal key must always arrive at its exact schema shape
                    // (the planner threads the column's own precision/scale
                    // through): a decimal-to-decimal arrow cast would rescale
                    // the values, so a mismatch is a planning bug, not a cast.
                    assert!(
                        !matches!(
                            array.data_type(),
                            DataType::Decimal64(_, _) | DataType::Decimal128(_, _)
                        ),
                        "decimal row key column arrived as {} but the schema declares {want}",
                        array.data_type()
                    );
                    arrow::compute::cast(array, want).expect("row key column cast failed")
                }
            })
            .collect();
        RowReader {
            casted,
            nullable: config.nullable().to_vec(),
            scratch,
        }
    }

    multitarget_kernel! {
        /// Encodes and hashes every key row. Schema nullability selects a
        /// specialized loop, so non-nullable schemas compile out validity checks.
        pub(super) fn encode_and_hash(&mut self, state: &RandomState, hashes: &mut [u64]) {
            if self.nullable.contains(&true) {
                self.encode_rows::<true>(state, hashes);
            } else {
                self.encode_rows::<false>(state, hashes);
            }
        }
    }

    // Must stay `inline(always)` so each target-feature clone compiles the row
    // loop for its own CPU tier. Otherwise the shared body stays at the floor.
    #[inline(always)]
    fn encode_rows<const ANY_NULLABLE: bool>(&mut self, state: &RandomState, hashes: &mut [u64]) {
        let encoders: Vec<ColumnEncoder> = self.casted.iter().map(ColumnEncoder::new).collect();
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
        let head = match encoders.last() {
            Some(ColumnEncoder::Str(_)) => encoders.len() - 1,
            _ => encoders.len(),
        };
        let mut start = 0usize;
        for (i, slot) in hashes.iter_mut().enumerate() {
            for (enc, &nullable) in encoders[..head].iter().zip(&self.nullable) {
                enc.encode(i, ANY_NULLABLE && nullable, &mut scratch.bytes);
            }
            if let Some(ColumnEncoder::Str(a)) = encoders.get(head) {
                // The trailing string skips its length prefix, but a nullable one
                // still leads with its validity byte (a NULL is that byte alone).
                let mut write_bytes = true;
                if ANY_NULLABLE && self.nullable[head] {
                    let valid = a.is_valid(i);
                    scratch.bytes.push(valid as u8);
                    write_bytes = valid;
                }
                if write_bytes {
                    scratch
                        .bytes
                        .extend_from_slice(unsafe { a.value_unchecked(i) }.as_bytes());
                }
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
