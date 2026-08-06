//! The encode side: hash a batch's key columns row by row straight off the
//! arrays, and encode a row's canonical blob only when the table actually
//! needs its bytes — on inserting a new group, or when comparing against a
//! persisted key (which walks the columns instead of building a blob at all).

use super::schema::RowKeySchema;
use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal64Type, Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_schema::DataType;
use std::cell::RefCell;
use std::hash::{BuildHasher, Hasher};

/// Generates [`ColumnEncoder`] (the per-column binder) and its methods from the
/// shared `int_key_types!` list, so its integer arms stay in lockstep with the
/// decode side. The `Str` arm is spelled out because it is genuinely different
/// (a `u32` length prefix, not a fixed-width value); the `Dec64`/`Dec128` arms
/// because a decimal carries a precision/scale the unit-variant pattern can't
/// name (its raw bytes encode exactly like an integer of the same width).
///
/// Per-type (rather than a single width-parameterised path) so each integer
/// encodes a *const*-width little-endian copy via `to_le_bytes`, which the
/// compiler lowers to a fixed-size `memcpy` from an unchecked load. On the
/// per-row hot path that's measurably tighter than a runtime-width slice copy.
macro_rules! define_column_encoder {
    ( $( ($variant:ident, $dt:ident, $arrow:ty, $native:ty) ),+ $(,)? ) => {
        /// One key column bound for the batch: a downcast primitive array per
        /// fixed-width type, or a string array. [`hash`] feeds a cell's
        /// canonical bytes to a hasher, [`encode`] appends them to a blob, and
        /// [`matches`] compares them against a persisted blob's cursor —
        /// three walks over the same canonical byte stream.
        ///
        /// [`hash`]: ColumnEncoder::hash
        /// [`encode`]: ColumnEncoder::encode
        /// [`matches`]: ColumnEncoder::matches
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

            /// Feed row `idx`'s canonical bytes into `hasher`: exactly the bytes
            /// [`encode`](Self::encode) would append, so equal tuples hash
            /// equally however many columns they span. Safety: `idx` is always
            /// within the batch row count, so the unchecked reads are sound.
            #[inline(always)]
            fn hash<H: Hasher>(&self, idx: usize, nullable: bool, hasher: &mut H) {
                if nullable {
                    let valid = self.is_valid(idx);
                    hasher.write_u8(valid as u8);
                    if !valid {
                        return;
                    }
                }
                macro_rules! le {
                    ($a:expr) => {
                        hasher.write(&$a.value_unchecked(idx).to_le_bytes())
                    };
                }
                unsafe {
                    match self {
                        $( ColumnEncoder::$variant(a) => le!(a), )+
                        ColumnEncoder::Dec64(a) => le!(a),
                        ColumnEncoder::Dec128(a) => le!(a),
                        ColumnEncoder::Str(a) => {
                            let s = a.value_unchecked(idx).as_bytes();
                            hasher.write(&(s.len() as u32).to_le_bytes());
                            hasher.write(s);
                        }
                    }
                }
            }

            /// Append row `idx`'s encoded bytes to `out`. A nullable field leads
            /// with a validity byte, and a NULL row is that byte alone.
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

            /// Compare row `idx` against the front of `cursor` (a persisted
            /// blob's remaining bytes), advancing past the field on a match.
            /// Returns `false` the moment any byte disagrees, cursor included
            /// running short.
            #[inline(always)]
            fn matches(&self, idx: usize, nullable: bool, cursor: &mut &[u8]) -> bool {
                if nullable {
                    let Some((&stored_valid, rest)) = cursor.split_first() else {
                        return false;
                    };
                    *cursor = rest;
                    let valid = self.is_valid(idx);
                    if stored_valid != valid as u8 {
                        return false;
                    }
                    if !valid {
                        return true;
                    }
                }
                // The chunk width is `want`'s array length, inferred const per
                // arm, so the comparison is a fixed-width load and compare.
                macro_rules! le {
                    ($a:expr) => {{
                        let want = $a.value_unchecked(idx).to_le_bytes();
                        let Some(field) = cursor.first_chunk() else {
                            return false;
                        };
                        if *field != want {
                            return false;
                        }
                        *cursor = &cursor[want.len()..];
                        true
                    }};
                }
                unsafe {
                    match self {
                        $( ColumnEncoder::$variant(a) => le!(a), )+
                        ColumnEncoder::Dec64(a) => le!(a),
                        ColumnEncoder::Dec128(a) => le!(a),
                        ColumnEncoder::Str(a) => {
                            let s = a.value_unchecked(idx).as_bytes();
                            let Some(len) = cursor.first_chunk::<4>() else {
                                return false;
                            };
                            if u32::from_le_bytes(*len) as usize != s.len()
                                || cursor.len() < 4 + s.len()
                            {
                                return false;
                            }
                            if !short_bytes_eq(&cursor[4..4 + s.len()], s) {
                                return false;
                            }
                            *cursor = &cursor[4 + s.len()..];
                            true
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

/// Equality on byte slices of equal length, tuned for the very short strings
/// group keys usually are: a byte loop the compiler unrolls, instead of a
/// `memcmp` call whose setup dwarfs a one-byte comparison. Falls back to slice
/// equality (libc memcmp) past the length where the call starts winning.
#[inline(always)]
fn short_bytes_eq(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    if a.len() > 16 {
        return a == b;
    }
    let mut equal = true;
    for i in 0..a.len() {
        equal &= a[i] == b[i];
    }
    equal
}

/// Per-worker reusable state, owned by the table and reused across batches.
#[derive(Default)]
pub struct RowScratch {
    /// The batch's (possibly cast) key columns, kept alive here so the reader's
    /// downcast borrows outlive the probe.
    arrays: Vec<ArrayRef>,
    /// The single-row blob built when a new group persists its key. Interior
    /// mutability because persisting happens under the probe's shared reader
    /// borrow; the capacity persists across inserts.
    insert_blob: RefCell<Vec<u8>>,
}

/// Per-batch reader: the bound key columns plus the schema facts the walks
/// need. [`new`](RowReader::new) binds; [`hash_rows`](RowReader::hash_rows)
/// fills the batch's hash array; [`eq_row`](RowReader::eq_row) and
/// [`encode_row`](RowReader::encode_row) serve the probe's per-candidate
/// compare and per-insert persist.
pub struct RowReader<'b> {
    encoders: Vec<ColumnEncoder<'b>>,
    nullable: Vec<bool>,
    /// Fields encoded with their prefix machinery; a *trailing* string field is
    /// past this index and encodes as raw bytes running to the blob's end (its
    /// length is recoverable from the blob length, so the prefix is redundant).
    /// This shaves 4 bytes off every such row and, more importantly, lets many
    /// more tuples inline into the 12-byte `ArenaKey`.
    head: usize,
    insert_blob: &'b RefCell<Vec<u8>>,
}

impl<'b> RowReader<'b> {
    /// Bind the key columns, casting any whose runtime type differs from the
    /// schema (e.g. a DATE arriving as its parquet-physical type). The owned
    /// cast results live in the scratch, which outlives the reader's borrows.
    pub(super) fn new(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        config: &RowKeySchema,
        scratch: &'b mut RowScratch,
    ) -> Self {
        scratch.arrays = key_cols
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
        let scratch: &'b RowScratch = scratch;
        let encoders: Vec<ColumnEncoder<'b>> =
            scratch.arrays.iter().map(ColumnEncoder::new).collect();
        let head = match encoders.last() {
            Some(ColumnEncoder::Str(_)) => encoders.len() - 1,
            _ => encoders.len(),
        };
        RowReader {
            encoders,
            nullable: config.nullable().to_vec(),
            head,
            insert_blob: &scratch.insert_blob,
        }
    }

    /// Write each row's hash into `hashes` (sized to the batch length by the
    /// caller), reading straight off the bound columns — no key bytes are
    /// materialised. The hashed stream length-prefixes every string (the
    /// trailing one included, unlike the stored blob, where its length is
    /// implied), so equal tuples hash equally and field boundaries stay
    /// unambiguous; nothing else is required of it: the table locates slots by
    /// this hash and resolves collisions through [`eq_row`](Self::eq_row), and
    /// the merge carries stored hashes without ever re-hashing. The row loop is
    /// instantiated per schema nullability: a NULL-free schema takes the
    /// `ANY_NULLABLE = false` copy, whose per-field validity checks const-fold
    /// away.
    pub(super) fn hash_rows(&self, state: &RandomState, hashes: &mut [u64]) {
        if self.nullable.contains(&true) {
            self.hash_rows_impl::<true>(state, hashes);
        } else {
            self.hash_rows_impl::<false>(state, hashes);
        }
    }

    fn hash_rows_impl<const ANY_NULLABLE: bool>(&self, state: &RandomState, hashes: &mut [u64]) {
        for (i, slot) in hashes.iter_mut().enumerate() {
            let mut hasher = state.build_hasher();
            for (enc, &nullable) in self.encoders.iter().zip(&self.nullable) {
                enc.hash(i, ANY_NULLABLE && nullable, &mut hasher);
            }
            *slot = hasher.finish();
        }
    }

    /// Whether row `idx`'s key tuple equals the persisted blob `stored`.
    /// Walks the columns against the blob instead of encoding the row, so a
    /// probe hit never materialises key bytes.
    #[inline(always)]
    pub(super) fn eq_row(&self, idx: usize, stored: &[u8]) -> bool {
        let mut cursor = stored;
        for (enc, &nullable) in self.encoders[..self.head].iter().zip(&self.nullable) {
            if !enc.matches(idx, nullable, &mut cursor) {
                return false;
            }
        }
        if let Some(ColumnEncoder::Str(a)) = self.encoders.get(self.head) {
            if self.nullable[self.head] {
                let Some((&stored_valid, rest)) = cursor.split_first() else {
                    return false;
                };
                cursor = rest;
                let valid = a.is_valid(idx);
                if stored_valid != valid as u8 {
                    return false;
                }
                if !valid {
                    return cursor.is_empty();
                }
            }
            let s = unsafe { a.value_unchecked(idx) }.as_bytes();
            return cursor.len() == s.len() && short_bytes_eq(cursor, s);
        }
        cursor.is_empty()
    }

    /// Encode row `idx`'s canonical blob into the shared insert buffer and hand
    /// it to `use_blob`. Called once per *new group*, never per row.
    #[inline(never)]
    pub(super) fn encode_row<R>(&self, idx: usize, use_blob: impl FnOnce(&[u8]) -> R) -> R {
        let mut blob = self.insert_blob.borrow_mut();
        blob.clear();
        for (enc, &nullable) in self.encoders[..self.head].iter().zip(&self.nullable) {
            enc.encode(idx, nullable, &mut blob);
        }
        if let Some(ColumnEncoder::Str(a)) = self.encoders.get(self.head) {
            // The trailing string skips its length prefix, but a nullable one
            // still leads with its validity byte (a NULL is that byte alone).
            let mut write_bytes = true;
            if self.nullable[self.head] {
                let valid = a.is_valid(idx);
                blob.push(valid as u8);
                write_bytes = valid;
            }
            if write_bytes {
                blob.extend_from_slice(unsafe { a.value_unchecked(idx) }.as_bytes());
            }
        }
        use_blob(&blob)
    }

    /// Feed row `idx`'s canonical byte stream into `hasher`, exactly as
    /// [`hash_rows`](Self::hash_rows) does per batch.
    pub(super) fn hash_row_into<H: Hasher>(&self, idx: usize, hasher: &mut H) {
        for (enc, &nullable) in self.encoders.iter().zip(&self.nullable) {
            enc.hash(idx, nullable, hasher);
        }
    }
}
