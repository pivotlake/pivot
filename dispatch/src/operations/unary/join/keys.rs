//! How a join reads, hashes, stores, and re-verifies its equi-join key.
//!
//! A [`JoinKey`] is the join's one point of contact with the key columns: the
//! build side reads and hashes each row's key through it and stores the
//! [`Stored`](JoinKey::Stored) value in the key arena, and the probe side
//! hashes its rows the same way and compares candidates' stored values during
//! the match walk. Everything else in the join (the directory, the prefetch
//! pipeline, the output accumulators) never sees a key column.
//!
//! A stored value need not determine the key exactly: a shape whose stored
//! value is lossy (a hash of the full key tuple) confirms each stored-equal
//! candidate through [`verify`](JoinKey::verify), which can read the actual
//! key columns of both sides. Shapes whose stored equality is already exact
//! use a `()` verifier and a `verify` that constant-folds to `true`, leaving
//! the match loop untouched.

use crate::arrays::IntBits;
use crate::operations::unary::join::{PAYLOAD_CHUNK_ROWS, PAYLOAD_CHUNK_SHIFT};
use ahash::RandomState;
use arrow::array::ArrayData;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;

/// One key shape of a hash equi-join. Implementations are zero-sized markers;
/// every method is associated, taking the per-batch state they build.
pub trait JoinKey: 'static {
    /// The `Copy` value stored in each build tuple and the key arena, compared
    /// against the probe row's value during the match walk.
    type Stored: Copy + Eq + Send;
    /// Per-batch accessor bound to one side's key columns.
    type Reader<'a>;
    /// Probe-side state for exact verification of a candidate whose stored
    /// value compared equal, bound to the build payload once the table
    /// publishes. `()` when stored equality is already exact.
    type Verifier<'a>;

    /// Bind `batch`'s key columns. Cheap: downcasts only. `state` is the
    /// join's shared hash state, for a shape whose stored value is itself a
    /// hash and so must carry the state into
    /// [`read_stored`](Self::read_stored).
    fn make_reader<'a>(
        batch: &'a RecordBatch,
        key_columns: &[usize],
        state: &RandomState,
    ) -> Self::Reader<'a>;

    /// Read row `idx`'s stored key. The row is in bounds, and its key columns
    /// are only null on an outer build's kept rows, which the caller screens
    /// with [`is_null`](Self::is_null) first.
    fn read_stored(reader: &Self::Reader<'_>, idx: usize) -> Self::Stored;

    /// Hash row `idx`'s key. Both sides of a join hash through the same shape
    /// and state, so equal keys land in the same directory slot.
    fn hash_row(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64;

    /// Whether any key column is null at row `idx`. Only an outer build's
    /// tuple generation asks; the other paths filter null-keyed rows out of
    /// the batch up front.
    fn is_null(reader: &Self::Reader<'_>, idx: usize) -> bool;

    /// Bind the build payload chunks' key columns for [`verify`](Self::verify).
    fn make_verifier<'a>(
        build_rows: &'a [RecordBatch],
        build_key_columns: &[usize],
    ) -> Self::Verifier<'a>;

    /// Whether probe row `probe_idx` (read through `reader`) and the build row
    /// at payload id `build_row` (read through `verifier`) hold equal keys,
    /// given their stored values already compared equal.
    fn verify(
        reader: &Self::Reader<'_>,
        verifier: &Self::Verifier<'_>,
        probe_idx: usize,
        build_row: u32,
    ) -> bool;
}

/// A join keyed on one primitive column of type `T`: the stored value is the
/// native value itself, so no verification is needed.
///
/// A wrapper rather than an impl on `T` directly because a blanket impl over
/// every `ArrowPrimitiveType` would forbid, by coherence, every other
/// [`JoinKey`] impl.
pub struct SingleColumnKey<T>(PhantomData<fn() -> T>);

impl<T: ArrowPrimitiveType<Native: Hash + Eq>> JoinKey for SingleColumnKey<T> {
    type Stored = T::Native;
    type Reader<'a> = &'a PrimitiveArray<T>;
    type Verifier<'a> = ();

    fn make_reader<'a>(
        batch: &'a RecordBatch,
        key_columns: &[usize],
        _state: &RandomState,
    ) -> Self::Reader<'a> {
        let [key_column] = key_columns else {
            panic!("a single-column join key reads exactly one column");
        };
        batch.column(*key_column).as_primitive::<T>()
    }

    #[inline(always)]
    fn read_stored(reader: &Self::Reader<'_>, idx: usize) -> T::Native {
        unsafe { reader.value_unchecked(idx) }
    }

    #[inline(always)]
    fn hash_row(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
        state.hash_one(Self::read_stored(reader, idx))
    }

    #[inline(always)]
    fn is_null(reader: &Self::Reader<'_>, idx: usize) -> bool {
        reader.is_null(idx)
    }

    fn make_verifier(_build_rows: &[RecordBatch], _build_key_columns: &[usize]) {}

    #[inline(always)]
    fn verify(
        _reader: &Self::Reader<'_>,
        _verifier: &(),
        _probe_idx: usize,
        _build_row: u32,
    ) -> bool {
        true
    }
}

/// The validity shared by `key_columns`: a row is valid when every one of the
/// columns is valid at it. `None` when no key column holds a null.
pub(crate) fn combined_key_validity(
    batch: &RecordBatch,
    key_columns: &[usize],
) -> Option<NullBuffer> {
    let mut combined: Option<NullBuffer> = None;
    for &key_column in key_columns {
        let column = batch.column(key_column);
        if column.null_count() == 0 {
            continue;
        }
        let nulls = column.nulls().unwrap();
        combined = Some(match combined {
            None => nulls.clone(),
            Some(previous) => NullBuffer::new(previous.inner() & nulls.inner()),
        });
    }
    combined
}

/// The per-batch reader of a [`PackedKey`]: one downcast primitive array per
/// key column, plus the columns' combined validity.
pub struct PackedReader<Arrays> {
    arrays: Arrays,
    validity: Option<NullBuffer>,
}

/// A join keyed on a tuple of integer-natured columns (integers, dates,
/// timestamps, `Decimal64`: anything whose native value is [`IntBits`]), each
/// value widened to one `u64` lane of the stored key. Lane equality is value
/// equality because both sides of a condition arrive as one type, so no
/// verification is needed.
///
/// The impls below are stamped for every supported arity, so a new key count
/// never needs its own implementation, only a dispatch arm naming the
/// concrete tuple.
pub struct PackedKey<T>(PhantomData<fn() -> T>);

macro_rules! impl_packed_join_key {
    ($lanes:literal; $($T:ident => $idx:tt),+) => {
        impl<$($T,)+> JoinKey for PackedKey<($($T,)+)>
        where
            $($T: ArrowPrimitiveType<Native: IntBits>,)+
        {
            type Stored = [u64; $lanes];
            type Reader<'a> = PackedReader<($(&'a PrimitiveArray<$T>,)+)>;
            type Verifier<'a> = ();

            fn make_reader<'a>(
                batch: &'a RecordBatch,
                key_columns: &[usize],
                _state: &RandomState,
            ) -> Self::Reader<'a> {
                assert_eq!(key_columns.len(), $lanes, "one key column per packed lane");
                PackedReader {
                    arrays: ($(batch.column(key_columns[$idx]).as_primitive::<$T>(),)+),
                    validity: combined_key_validity(batch, key_columns),
                }
            }

            #[inline(always)]
            fn read_stored(reader: &Self::Reader<'_>, idx: usize) -> [u64; $lanes] {
                [$(unsafe { reader.arrays.$idx.value_unchecked(idx) }.to_u64(),)+]
            }

            #[inline(always)]
            fn hash_row(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
                state.hash_one(Self::read_stored(reader, idx))
            }

            #[inline(always)]
            fn is_null(reader: &Self::Reader<'_>, idx: usize) -> bool {
                match &reader.validity {
                    Some(validity) => !validity.is_valid(idx),
                    None => false,
                }
            }

            fn make_verifier(_build_rows: &[RecordBatch], _build_key_columns: &[usize]) {}

            #[inline(always)]
            fn verify(
                _reader: &Self::Reader<'_>,
                _verifier: &(),
                _probe_idx: usize,
                _build_row: u32,
            ) -> bool {
                true
            }
        }
    };
}

impl_packed_join_key!(2; A => 0, B => 1);
impl_packed_join_key!(3; A => 0, B => 1, C => 2);
impl_packed_join_key!(4; A => 0, B => 1, C => 2, D => 3);

/// One key column of a [`DynamicRowKey`], bound for hashing and comparison.
/// Owned rather than borrowed (the underlying buffers are refcounted), so the
/// reader and verifier need no borrow of the batch they were bound from.
enum DynamicKeyColumn {
    /// Any fixed-width type, viewed as each value's raw bytes. Both sides of
    /// a condition arrive as one type, so byte equality is value equality.
    Fixed {
        values: ArrayData,
        width: usize,
    },
    Utf8View(StringViewArray),
}

impl DynamicKeyColumn {
    fn bind(column: &ArrayRef) -> Self {
        match column.data_type() {
            DataType::Utf8View => Self::Utf8View(column.as_string_view().clone()),
            data_type => {
                let width = data_type.primitive_width().unwrap_or_else(|| {
                    panic!(
                        "unsupported dynamic join key type {data_type}; the planner gates key types"
                    )
                });
                Self::Fixed {
                    values: column.to_data(),
                    width,
                }
            }
        }
    }

    #[inline]
    fn value_bytes(&self, idx: usize) -> &[u8] {
        match self {
            Self::Fixed { values, width } => {
                let start = (values.offset() + idx) * width;
                &values.buffers()[0].as_slice()[start..start + width]
            }
            Self::Utf8View(strings) => strings.value(idx).as_bytes(),
        }
    }

    #[inline]
    fn write_row_to_hasher(&self, idx: usize, hasher: &mut impl Hasher) {
        let bytes = self.value_bytes(idx);
        // The length keeps adjacent variable-length values from aliasing
        // across column boundaries; for fixed-width columns it is constant.
        hasher.write_usize(bytes.len());
        hasher.write(bytes);
    }
}

/// Per-batch state of a [`DynamicRowKey`]: the bound key columns, their
/// combined validity, and the hash state `read_stored` folds each row's
/// column values through.
pub struct DynamicReader {
    columns: Vec<DynamicKeyColumn>,
    validity: Option<NullBuffer>,
    hash_state: RandomState,
}

/// The build payload's key columns, compared against during
/// [`DynamicRowKey::verify`]: per key column, one bound accessor per payload
/// chunk, addressed by the payload id's chunk bits.
pub struct DynamicVerifier {
    columns: Vec<Vec<DynamicKeyColumn>>,
}

/// The fallback join key: any number of key columns of any fixed-width or
/// string type. The stored value is a hash of the whole key tuple, so
/// distinct tuples can collide; every candidate whose stored hash matches is
/// re-compared column by column against the build payload before it counts as
/// a match.
pub struct DynamicRowKey;

impl JoinKey for DynamicRowKey {
    type Stored = u64;
    type Reader<'a> = DynamicReader;
    type Verifier<'a> = DynamicVerifier;

    fn make_reader(
        batch: &RecordBatch,
        key_columns: &[usize],
        state: &RandomState,
    ) -> DynamicReader {
        DynamicReader {
            columns: key_columns
                .iter()
                .map(|&key_column| DynamicKeyColumn::bind(batch.column(key_column)))
                .collect(),
            validity: combined_key_validity(batch, key_columns),
            hash_state: state.clone(),
        }
    }

    #[inline]
    fn read_stored(reader: &DynamicReader, idx: usize) -> u64 {
        let mut hasher = reader.hash_state.build_hasher();
        for column in &reader.columns {
            column.write_row_to_hasher(idx, &mut hasher);
        }
        hasher.finish()
    }

    #[inline]
    fn hash_row(reader: &DynamicReader, idx: usize, _state: &RandomState) -> u64 {
        Self::read_stored(reader, idx)
    }

    #[inline]
    fn is_null(reader: &DynamicReader, idx: usize) -> bool {
        match &reader.validity {
            Some(validity) => !validity.is_valid(idx),
            None => false,
        }
    }

    fn make_verifier(build_rows: &[RecordBatch], build_key_columns: &[usize]) -> DynamicVerifier {
        DynamicVerifier {
            columns: build_key_columns
                .iter()
                .map(|&key_column| {
                    build_rows
                        .iter()
                        .map(|chunk| DynamicKeyColumn::bind(chunk.column(key_column)))
                        .collect()
                })
                .collect(),
        }
    }

    #[inline]
    fn verify(
        reader: &DynamicReader,
        verifier: &DynamicVerifier,
        probe_idx: usize,
        build_row: u32,
    ) -> bool {
        let chunk = (build_row >> PAYLOAD_CHUNK_SHIFT) as usize;
        let row = (build_row as usize) & (PAYLOAD_CHUNK_ROWS - 1);
        reader
            .columns
            .iter()
            .zip(&verifier.columns)
            .all(|(probe_column, build_chunks)| {
                probe_column.value_bytes(probe_idx) == build_chunks[chunk].value_bytes(row)
            })
    }
}
