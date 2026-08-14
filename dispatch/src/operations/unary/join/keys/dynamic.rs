//! The fallback join key: any column mix, verified against the stored rows.

use super::{JoinKey, combined_key_validity};
use crate::operations::unary::join::build_rows::split_row_id;
use ahash::RandomState;
use arrow::array::ArrayData;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, StringViewArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use std::hash::{BuildHasher, Hasher};

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
    Boolean(BooleanArray),
    Utf8View(StringViewArray),
}

impl DynamicKeyColumn {
    fn bind(column: &ArrayRef) -> Self {
        match column.data_type() {
            DataType::Boolean => Self::Boolean(column.as_boolean().clone()),
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
    fn write_row_to_hasher(&self, idx: usize, hasher: &mut impl Hasher) {
        match self {
            Self::Fixed { values, width } => {
                let start = (values.offset() + idx) * width;
                let bytes = &values.buffers()[0].as_slice()[start..start + width];
                hasher.write_usize(bytes.len());
                hasher.write(bytes);
            }
            Self::Boolean(values) => {
                hasher.write_usize(1);
                hasher.write_u8(u8::from(values.value(idx)));
            }
            Self::Utf8View(strings) => {
                let bytes = strings.value(idx).as_bytes();
                hasher.write_usize(bytes.len());
                hasher.write(bytes);
            }
        }
    }

    #[inline]
    fn value_equals(&self, left_idx: usize, other: &Self, right_idx: usize) -> bool {
        match (self, other) {
            (
                Self::Fixed {
                    values: left,
                    width,
                },
                Self::Fixed { values: right, .. },
            ) => {
                let left_start = (left.offset() + left_idx) * width;
                let right_start = (right.offset() + right_idx) * width;
                left.buffers()[0].as_slice()[left_start..left_start + width]
                    == right.buffers()[0].as_slice()[right_start..right_start + width]
            }
            (Self::Boolean(left), Self::Boolean(right)) => {
                left.value(left_idx) == right.value(right_idx)
            }
            (Self::Utf8View(left), Self::Utf8View(right)) => {
                left.value(left_idx) == right.value(right_idx)
            }
            _ => unreachable!("probe and build dynamic join key variants must match"),
        }
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

/// The stored build rows' key columns, compared against during
/// [`DynamicRowKey::verify`]: per key column, one bound accessor per build
/// row batch, addressed by the row id's batch half.
pub struct DynamicVerifier {
    columns: Vec<Vec<DynamicKeyColumn>>,
}

/// The fallback join key: any number of key columns of any fixed-width or
/// string type. The stored value is a hash of the whole key tuple, so
/// distinct tuples can collide; every candidate whose stored hash matches is
/// re-compared column by column against the stored build rows before it
/// counts as a match.
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

    fn make_verifier(
        build_row_batches: &[RecordBatch],
        build_key_columns: &[usize],
    ) -> DynamicVerifier {
        DynamicVerifier {
            columns: build_key_columns
                .iter()
                .map(|&key_column| {
                    build_row_batches
                        .iter()
                        .map(|batch| DynamicKeyColumn::bind(batch.column(key_column)))
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
        build_row_id: u32,
    ) -> bool {
        let (batch_idx, row) = split_row_id(build_row_id);
        reader
            .columns
            .iter()
            .zip(&verifier.columns)
            .all(|(probe_column, build_batches)| {
                probe_column.value_equals(probe_idx, &build_batches[batch_idx], row)
            })
    }
}
