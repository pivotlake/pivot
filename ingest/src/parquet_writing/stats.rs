//! A column's min/max, and the bytes the footer records them as.
//!
//! Used at both ends of the pipeline. The [`encoder`](super::encoder) gives every
//! leaf its row group's statistics, which is what lets a reader prune row groups
//! by any column rather than only by the sort key. The
//! [`partition`](super::partition) stage takes the same min/max over a whole
//! file, for the `sort_bounds` it records in the manifest.

use std::sync::Arc;

use arrow_arith::aggregate::{max, min};
use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray, StringViewArray,
};
use arrow_schema::DataType;

/// A column's min and max, as single-element Arrow arrays.
///
/// `None` when the column is empty or all-null, or when its type is not one the
/// write path (and so the reader's `decode_scalar`) supports — a binary leaf, for
/// instance. Such a column simply records no min/max and is never pruned by
/// range, which costs a scan but is never unsound.
pub(super) fn column_min_max(array: &ArrayRef) -> Option<(ArrayRef, ArrayRef)> {
    macro_rules! numeric {
        ($arr:ty) => {{
            let a = array.as_any().downcast_ref::<$arr>()?;
            let lo: ArrayRef = Arc::new(<$arr>::from(vec![min(a)?]));
            let hi: ArrayRef = Arc::new(<$arr>::from(vec![max(a)?]));
            Some((lo, hi))
        }};
    }
    macro_rules! strings {
        ($arr:ty) => {{
            let a = array.as_any().downcast_ref::<$arr>()?;
            // Rust `&str` ordering is byte-lexicographic, matching Parquet's
            // unsigned-byte ordering for UTF8 columns.
            let lo: ArrayRef = Arc::new(<$arr>::from(vec![a.iter().flatten().min()?]));
            let hi: ArrayRef = Arc::new(<$arr>::from(vec![a.iter().flatten().max()?]));
            Some((lo, hi))
        }};
    }
    match array.data_type() {
        DataType::Int32 => numeric!(Int32Array),
        DataType::Int64 => numeric!(Int64Array),
        DataType::Float32 => numeric!(Float32Array),
        DataType::Float64 => numeric!(Float64Array),
        DataType::Utf8 => strings!(StringArray),
        DataType::Utf8View => strings!(StringViewArray),
        _ => None,
    }
}

/// Encode a single-element stats array (a column's min or max, from
/// [`column_min_max`]) into the Parquet `min_value`/`max_value` bytes.
///
/// The encoding must match the reader's `decode_scalar` exactly — little-endian
/// for primitives, raw UTF-8 for strings — or stats-based row-group pruning would
/// silently drop rows.
pub(super) fn stat_bytes(value: &ArrayRef) -> Option<Vec<u8>> {
    macro_rules! le_bytes {
        ($arr:ty) => {
            value
                .as_any()
                .downcast_ref::<$arr>()?
                .value(0)
                .to_le_bytes()
                .to_vec()
        };
    }
    macro_rules! raw_bytes {
        ($arr:ty) => {
            value
                .as_any()
                .downcast_ref::<$arr>()?
                .value(0)
                .as_bytes()
                .to_vec()
        };
    }
    Some(match value.data_type() {
        DataType::Int32 => le_bytes!(Int32Array),
        DataType::Int64 => le_bytes!(Int64Array),
        DataType::Float32 => le_bytes!(Float32Array),
        DataType::Float64 => le_bytes!(Float64Array),
        DataType::Utf8 => raw_bytes!(StringArray),
        DataType::Utf8View => raw_bytes!(StringViewArray),
        _ => return None,
    })
}
