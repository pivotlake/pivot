//! Computes a sort column's min/max — a private helper for the partition stage
//! ([`super`]), which records them as the file's manifest `sort_bounds` and stamps
//! each row group's into its [`PartitionTag`](super::PartitionTag) for the footer.
//! The assembler later encodes the row-group min/max into the footer bytes.

use std::sync::Arc;

use arrow_arith::aggregate::{max, min};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    StringViewArray,
};
use arrow_schema::DataType;

/// A column's min and max as single-element Arrow arrays, with the column's null
/// count. `None` when the column is empty/all-null or its type isn't one the
/// write path (and the reader's `decode_scalar`) supports — in which case the
/// column simply gets no statistics (pruning won't apply; never unsound).
pub(super) fn column_min_max(array: &ArrayRef) -> Option<(ArrayRef, ArrayRef, i64)> {
    let null_count = array.null_count() as i64;
    macro_rules! numeric {
        ($arr:ty) => {{
            let a = array.as_any().downcast_ref::<$arr>()?;
            let lo: ArrayRef = Arc::new(<$arr>::from(vec![min(a)?]));
            let hi: ArrayRef = Arc::new(<$arr>::from(vec![max(a)?]));
            Some((lo, hi, null_count))
        }};
    }
    macro_rules! strings {
        ($arr:ty) => {{
            let a = array.as_any().downcast_ref::<$arr>()?;
            // Rust `&str` ordering is byte-lexicographic, matching Parquet's
            // unsigned-byte ordering for UTF8 columns.
            let lo: ArrayRef = Arc::new(<$arr>::from(vec![a.iter().flatten().min()?]));
            let hi: ArrayRef = Arc::new(<$arr>::from(vec![a.iter().flatten().max()?]));
            Some((lo, hi, null_count))
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
