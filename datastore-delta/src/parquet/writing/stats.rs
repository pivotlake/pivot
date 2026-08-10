//! A column's min/max, and the bytes the footer records them as.
//!
//! Used at both ends of the pipeline. The [`encoder`](super::encoder) gives every
//! leaf its row group's statistics, which is what lets a reader prune row groups
//! by any column rather than only by the sort key. The whole-file aggregation
//! feeds the per-file statistics the catalog commits into the Delta log.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_arith::aggregate::{max, min};
use arrow_array::{
    Array, ArrayRef, Date32Array, Datum, Decimal64Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, Scalar, StringArray,
    StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, TimeUnit};

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
        DataType::Int8 => numeric!(Int8Array),
        DataType::Int16 => numeric!(Int16Array),
        DataType::Int32 => numeric!(Int32Array),
        DataType::Int64 => numeric!(Int64Array),
        // Taking the bounds from the unsigned array is what makes them unsigned
        // bounds: the same values compared as the signed physical type they are
        // stored in would order everything past the signed maximum below zero,
        // and a reader pruning on those bounds would drop live rows.
        DataType::UInt8 => numeric!(UInt8Array),
        DataType::UInt16 => numeric!(UInt16Array),
        DataType::UInt32 => numeric!(UInt32Array),
        DataType::UInt64 => numeric!(UInt64Array),
        DataType::Date32 => numeric!(Date32Array),
        DataType::Timestamp(TimeUnit::Microsecond, None) => numeric!(TimestampMicrosecondArray),
        DataType::Float32 => numeric!(Float32Array),
        DataType::Float64 => numeric!(Float64Array),
        // Ordering decimals by their unscaled integers is the numeric order,
        // since every value in the column shares the column's scale. The bounds
        // are restamped with the column's own precision and scale, which a bare
        // `from` would replace with the type's defaults.
        DataType::Decimal64(_, _) => {
            let a = array.as_any().downcast_ref::<Decimal64Array>()?;
            let lo: ArrayRef = Arc::new(
                Decimal64Array::from(vec![min(a)?]).with_data_type(array.data_type().clone()),
            );
            let hi: ArrayRef = Arc::new(
                Decimal64Array::from(vec![max(a)?]).with_data_type(array.data_type().clone()),
            );
            Some((lo, hi))
        }
        DataType::Decimal128(_, _) => {
            let a = array.as_any().downcast_ref::<Decimal128Array>()?;
            let lo: ArrayRef = Arc::new(
                Decimal128Array::from(vec![min(a)?]).with_data_type(array.data_type().clone()),
            );
            let hi: ArrayRef = Arc::new(
                Decimal128Array::from(vec![max(a)?]).with_data_type(array.data_type().clone()),
            );
            Some((lo, hi))
        }
        DataType::Utf8 => strings!(StringArray),
        DataType::Utf8View => strings!(StringViewArray),
        _ => None,
    }
}

/// Encode a single-element stats array (a column's min or max, from
/// [`column_min_max`]) into the Parquet `min_value`/`max_value` bytes.
///
/// The encoding must match the reader's `decode_scalar` exactly — little-endian
/// for primitives and for the INT32/INT64 decimal storages, raw UTF-8 for
/// strings, and big-endian 16-byte two's-complement for wide decimals (the
/// spec's FIXED_LEN_BYTE_ARRAY stats encoding) — or stats-based row-group
/// pruning would silently drop rows.
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
    // A stat for a value narrower than its physical type is widened the same way
    // the encoder widens the values themselves, so both sides of the file agree
    // on the width a reader reads back.
    macro_rules! widened_le_bytes {
        ($arr:ty, $physical:ty) => {
            (value.as_any().downcast_ref::<$arr>()?.value(0) as $physical)
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
        DataType::Int8 => widened_le_bytes!(Int8Array, i32),
        DataType::Int16 => widened_le_bytes!(Int16Array, i32),
        DataType::Int32 => le_bytes!(Int32Array),
        DataType::Int64 => le_bytes!(Int64Array),
        DataType::UInt8 => widened_le_bytes!(UInt8Array, u32),
        DataType::UInt16 => widened_le_bytes!(UInt16Array, u32),
        DataType::UInt32 => le_bytes!(UInt32Array),
        DataType::UInt64 => le_bytes!(UInt64Array),
        DataType::Date32 => le_bytes!(Date32Array),
        DataType::Timestamp(TimeUnit::Microsecond, None) => le_bytes!(TimestampMicrosecondArray),
        DataType::Float32 => le_bytes!(Float32Array),
        DataType::Float64 => le_bytes!(Float64Array),
        // Decimal stats bytes follow the precision-chosen value storage:
        // little-endian for the INT32/INT64 forms, 16 big-endian bytes for the
        // wide fixed-length form.
        DataType::Decimal64(precision, _) => {
            let unscaled = value.as_any().downcast_ref::<Decimal64Array>()?.value(0);
            decimal_stat_bytes(unscaled as i128, *precision)
        }
        DataType::Decimal128(precision, _) => {
            let unscaled = value.as_any().downcast_ref::<Decimal128Array>()?.value(0);
            decimal_stat_bytes(unscaled, *precision)
        }
        DataType::Utf8 => raw_bytes!(StringArray),
        DataType::Utf8View => raw_bytes!(StringViewArray),
        _ => return None,
    })
}

/// Encode one decimal bound's unscaled integer in the value storage its
/// precision selects (see `decimal_write_storage`).
fn decimal_stat_bytes(unscaled: i128, precision: u8) -> Vec<u8> {
    match crate::parquet::decimal_write_storage(precision) {
        crate::parquet::DecimalWriteStorage::Int32 => (unscaled as i32).to_le_bytes().to_vec(),
        crate::parquet::DecimalWriteStorage::Int64 => (unscaled as i64).to_le_bytes().to_vec(),
        crate::parquet::DecimalWriteStorage::FixedLen => unscaled.to_be_bytes().to_vec(),
    }
}

/// Aggregate a file's per-column Parquet statistics over its row groups into the
/// [`FileStats`](crate::manifest::FileStats) persisted in the Delta `Add` action:
/// the row count, and each column's min (min of the groups' mins), max (max of
/// the maxes), and null count (their sum). A column's bound is kept only when
/// every row group carries it, so a bound is never claimed from partial coverage.
/// A column whose type has no decodable min/max (a variant's binary leaf) records
/// none, which is sound: it is simply never pruned by range.
pub(crate) fn aggregate_file_stats(
    row_groups: &[Arc<crate::parquet::RowGroupMetadata>],
) -> crate::manifest::FileStats {
    // The footer path always knows the count -- it is the sum of the row groups'
    // row counts -- so it is always recorded here; only a reload from the log can
    // leave it unknown.
    let num_records = Some(row_groups.iter().map(|rg| rg.num_rows).sum());
    let mut min_values = HashMap::new();
    let mut max_values = HashMap::new();
    let mut null_counts = HashMap::new();

    let Some(first) = row_groups.first() else {
        return crate::manifest::FileStats {
            num_records,
            min_values,
            max_values,
            null_counts,
        };
    };
    for (column, field) in first.schema.fields().iter().enumerate() {
        // Fold the row groups' stats into the file's: the smallest group min, the
        // largest group max, and the sum of null counts. The running min/max
        // borrow into `row_groups` (each is a single-value stat array); a bound or
        // count is kept only when every group carries it, so it is never claimed
        // from partial coverage.
        let mut min: Option<&Scalar<ArrayRef>> = None;
        let mut max: Option<&Scalar<ArrayRef>> = None;
        let mut every_group_has_bounds = true;
        let mut null_sum: i64 = 0;
        let mut every_group_has_null_count = true;
        for rg in row_groups {
            match rg.column_statistics(column) {
                Some(stats) => {
                    match (stats.min.as_ref(), stats.max.as_ref()) {
                        (Some(lo), Some(hi)) => {
                            if min.is_none_or(|current| scalar_lt(lo, current)) {
                                min = Some(lo);
                            }
                            if max.is_none_or(|current| scalar_lt(current, hi)) {
                                max = Some(hi);
                            }
                        }
                        _ => every_group_has_bounds = false,
                    }
                    match stats.null_count {
                        Some(n) => null_sum += n,
                        None => every_group_has_null_count = false,
                    }
                }
                None => {
                    every_group_has_bounds = false;
                    every_group_has_null_count = false;
                }
            }
        }
        if every_group_has_bounds && let (Some(min), Some(max)) = (min, max) {
            min_values.insert(field.name().clone(), min.get().0.slice(0, 1));
            max_values.insert(field.name().clone(), max.get().0.slice(0, 1));
        }
        if every_group_has_null_count {
            null_counts.insert(field.name().clone(), null_sum);
        }
    }
    crate::manifest::FileStats {
        num_records,
        min_values,
        max_values,
        null_counts,
    }
}

/// `a < b` for two single-value scalar bounds, compared in their shared physical
/// type. A null, type mismatch, or kernel error reads as `false`, so folding a
/// column's per-row-group bounds gets a well-defined, never-panicking answer.
fn scalar_lt(a: &Scalar<ArrayRef>, b: &Scalar<ArrayRef>) -> bool {
    arrow_ord::cmp::lt(a as &dyn Datum, b as &dyn Datum)
        .is_ok_and(|result| result.len() == 1 && result.is_valid(0) && result.value(0))
}
