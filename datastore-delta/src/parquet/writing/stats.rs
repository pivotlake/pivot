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

/// How many bytes of a string bound are worth keeping.
///
/// A bound only has to *contain* the column's range, never equal it, so a long
/// value can be replaced by a short one that still encloses everything it
/// bounded. Anything shorter than this is kept exactly, which covers the
/// identifier-like columns that pruning actually uses; it is free-text columns
/// (a log body, a serialized document) that would otherwise pin a whole value
/// per column per file, in memory for as long as the catalog holds the file and
/// again in every log commit and checkpoint.
const MAX_STAT_BYTES: usize = 64;

/// The longest prefix of `value` within [`MAX_STAT_BYTES`] that ends on a
/// character boundary, or `None` when `value` already fits.
fn stat_prefix(value: &str) -> Option<&str> {
    if value.len() <= MAX_STAT_BYTES {
        return None;
    }
    let mut end = MAX_STAT_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    Some(&value[..end])
}

/// The code point after `c`, stepping over the surrogate range that is not a
/// legal `char`. `None` only at the top of the Unicode range.
fn next_code_point(c: char) -> Option<char> {
    let next = match c as u32 + 1 {
        0xD800 => 0xE000,
        code => code,
    };
    char::from_u32(next)
}

/// A shortened stand-in for a `min` bound: a prefix of the true minimum. Cutting
/// the tail off a string can only lower it, so the result still sits at or below
/// every value the original bounded. `None` leaves the value as it is.
fn shorten_lower_bound(value: &str) -> Option<String> {
    stat_prefix(value).map(str::to_owned)
}

/// A shortened stand-in for a `max` bound: a prefix of the true maximum with its
/// last character raised to the next code point, which sorts above every string
/// that starts with that prefix (UTF-8 compares byte-wise in code point order).
/// Characters that cannot be raised are dropped and the one before them is
/// raised instead; `None` when nothing can be (an all-`char::MAX` prefix), and
/// the caller then keeps the value untouched rather than record a bound that
/// could sit below a real one.
fn shorten_upper_bound(value: &str) -> Option<String> {
    let mut bound = stat_prefix(value)?.to_owned();
    while let Some(last) = bound.pop() {
        if let Some(raised) = next_code_point(last) {
            bound.push(raised);
            return Some(bound);
        }
    }
    None
}

/// A column's min and max, as single-element Arrow arrays.
///
/// `None` when the column is empty or all-null, or when its type is not one the
/// write path (and so the reader's `decode_scalar`) supports — a binary leaf, for
/// instance. Such a column simply records no min/max and is never pruned by
/// range, which costs a scan but is never unsound.
///
/// String bounds past [`MAX_STAT_BYTES`] are widened to short ones (see
/// [`shorten_lower_bound`] / [`shorten_upper_bound`]); the range they describe
/// still encloses the column, so pruning stays sound and only gets coarser.
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
            let lo = a.iter().flatten().min()?;
            let hi = a.iter().flatten().max()?;
            let (lo_short, hi_short) = (shorten_lower_bound(lo), shorten_upper_bound(hi));
            let lo: ArrayRef = Arc::new(<$arr>::from(vec![lo_short.as_deref().unwrap_or(lo)]));
            let hi: ArrayRef = Arc::new(<$arr>::from(vec![hi_short.as_deref().unwrap_or(hi)]));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(values: Vec<&str>) -> (String, String) {
        let array: ArrayRef = Arc::new(StringViewArray::from(values));
        let (lo, hi) = column_min_max(&array).unwrap();
        let text = |a: ArrayRef| {
            a.as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(0)
                .to_owned()
        };
        (text(lo), text(hi))
    }

    #[test]
    fn short_string_bounds_are_kept_exactly() {
        let (lo, hi) = bounds(vec!["banana", "apple", "cherry"]);

        assert_eq!(lo, "apple");
        assert_eq!(hi, "cherry");
    }

    #[test]
    fn long_string_bounds_shrink_but_still_enclose_the_column() {
        let low = format!("aaa{}", "x".repeat(200));
        let high = format!("zzz{}", "x".repeat(200));

        let (lo, hi) = bounds(vec![&low, &high]);

        assert!(lo.len() <= MAX_STAT_BYTES);
        assert!(
            lo.as_str() <= low.as_str(),
            "{lo:?} must not exceed the min"
        );
        assert!(
            hi.as_str() >= high.as_str(),
            "{hi:?} must not fall below the max"
        );
    }

    /// Shortening both bounds to the same prefix would make them compare equal,
    /// which `NotEqual` pruning reads as "every row holds this one value".
    /// Raising the upper bound is what keeps the range non-empty.
    #[test]
    fn bounds_sharing_a_prefix_stay_distinct() {
        let shared = "s".repeat(100);

        let (lo, hi) = bounds(vec![&format!("{shared}a"), &format!("{shared}b")]);

        assert!(lo < hi, "{lo:?} must sort below {hi:?}");
    }

    /// Three-byte characters straddle the byte limit, so a naive cut lands
    /// mid-character and yields bytes that are not a string at all.
    #[test]
    fn shortening_splits_on_character_boundaries() {
        let value = "\u{4e2d}".repeat(100);

        let lo = shorten_lower_bound(&value).unwrap();
        let hi = shorten_upper_bound(&value).unwrap();

        assert!(value.starts_with(&lo));
        assert!(lo.len() <= MAX_STAT_BYTES);
        assert!(hi.as_str() > value.as_str());
    }

    #[test]
    fn an_upper_bound_at_the_top_of_unicode_keeps_the_value() {
        let value = char::MAX.to_string().repeat(40);

        assert_eq!(shorten_upper_bound(&value), None);
    }
}
