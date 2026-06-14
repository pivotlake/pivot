//! Helpers shared by more than one expression's `compile` impl: type-coercing
//! wrappers around arrow's comparison/arithmetic kernels, and the civil-date
//! math the timestamp parts ([`DatePart`](super::DatePart)) are built on.

use arrow_array::{ArrayRef, BooleanArray, Datum, Scalar};
use arrow_schema::{ArrowError, DataType};

/// Signature shared by arrow's scalar comparison kernels.
pub(crate) type CmpKernel = fn(&dyn Datum, &dyn Datum) -> std::result::Result<BooleanArray, ArrowError>;

/// Run a comparison kernel, coercing both operands to Int64 when their data
/// types differ. The declared logical type (e.g. DATE) and the physical parquet
/// array (e.g. UInt16 day counts) can diverge, and arrow's kernels require
/// matching types; Int64 is a safe common type for every integer/date/timestamp
/// column we compare.
pub(crate) fn compare_coerced(left: &dyn Datum, right: &dyn Datum, kernel: CmpKernel) -> BooleanArray {
    let (la, l_scalar) = left.get();
    let (ra, r_scalar) = right.get();
    if la.data_type() == ra.data_type() {
        kernel(left, right).unwrap()
    } else {
        let lc = arrow::compute::cast(la, &DataType::Int64).unwrap();
        let rc = arrow::compute::cast(ra, &DataType::Int64).unwrap();
        let ld: Box<dyn Datum> = if l_scalar {
            Box::new(Scalar::new(lc))
        } else {
            Box::new(lc)
        };
        let rd: Box<dyn Datum> = if r_scalar {
            Box::new(Scalar::new(rc))
        } else {
            Box::new(rc)
        };
        kernel(ld.as_ref(), rd.as_ref()).unwrap()
    }
}

/// Signature shared by arrow's wrapping arithmetic kernels.
pub(crate) type ArithKernel = fn(&dyn Datum, &dyn Datum) -> std::result::Result<ArrayRef, ArrowError>;

/// Run an arithmetic kernel, coercing both operands to a common type when
/// their data types differ — arrow's kernels require matching types (the
/// kernel falls through to `InvalidArgumentError` on any mismatch). DuckDB
/// normally pre-casts both operands to one type, so the matching-type branch
/// (which runs the kernel natively for integers, floats, and decimals alike)
/// is the common case; the coercion branch only fires when the declared
/// logical type and the physical parquet array diverge.
///
/// The common type must *preserve values*: unconditionally casting to Int64
/// would silently truncate floating-point and decimal operands (`1.5 -> 1`),
/// so we promote to Float64 whenever either side is non-integer and only fall
/// back to Int64 for genuinely integer operands.
pub(crate) fn arith_coerced(left: &dyn Datum, right: &dyn Datum, kernel: ArithKernel) -> ArrayRef {
    let (la, l_scalar) = left.get();
    let (ra, r_scalar) = right.get();
    if la.data_type() == ra.data_type() {
        return kernel(left, right).unwrap();
    }
    let common = if la.data_type().is_integer() && ra.data_type().is_integer() {
        DataType::Int64
    } else {
        DataType::Float64
    };
    let lc = arrow::compute::cast(la, &common).unwrap();
    let rc = arrow::compute::cast(ra, &common).unwrap();
    let ld: Box<dyn Datum> = if l_scalar {
        Box::new(Scalar::new(lc))
    } else {
        Box::new(lc)
    };
    let rd: Box<dyn Datum> = if r_scalar {
        Box::new(Scalar::new(rc))
    } else {
        Box::new(rc)
    };
    kernel(ld.as_ref(), rd.as_ref()).unwrap()
}

/// Seconds in a day / hour, for the time-of-day parts.
pub(crate) const SECS_PER_DAY: i64 = 86_400;
pub(crate) const SECS_PER_HOUR: i64 = 3_600;

/// Convert a day count relative to the Unix epoch (1970-01-01) into a
/// `(year, month, day)` civil date. Howard Hinnant's `civil_from_days`
/// (<http://howardhinnant.github.io/date_algorithms.html>); valid for the full
/// proleptic Gregorian range, including negative (pre-epoch) day counts.
pub(crate) fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // day of era, [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

/// Inverse of [`civil_from_days`]: the epoch-relative day count for a civil
/// date. Used to derive day-of-year and ISO week.
pub(crate) fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Number of ISO 8601 weeks (52 or 53) in the given year. A year has 53 weeks
/// iff it starts on a Thursday, or it is a leap year starting on a Wednesday.
pub(crate) fn iso_weeks_in_year(y: i64) -> i64 {
    let p = |y: i64| (y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)).rem_euclid(7);
    if p(y) == 4 || p(y - 1) == 3 { 53 } else { 52 }
}

/// ISO 8601 week-of-year (1–53) for an epoch-relative day count. Week 1 is the
/// week containing the year's first Thursday; days before it belong to the
/// prior year's last week, and the year's tail can roll into week 1.
pub(crate) fn iso_week(days: i64) -> i64 {
    let (y, ..) = civil_from_days(days);
    let ordinal = days - days_from_civil(y, 1, 1) + 1;
    let iso_dow = (days + 3).rem_euclid(7) + 1;
    let week = (ordinal - iso_dow + 10).div_euclid(7);
    if week < 1 {
        iso_weeks_in_year(y - 1)
    } else if week > iso_weeks_in_year(y) {
        1
    } else {
        week
    }
}
