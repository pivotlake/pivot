//! Partition transforms, and what a comparison on a column says about its
//! transformed values. The transforms follow
//! <https://iceberg.apache.org/spec/#partition-transforms>.

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryViewArray, Datum, Scalar, StringViewArray};
use arrow_schema::{DataType, TimeUnit};
use chrono::{Datelike, NaiveDate, TimeDelta};

use crate::{Comparison, scalar};

/// How the bounded values derive from the column's. Calendar values are UTC
/// offsets from the Unix epoch: years, months, dates, or hours. A column type
/// the transform is not defined for, or a zero bucket count or width, proves
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// The column's own values.
    Identity,
    Bucket(u32),
    Truncate(u32),
    Year,
    Month,
    Day,
    Hour,
}

impl Transform {
    /// Comparisons on the transformed value that every row satisfying
    /// `column <comparison> constant` also satisfies. Empty when the transform
    /// keeps nothing of the comparison.
    pub(crate) fn project(
        self,
        comparison: Comparison,
        constant: &Scalar<ArrayRef>,
    ) -> Vec<(Comparison, Scalar<ArrayRef>)> {
        if self == Self::Identity {
            return vec![(comparison, constant.clone())];
        }
        // A many-to-one transform keeps nothing of `<>`: a different column
        // value can share the transformed value. A hash keeps no order either.
        if comparison == Comparison::NotEqual
            || (matches!(self, Self::Bucket(_)) && comparison != Comparison::Equal)
        {
            return Vec::new();
        }
        // The other transforms are monotonic, so a range keeps its direction
        // but loses its strictness. For a discrete type, first move to the
        // nearest value the strict comparison admits: `ts < midnight` then
        // excludes the day that starts at midnight.
        let (comparison, nearest) = match comparison {
            Comparison::Less => (Comparison::LessEqual, step(constant, -1)),
            Comparison::Greater => (Comparison::GreaterEqual, step(constant, 1)),
            other => (other, None),
        };
        let Some(value) = self.apply(nearest.as_ref().unwrap_or(constant)) else {
            return Vec::new();
        };
        // Older calendar writers could round negative offsets toward zero.
        // Keep their adjacent partition too. Some readers/writers also rounded
        // pre-epoch subsecond days upward, including timestamps with a zone.
        // The extra partition is harmless for writers that always floor.
        let legacy_calendar = matches!(self, Self::Year | Self::Month)
            || (self == Self::Day
                && matches!(constant.get().0.data_type(), DataType::Timestamp(..)));
        if legacy_calendar
            && let Some(offset) = scalar::integer(value.get().0).filter(|offset| *offset < 0)
            && let Some(adjacent) = scalar::from_integer(offset + 1, value.get().0.data_type())
        {
            match comparison {
                Comparison::Equal => {
                    return vec![
                        (Comparison::GreaterEqual, value),
                        (Comparison::LessEqual, adjacent),
                    ];
                }
                Comparison::LessEqual => return vec![(comparison, adjacent)],
                _ => {}
            }
        }
        vec![(comparison, value)]
    }

    /// The transformed value of one column value.
    pub(crate) fn apply(self, value: &Scalar<ArrayRef>) -> Option<Scalar<ArrayRef>> {
        let array = value.get().0;
        match self {
            Self::Identity => Some(value.clone()),
            Self::Bucket(count) => bucket(array, count),
            Self::Truncate(width) => truncate(array, width),
            Self::Year | Self::Month | Self::Day | Self::Hour => calendar(array, self),
        }
    }
}

/// The value `delta` away from a constant of a discrete type, when the type
/// has one.
fn step(constant: &Scalar<ArrayRef>, delta: i128) -> Option<Scalar<ArrayRef>> {
    let array = constant.get().0;
    scalar::from_integer(scalar::integer(array)? + delta, array.data_type())
}

fn ticks_per_second(unit: &TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    }
}

fn calendar(array: &dyn Array, transform: Transform) -> Option<Scalar<ArrayRef>> {
    let value = scalar::integer(array)?;
    let days = match array.data_type() {
        DataType::Date32 if transform != Transform::Hour => value,
        DataType::Timestamp(unit, _) => {
            let hour = value.div_euclid(ticks_per_second(unit) * 3_600);
            if transform == Transform::Hour {
                return scalar::from_integer(hour, &DataType::Int32);
            }
            hour.div_euclid(24)
        }
        _ => return None,
    };
    if transform == Transform::Day {
        return scalar::from_integer(days, &DataType::Date32);
    }
    let date = NaiveDate::from_ymd_opt(1970, 1, 1)?
        .checked_add_signed(TimeDelta::try_days(days.try_into().ok()?)?)?;
    let years = i128::from(date.year()) - 1970;
    let offset = match transform {
        Transform::Year => years,
        Transform::Month => years * 12 + i128::from(date.month0()),
        _ => return None,
    };
    scalar::from_integer(offset, &DataType::Int32)
}

fn truncate(array: &dyn Array, width: u32) -> Option<Scalar<ArrayRef>> {
    if width == 0 {
        return None;
    }
    match array.data_type() {
        DataType::Int32 | DataType::Int64 | DataType::Decimal64(..) | DataType::Decimal128(..) => {
            let value = scalar::integer(array)?;
            let value = value.checked_sub(value.rem_euclid(i128::from(width)))?;
            scalar::from_integer(value, array.data_type())
        }
        data_type if data_type.is_string() => {
            let value = string(array)?;
            let end = value
                .char_indices()
                .nth(width as usize)
                .map_or(value.len(), |(i, _)| i);
            let value = StringViewArray::from(vec![&value[..end]]);
            Some(Scalar::new(arrow_cast::cast(&value, data_type).ok()?))
        }
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            let value = binary(array)?;
            let value = BinaryViewArray::from(vec![&value[..value.len().min(width as usize)]]);
            Some(Scalar::new(
                arrow_cast::cast(&value, array.data_type()).ok()?,
            ))
        }
        _ => None,
    }
}

fn string(array: &dyn Array) -> Option<&str> {
    Some(match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value(0),
        DataType::LargeUtf8 => array.as_string::<i64>().value(0),
        DataType::Utf8View => array.as_string_view().value(0),
        _ => return None,
    })
}

fn binary(array: &dyn Array) -> Option<&[u8]> {
    Some(match array.data_type() {
        DataType::Binary => array.as_binary::<i32>().value(0),
        DataType::LargeBinary => array.as_binary::<i64>().value(0),
        DataType::BinaryView => array.as_binary_view().value(0),
        DataType::FixedSizeBinary(_) => array.as_fixed_size_binary().value(0),
        _ => return None,
    })
}

fn bucket(array: &dyn Array, count: u32) -> Option<Scalar<ArrayRef>> {
    if count == 0 || count > i32::MAX as u32 {
        return None;
    }
    let hash = |mut bytes: &[u8]| murmur3::murmur3_32(&mut bytes, 0).ok();
    let hashed = match array.data_type() {
        // Int32 is sign-extended so promotion to Int64 preserves the bucket.
        DataType::Int32 | DataType::Int64 | DataType::Date32 => {
            hash(&i64::try_from(scalar::integer(array)?).ok()?.to_le_bytes())?
        }
        DataType::Timestamp(unit, _) => {
            let ticks = scalar::integer(array)?;
            let micros = (ticks * 1_000_000).div_euclid(ticks_per_second(unit));
            hash(&i64::try_from(micros).ok()?.to_le_bytes())?
        }
        DataType::Decimal64(..) | DataType::Decimal128(..) => {
            // Minimum two's-complement big-endian representation; retain a
            // leading sign byte when dropping it would change the sign.
            let bytes = scalar::integer(array)?.to_be_bytes();
            let mut start = 0;
            while start + 1 < bytes.len()
                && ((bytes[start] == 0 && bytes[start + 1] & 0x80 == 0)
                    || (bytes[start] == 0xff && bytes[start + 1] & 0x80 != 0))
            {
                start += 1;
            }
            hash(&bytes[start..])?
        }
        data_type if data_type.is_string() => hash(string(array)?.as_bytes())?,
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => hash(binary(array)?)?,
        _ => return None,
    };
    scalar::from_integer(i128::from((hashed & 0x7fff_ffff) % count), &DataType::Int32)
}

#[cfg(test)]
mod tests;
