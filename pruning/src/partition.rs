//! Inclusive partition projection, independent of storage metadata. Transform
//! values follow <https://iceberg.apache.org/spec/#partition-transforms>; both
//! catalog adapters describe their partitions using these same operations.

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryViewArray, Datum, Scalar, StringViewArray};
use arrow_schema::{DataType, TimeUnit};
use chrono::{Datelike, NaiveDate, TimeDelta};

use crate::scalar::{self, BoundConstant};
use crate::{Comparison, PruningPredicate};

/// The operation whose values a partition's bounds describe. Calendar values
/// are UTC offsets from the Unix epoch: years, months, dates, or hours. Bucket
/// hashing and truncation use the encodings documented in the Iceberg spec.
/// Unsupported source types or zero bucket counts/widths cannot prove exclusions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionTransform {
    Identity,
    Bucket(u32),
    Truncate(u32),
    Year,
    Month,
    Day,
    Hour,
}

impl PartitionTransform {
    pub(crate) fn project(
        self,
        source: &DataType,
        compare: Comparison,
        constant: &Scalar<ArrayRef>,
    ) -> PruningPredicate {
        let mut value = match scalar::bind(constant, source, compare) {
            BoundConstant::Value(value) => value,
            BoundConstant::Always(value) => return PruningPredicate::Always(value),
            BoundConstant::Unknown => return PruningPredicate::Always(true),
        };
        let comparison = |compare, value| PruningPredicate::Compare { compare, value };
        if self == Self::Identity {
            return comparison(compare, value);
        }
        // A many-to-one transform cannot project <> inclusively: a different
        // source value can still have the same partition value. Hash ordering
        // also says nothing about the source's ordering.
        if compare == Comparison::NotEqual
            || (matches!(self, Self::Bucket(_)) && compare != Comparison::Equal)
        {
            return PruningPredicate::Always(true);
        }
        let projected_compare = match compare {
            Comparison::Less | Comparison::Greater => {
                // For discrete types, move to the last/first source value
                // satisfying the strict comparison BEFORE transforming it.
                // For strings, retaining the boundary's prefix is conservative.
                let (array, _) = value.get();
                if let Some(integer) = scalar::integer(array)
                    && let Some((min, max)) = scalar::domain(source)
                {
                    let next = if compare == Comparison::Less {
                        integer.checked_sub(1).filter(|value| *value >= min)
                    } else {
                        integer.checked_add(1).filter(|value| *value <= max)
                    };
                    let Some(next) = next else {
                        return PruningPredicate::Always(false);
                    };
                    let Some(next) = scalar::from_integer(next, source) else {
                        return PruningPredicate::Always(true);
                    };
                    value = next;
                }
                if compare == Comparison::Less {
                    Comparison::LessEqual
                } else {
                    Comparison::GreaterEqual
                }
            }
            _ => compare,
        };
        let Some(value) = self.apply(&value) else {
            return PruningPredicate::Always(true);
        };
        // Older calendar writers could round negative offsets toward zero.
        // Keep their adjacent partition too. Some readers/writers also rounded
        // pre-epoch subsecond days upward, including timestamps with a zone.
        // The extra partition is harmless for writers that always floor.
        let legacy_calendar = matches!(self, Self::Year | Self::Month)
            || (self == Self::Day && matches!(source, DataType::Timestamp(..)));
        if legacy_calendar
            && let Some(offset) = scalar::integer(value.get().0).filter(|value| *value < 0)
            && let Some(adjacent) = scalar::from_integer(offset + 1, value.get().0.data_type())
        {
            match compare {
                Comparison::Equal => {
                    return PruningPredicate::Or(vec![
                        comparison(compare, value),
                        comparison(compare, adjacent),
                    ]);
                }
                Comparison::Less | Comparison::LessEqual => {
                    return comparison(projected_compare, adjacent);
                }
                _ => {}
            }
        }
        comparison(projected_compare, value)
    }

    fn apply(self, value: &Scalar<ArrayRef>) -> Option<Scalar<ArrayRef>> {
        let array = value.get().0;
        match self {
            Self::Identity => Some(value.clone()),
            Self::Bucket(count) => bucket(array, count),
            Self::Truncate(width) => truncate(array, width),
            Self::Year | Self::Month | Self::Day | Self::Hour => calendar(array, self),
        }
    }
}

fn ticks_per_second(unit: &TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    }
}

fn calendar(array: &dyn Array, transform: PartitionTransform) -> Option<Scalar<ArrayRef>> {
    let value = scalar::integer(array)?;
    let days = match array.data_type() {
        DataType::Date32 if transform != PartitionTransform::Hour => value,
        DataType::Timestamp(unit, _) => {
            let hour = value.div_euclid(ticks_per_second(unit) * 3_600);
            if transform == PartitionTransform::Hour {
                return scalar::from_integer(hour, &DataType::Int32);
            }
            hour.div_euclid(24)
        }
        _ => return None,
    };
    if transform == PartitionTransform::Day {
        return scalar::from_integer(days, &DataType::Date32);
    }
    let date = NaiveDate::from_ymd_opt(1970, 1, 1)?
        .checked_add_signed(TimeDelta::try_days(days.try_into().ok()?)?)?;
    let years = i128::from(date.year()) - 1970;
    let offset = match transform {
        PartitionTransform::Year => years,
        PartitionTransform::Month => years * 12 + i128::from(date.month0()),
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
