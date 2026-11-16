//! Collects a leaf's distinct values for a dictionary, stopping at the first
//! value that takes them past what a dictionary may hold.
//!
//! A leaf whose values are nearly all different, or few but very large, gets
//! no dictionary. Finding that out should cost in proportion to the limit and
//! not to the leaf, so the values are visited one at a time and the collection
//! is abandoned the moment a limit is reached, instead of building the whole
//! dictionary and measuring it afterwards.

use ahash::HashMap;
use arrow_array::cast::AsArray;
use arrow_array::types::{BinaryViewType, StringViewType};
use arrow_array::{Array, ArrayRef, BinaryViewArray, StringArray, StringViewArray, UInt32Array};
use arrow_data::ArrayData;
use arrow_schema::DataType;

use super::super::super::error::WriteResult;
use super::super::plain;

/// What a leaf's distinct values may not reach if they are to be a
/// dictionary.
pub(super) struct DistinctLimits {
    /// How many distinct values.
    pub(super) count: usize,
    /// How many bytes the distinct values take PLAIN-encoded.
    pub(super) plain_bytes: usize,
}

/// A leaf's distinct values and where each of its values is among them.
pub(super) struct DistinctValues {
    /// The distinct values, in the order the leaf first holds them.
    pub(super) values: ArrayRef,
    /// Each of the leaf's values as its position in `values`.
    pub(super) indices: Vec<u32>,
}

/// Reads each value of a leaf as the bytes that tell it apart from the
/// others.
enum ValueBytes<'leaf> {
    /// Fixed-width values back to back, `width` bytes each.
    Fixed {
        bytes: &'leaf [u8],
        width: usize,
    },
    StringView(&'leaf StringViewArray),
    BinaryView(&'leaf BinaryViewArray),
    String(&'leaf StringArray),
}

impl<'leaf> ValueBytes<'leaf> {
    /// The reader for `values`, whose buffers `data` holds, or `None` for a
    /// type that has no dictionary form here.
    fn try_new(values: &'leaf ArrayRef, data: &'leaf ArrayData) -> Option<Self> {
        match values.data_type() {
            DataType::Utf8View => Some(Self::StringView(values.as_byte_view::<StringViewType>())),
            DataType::BinaryView => Some(Self::BinaryView(values.as_byte_view::<BinaryViewType>())),
            DataType::Utf8 => Some(Self::String(values.as_string::<i32>())),
            // A boolean is bits already, which the format stores without a
            // dictionary.
            DataType::Boolean => None,
            data_type => {
                let width = data_type.primitive_width()?;
                let first_value = data.offset() * width;
                Some(Self::Fixed {
                    bytes: &data.buffers()[0].as_slice()[first_value..],
                    width,
                })
            }
        }
    }

    fn value_at(&self, row: usize) -> &'leaf [u8] {
        match self {
            Self::Fixed { bytes, width } => &bytes[row * width..(row + 1) * width],
            Self::StringView(values) => values.value(row).as_bytes(),
            Self::BinaryView(values) => values.value(row),
            Self::String(values) => values.value(row).as_bytes(),
        }
    }
}

/// How many bytes one value of `values` takes PLAIN-encoded when every value
/// takes the same, or `None` for a type whose values are each as long as
/// their bytes plus a length prefix.
fn fixed_plain_width(values: &ArrayRef) -> WriteResult<Option<usize>> {
    if matches!(
        values.data_type(),
        DataType::Utf8View | DataType::BinaryView | DataType::Utf8
    ) {
        return Ok(None);
    }
    let mut one_value = Vec::new();
    plain::encode_into(values.slice(0, 1).as_ref(), &mut one_value)?;
    Ok(Some(one_value.len()))
}

/// The distinct values of `values`, or `None` as soon as they reach either of
/// `limits`, or for a type that has no dictionary form.
pub(super) fn collect_distinct_values(
    values: &ArrayRef,
    limits: &DistinctLimits,
) -> WriteResult<Option<DistinctValues>> {
    if values.is_empty() || limits.count == 0 {
        return Ok(None);
    }
    let data = values.to_data();
    let Some(reader) = ValueBytes::try_new(values, &data) else {
        return Ok(None);
    };
    let fixed_plain_width = fixed_plain_width(values)?;

    let mut position_of_value: HashMap<&[u8], u32> = HashMap::default();
    // The row each distinct value was first seen at.
    let mut first_rows: Vec<u32> = Vec::new();
    let mut plain_bytes = 0;
    let mut indices = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        let value = reader.value_at(row);
        let next_position = first_rows.len() as u32;
        let position = *position_of_value.entry(value).or_insert(next_position);
        if position == next_position {
            first_rows.push(row as u32);
            plain_bytes += fixed_plain_width.unwrap_or(size_of::<u32>() + value.len());
            if first_rows.len() >= limits.count || plain_bytes >= limits.plain_bytes {
                return Ok(None);
            }
        }
        indices.push(position);
    }
    let distinct = arrow_select::take::take(values.as_ref(), &UInt32Array::from(first_rows), None)?;
    Ok(Some(DistinctValues {
        values: distinct,
        indices,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use std::sync::Arc;

    fn limits(count: usize, plain_bytes: usize) -> DistinctLimits {
        DistinctLimits { count, plain_bytes }
    }

    #[test]
    fn repeated_values_are_kept_once_in_first_seen_order() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![7, 3, 7, 7, 3, 9]));

        let distinct = collect_distinct_values(&values, &limits(100, 1_000))
            .unwrap()
            .unwrap();

        assert_eq!(
            distinct
                .values
                .as_primitive::<arrow_array::types::Int64Type>()
                .values(),
            &[7, 3, 9]
        );
        assert_eq!(distinct.indices, [0, 1, 0, 0, 1, 2]);
    }

    #[test]
    fn too_many_distinct_values_give_no_dictionary() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4]));

        let distinct = collect_distinct_values(&values, &limits(3, 1_000)).unwrap();

        assert!(distinct.is_none());
    }

    #[test]
    fn distinct_values_too_large_together_give_no_dictionary() {
        let values: ArrayRef = Arc::new(StringViewArray::from(vec![
            "a value of some length",
            "another value of some length",
        ]));

        let distinct = collect_distinct_values(&values, &limits(100, 40)).unwrap();

        assert!(distinct.is_none());
    }
}
