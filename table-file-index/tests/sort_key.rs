//! A file's bounds on the sort key as bytes that order as the bounds do.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow_schema::DataType;
use table_file_index::SortKeyEncoder;

fn int(value: i64) -> Option<ArrayRef> {
    Some(Arc::new(Int64Array::from(vec![value])))
}

fn text(value: &str) -> Option<ArrayRef> {
    Some(Arc::new(StringArray::from(vec![value])))
}

#[test]
fn bounds_order_as_their_values_do() {
    let encoder = SortKeyEncoder::new(vec![DataType::Int64, DataType::Utf8]).unwrap();
    let encode = |number: i64, name: &str| {
        encoder
            .encode_bound(|position| match position {
                0 => int(number),
                _ => text(name),
            })
            .unwrap()
    };

    let mut keys = vec![encode(7, "b"), encode(-3, "z"), encode(7, "a")];
    keys.sort();

    assert_eq!(keys, [encode(-3, "z"), encode(7, "a"), encode(7, "b")]);
}

#[test]
fn a_bound_of_another_type_is_cast_to_the_columns() {
    let encoder = SortKeyEncoder::new(vec![DataType::Int64]).unwrap();
    let narrow: ArrayRef = Arc::new(Int32Array::from(vec![5]));

    let cast = encoder.encode_bound(|_| Some(narrow.clone())).unwrap();

    assert_eq!(cast, encoder.encode_bound(|_| int(5)).unwrap());
}

#[test]
fn a_column_without_a_bound_orders_first() {
    let encoder = SortKeyEncoder::new(vec![DataType::Int64]).unwrap();

    let unbounded = encoder.encode_bound(|_| None).unwrap();

    assert!(unbounded < encoder.encode_bound(|_| int(i64::MIN)).unwrap());
}

#[test]
fn a_range_is_unknown_without_both_leading_bounds() {
    let encoder = SortKeyEncoder::new(vec![DataType::Int64]).unwrap();

    let range = encoder.encode_range(|_| int(1), |_| None).unwrap();

    assert!(range.maximum.is_none());
}

#[test]
fn a_table_without_a_sort_key_has_no_range() {
    let encoder = SortKeyEncoder::new(Vec::new()).unwrap();

    let range = encoder.encode_range(|_| int(1), |_| int(2)).unwrap();

    assert!(range.minimum.is_empty() && range.maximum.is_none());
}
