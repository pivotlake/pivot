//! The order a table's sort key puts the bounds of its files in.

mod support;

use std::cmp::Ordering;

use support::{Record, record, record_between};
use table_file_index::SortValue::{Boolean, Float, Integer, Missing, String as Text};
use table_file_index::{BoundSide, SortValue, compare_sort_bounds, has_sort_range};

#[test]
fn the_end_of_one_file_is_ordered_against_the_start_of_another() {
    let reaching = record_between("reaching", &[Integer(1)], &[Integer(20)], 1);
    let short = record_between("short", &[Integer(1)], &[Integer(9)], 1);
    let later = record_between("later", &[Integer(10)], &[Integer(30)], 1);

    let order_ends = |earlier: &Record| {
        compare_sort_bounds(earlier, BoundSide::Upper, later.as_ref(), BoundSide::Lower)
    };

    assert_eq!(order_ends(reaching.as_ref()), Ordering::Greater);
    assert_eq!(order_ends(short.as_ref()), Ordering::Less);
}

#[test]
fn a_bound_of_fewer_columns_orders_before_the_bound_it_begins() {
    let leading = record("leading", &[Integer(5)], 1);
    let whole = record("whole", &[Integer(5), Text("")], 1);

    let order = compare_sort_bounds(
        leading.as_ref(),
        BoundSide::Lower,
        whole.as_ref(),
        BoundSide::Lower,
    );

    assert_eq!(order, Ordering::Less);
}

#[test]
fn values_of_each_kind_order_by_value_and_after_a_missing_bound() {
    let ascending = [
        [Missing, Boolean(false), Boolean(true)],
        [Missing, Integer(-3), Integer(i128::from(u64::MAX))],
        [Missing, Float(f64::NEG_INFINITY), Float(f64::NAN)],
        [Missing, Text(""), Text("a\u{0}")],
    ];

    let is_ascending = ascending.map(|values| values.is_sorted_by(|left, right| left < right));

    assert_eq!(is_ascending, [true; 4]);
}

#[test]
fn a_narrow_nan_reads_as_one_value_whatever_else_its_bits_say() {
    let quiet = SortValue::from_single_precision(f32::NAN);
    let with_payload = SortValue::from_single_precision(f32::from_bits(0x7fc0_0001));
    let negative = SortValue::from_single_precision(-f32::NAN);

    let orders = [quiet.cmp(&with_payload), negative.cmp(&quiet)];

    assert_eq!(orders, [Ordering::Equal, Ordering::Less]);
    assert!(negative < Float(f64::NEG_INFINITY) && Float(f64::INFINITY) < quiet);
}

#[test]
fn a_range_is_unknown_without_both_leading_bounds() {
    let bounded = record_between("bounded", &[Integer(1), Missing], &[Integer(2), Missing], 1);
    let open_above = record_between("open-above", &[Integer(1)], &[Missing], 1);
    let open_below = record_between("open-below", &[Missing], &[Integer(2)], 1);

    let known = [
        has_sort_range(bounded.as_ref()),
        has_sort_range(open_above.as_ref()),
        has_sort_range(open_below.as_ref()),
    ];

    assert_eq!(known, [true, false, false]);
}

#[test]
fn a_file_of_a_table_without_a_sort_key_has_no_range() {
    let unsorted = record("unsorted", &[], 1);

    let is_known = has_sort_range(unsorted.as_ref());

    assert!(!is_known);
}
