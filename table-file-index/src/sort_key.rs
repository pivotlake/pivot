//! The order a table's sort key puts the bounds of its files in.
//!
//! A file's record holds the bounds its table's metadata recorded for the
//! file's columns, and nothing else for the order: a bound on a sort column
//! is read where the record holds it
//! ([`IndexedRecord::read_sort_bound`]), as a [`SortValue`], and the bounds
//! of two files on the key are compared as those values
//! ([`compare_sort_bounds`]).
//!
//! # How two bounds on the sort key compare
//!
//! - The sort columns are compared in the order the table is sorted by
//!   them, the next only when all before it tie.
//! - A column the file has no bound on orders before every value
//!   ([`SortValue::Missing`]): its metadata records none, or a NULL, or
//!   something that is not a value of the column's type.
//! - `false` orders before `true`; an integer, a date, a timestamp and a
//!   decimal by value; a string by its bytes, so a prefix orders before
//!   what extends it.
//! - Floats order as IEEE 754's total order has them: negative before
//!   positive, `-0.0` before `+0.0`, a negative NaN before every number and
//!   a positive one after.
//! - A bound recorded in another type than its column has now, that of a
//!   file written before the column was widened, is read as its value in
//!   the column's current type, so it orders by value among the others.
//!
//! The order is total whatever the records hold: every bound reads as one
//! [`SortValue`], the same for as long as its record lives, and the order
//! of those values is total.

use std::cmp::Ordering;

use crate::file_index::IndexedRecord;

/// The side of a file's rows a bound is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundSide {
    Lower,
    Upper,
}

/// The bound of one file on one sort column, as a value of the column's
/// current type, borrowed from the file's record.
#[derive(Clone, Copy, Debug)]
pub enum SortValue<'f> {
    /// No bound: the file's metadata records none for the column, or a
    /// NULL, or something that does not read as a value of the column's
    /// type. Orders before every value.
    Missing,
    Boolean(bool),
    /// An integer of any width, a date as its days and a timestamp as its
    /// microseconds since the epoch, or the unscaled value of a decimal at
    /// its column's scale.
    Integer(i128),
    /// A float of either width, ordered as IEEE 754's total order has it.
    Float(f64),
    String(&'f str),
}

impl SortValue<'_> {
    /// `value` as a float bound. A NaN keeps its sign and nothing else, so
    /// a NaN of the narrow width reads as one value wherever it is widened.
    pub fn from_single_precision(value: f32) -> Self {
        /// The bits of the positive quiet NaN with no payload.
        const QUIET_NAN_BITS: u64 = 0x7ff8 << 48;
        /// The bit a negative float has set.
        const SIGN_BIT: u64 = 1 << 63;

        if !value.is_nan() {
            return Self::Float(f64::from(value));
        }
        let sign = if value.is_sign_negative() {
            SIGN_BIT
        } else {
            0
        };
        Self::Float(f64::from_bits(QUIET_NAN_BITS | sign))
    }

    /// Where the kind of the value stands among the kinds, which is what
    /// orders two values of different kinds: a missing bound first.
    fn rank_kind(&self) -> u8 {
        match self {
            Self::Missing => 0,
            Self::Boolean(_) => 1,
            Self::Integer(_) => 2,
            Self::Float(_) => 3,
            Self::String(_) => 4,
        }
    }
}

impl Ord for SortValue<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Boolean(left), Self::Boolean(right)) => left.cmp(right),
            (Self::Integer(left), Self::Integer(right)) => left.cmp(right),
            (Self::Float(left), Self::Float(right)) => left.total_cmp(right),
            (Self::String(left), Self::String(right)) => left.cmp(right),
            _ => self.rank_kind().cmp(&other.rank_kind()),
        }
    }
}

impl PartialOrd for SortValue<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SortValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for SortValue<'_> {}

/// The bound of `record` on `side` of the sort column at `position`;
/// missing past the last column the record's table is sorted by.
fn read_bound<Record: IndexedRecord + ?Sized>(
    record: &Record,
    position: usize,
    side: BoundSide,
) -> SortValue<'_> {
    if position < record.count_sort_columns() {
        record.read_sort_bound(position, side)
    } else {
        SortValue::Missing
    }
}

/// Order the bound of `left` on `left_side` of the sort key against the
/// bound of `right` on `right_side`: column by column in sort order, the
/// next only when all before it tie.
pub fn compare_sort_bounds<Left, Right>(
    left: &Left,
    left_side: BoundSide,
    right: &Right,
    right_side: BoundSide,
) -> Ordering
where
    Left: IndexedRecord + ?Sized,
    Right: IndexedRecord + ?Sized,
{
    let column_count = left.count_sort_columns().max(right.count_sort_columns());
    for position in 0..column_count {
        let left_bound = read_bound(left, position, left_side);
        let order = left_bound.cmp(&read_bound(right, position, right_side));
        if order.is_ne() {
            return order;
        }
    }
    Ordering::Equal
}

/// Whether the stretch of the sort key the rows of `record`'s file cover
/// is known: its table has a sort key, and the column the key leads with
/// is bounded by a value on both sides.
pub fn has_sort_range<Record: IndexedRecord + ?Sized>(record: &Record) -> bool {
    [BoundSide::Lower, BoundSide::Upper]
        .into_iter()
        .all(|side| read_bound(record, 0, side) != SortValue::Missing)
}
