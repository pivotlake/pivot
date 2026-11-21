//! The order a table's sort key puts the bounds of its files in, and the
//! order the compacter holds a partition's files in.
//!
//! A file's record holds the bounds its table's metadata recorded for the
//! file's columns in a form of its format's own, and nothing else for the
//! order: a bound on a sort column is read where the record holds it
//! ([`CompactionFile::read_sort_bound`]), as a [`SortValue`], and the bounds
//! of two files on the key are compared as those values
//! ([`compare_sort_bounds`]). Every rule of the order is here and nowhere
//! else: a format says what a bound is, never where it stands.
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
//!
//! # The order a partition's files stand in
//!
//! Files stand by where they start on the table's sort key, their lower
//! bounds on it compared as above. Files that start at the same place,
//! which is every file of a table without a sort key, stand in the order
//! they were committed in, and files of one commit by path, which only
//! makes the order total ([`compare_file_order`]). No format hands its
//! files out in this order: the compacter orders the records it takes.

use std::cmp::Ordering;

use crate::table_format::CompactionFile;

/// The side of a file's rows a bound is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundSide {
    Lower,
    Upper,
}

/// The bound of one file on one sort column, as a value of the column's
/// current type, borrowed from the file's record.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SortValue<'f> {
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
    pub(crate) fn from_single_precision(value: f32) -> Self {
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

/// The bound of `file` on `side` of the sort column at `position`; missing
/// past the last column the file's table is sorted by.
fn read_bound<File: CompactionFile + ?Sized>(
    file: &File,
    position: usize,
    side: BoundSide,
) -> SortValue<'_> {
    if position < file.count_sort_columns() {
        file.read_sort_bound(position, side)
    } else {
        SortValue::Missing
    }
}

/// Order the bound of `left` on `left_side` of the sort key against the
/// bound of `right` on `right_side`: column by column in sort order, the
/// next only when all before it tie.
pub(crate) fn compare_sort_bounds<Left, Right>(
    left: &Left,
    left_side: BoundSide,
    right: &Right,
    right_side: BoundSide,
) -> Ordering
where
    Left: CompactionFile + ?Sized,
    Right: CompactionFile + ?Sized,
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

/// Whether the stretch of the sort key the rows of `file` cover is known:
/// its table has a sort key, and the column the key leads with is bounded
/// by a value on both sides.
pub(crate) fn has_sort_range<File: CompactionFile + ?Sized>(file: &File) -> bool {
    [BoundSide::Lower, BoundSide::Upper]
        .into_iter()
        .all(|side| read_bound(file, 0, side) != SortValue::Missing)
}

/// How `left` and `right` stand in the order the compacter holds a
/// partition's files in: by the lower bound of the table's sort key in
/// each, then by when each was committed, then by path, which only makes
/// the order total.
pub(crate) fn compare_file_order<Left, Right>(left: &Left, right: &Right) -> Ordering
where
    Left: CompactionFile + ?Sized,
    Right: CompactionFile + ?Sized,
{
    compare_sort_bounds(left, BoundSide::Lower, right, BoundSide::Lower)
        .then_with(|| left.commit_sequence().cmp(&right.commit_sequence()))
        .then_with(|| {
            let (left_path, right_path) = (&left.file().path, &right.file().path);
            left_path.as_str().cmp(right_path.as_str())
        })
}

#[cfg(test)]
mod tests {
    use object_storage::{FileRef, ObjectPath};

    use super::SortValue::{Boolean, Float, Integer, Missing, String as Text};
    use super::*;
    use crate::file_entry::FileEntry;

    /// A file whose bounds on the sort key are the values a test gives it.
    struct Record {
        file: FileRef,
        commit_sequence: u64,
        lower: Vec<SortValue<'static>>,
        upper: Vec<SortValue<'static>>,
    }

    impl CompactionFile for Record {
        fn file(&self) -> &FileRef {
            &self.file
        }

        fn row_count(&self) -> Option<u64> {
            Some(1)
        }

        fn commit_sequence(&self) -> u64 {
            self.commit_sequence
        }

        fn count_sort_columns(&self) -> usize {
            self.lower.len()
        }

        fn read_sort_bound(&self, position: usize, side: BoundSide) -> SortValue<'_> {
            match side {
                BoundSide::Lower => self.lower[position],
                BoundSide::Upper => self.upper[position],
            }
        }

        fn build_entry(&self) -> FileEntry {
            FileEntry::new(self.file.clone())
        }
    }

    /// The record of the file at `path`, committed at `commit_sequence`,
    /// whose rows run from `lower` to `upper` on the sort key, a value per
    /// column.
    fn record_between(
        path: &str,
        lower: &[SortValue<'static>],
        upper: &[SortValue<'static>],
        commit_sequence: u64,
    ) -> Record {
        let file = FileRef {
            path: ObjectPath::new(path),
            size: 10,
        };
        Record {
            file,
            commit_sequence,
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        }
    }

    /// The record of the file at `path`, committed at `commit_sequence`,
    /// whose rows all hold `key` on the sort key, a value per column.
    fn record(path: &str, key: &[SortValue<'static>], commit_sequence: u64) -> Record {
        record_between(path, key, key, commit_sequence)
    }

    /// The paths of `records`, in the order the compacter holds a
    /// partition's files in.
    fn order_paths(records: &[Record]) -> Vec<&str> {
        let mut ordered: Vec<&Record> = records.iter().collect();
        ordered.sort_by(|left, right| compare_file_order(*left, *right));
        ordered
            .iter()
            .map(|record| record.file.path.as_str())
            .collect()
    }

    #[test]
    fn the_end_of_one_file_is_ordered_against_the_start_of_another() {
        let reaching = record_between("reaching", &[Integer(1)], &[Integer(20)], 1);
        let short = record_between("short", &[Integer(1)], &[Integer(9)], 1);
        let later = record_between("later", &[Integer(10)], &[Integer(30)], 1);

        let order_ends = |earlier: &Record| {
            compare_sort_bounds(earlier, BoundSide::Upper, &later, BoundSide::Lower)
        };

        assert_eq!(order_ends(&reaching), Ordering::Greater);
        assert_eq!(order_ends(&short), Ordering::Less);
    }

    #[test]
    fn a_bound_of_fewer_columns_orders_before_the_bound_it_begins() {
        let leading = record("leading", &[Integer(5)], 1);
        let whole = record("whole", &[Integer(5), Text("")], 1);

        let order = compare_sort_bounds(&leading, BoundSide::Lower, &whole, BoundSide::Lower);

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

        let known = [&bounded, &open_above, &open_below].map(has_sort_range);

        assert_eq!(known, [true, false, false]);
    }

    #[test]
    fn a_file_of_a_table_without_a_sort_key_has_no_range() {
        let unsorted = record("unsorted", &[], 1);

        let is_known = has_sort_range(&unsorted);

        assert!(!is_known);
    }

    #[test]
    fn files_of_one_commit_stand_by_path() {
        let files = [
            record("z", &[], 1),
            record("b", &[], 2),
            record("a", &[], 2),
        ];

        let ordered = order_paths(&files);

        assert_eq!(ordered, ["z", "a", "b"]);
    }

    #[test]
    fn files_starting_at_either_zero_or_an_infinity_stand_as_the_total_order_has_them() {
        let files = [
            record("zero", &[Float(0.0)], 1),
            record("negative-zero", &[Float(-0.0)], 2),
            record("infinite", &[Float(f64::NEG_INFINITY)], 3),
            record("negative", &[Float(-2.5)], 4),
        ];

        let ordered = order_paths(&files);

        assert_eq!(ordered, ["infinite", "negative", "negative-zero", "zero"]);
    }
}
