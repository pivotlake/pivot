//! The bound of a file on one column of its table's sort key, as every
//! table format hands it to whoever orders files by their bounds, and the
//! order those bounds stand in.
//!
//! A format holds the bounds its metadata recorded for a file in a form of
//! its own and reads one as a [`SortValue`]: it says what a bound is, never
//! where it stands. Every rule of the order is here and nowhere else:
//!
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

/// Widen `value` to double precision. A number widens exactly. A NaN keeps
/// its sign and nothing else, so a NaN of the narrow width reads as one
/// value wherever it is widened.
pub fn widen_single_precision(value: f32) -> f64 {
    /// The bits of the positive quiet NaN with no payload.
    const QUIET_NAN_BITS: u64 = 0x7ff8 << 48;
    /// The bit a negative float has set.
    const SIGN_BIT: u64 = 1 << 63;

    if !value.is_nan() {
        return f64::from(value);
    }
    let sign = if value.is_sign_negative() {
        SIGN_BIT
    } else {
        0
    };
    f64::from_bits(QUIET_NAN_BITS | sign)
}

impl SortValue<'_> {
    /// `value` as a float bound, widened as [`widen_single_precision`] has
    /// it.
    pub fn from_single_precision(value: f32) -> Self {
        Self::Float(widen_single_precision(value))
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

#[cfg(test)]
mod tests {
    use super::SortValue::{Boolean, Float, Integer, Missing, String as Text};
    use super::*;

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
}
