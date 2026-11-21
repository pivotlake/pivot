//! One partition of a table, which every pass names by the partition
//! itself.
//!
//! A partition is the files written under one partition spec with one
//! tuple, so an unpartitioned table is a single partition. Partitions are
//! told apart, ordered and hashed by the spec's id and then the tuple's
//! values.
//!
//! A value is told from another by what it is, whatever width it is stored
//! in ([`ValueClass`]): a column a spec partitions by may be widened, an
//! `int` to a `long` or a `float` to a `double`, and a manifest written
//! before holds the narrow value where one written after holds the wide
//! one. Both are the one partition, which each holder keeps in the form it
//! read: a tuple is never rewritten.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use iceberg::spec::{Literal, PrimitiveLiteral, Struct};
use parquet_engine::{measure_allocation, measure_shared_allocation, widen_single_precision};

/// One partition of a table: the files written under one partition spec
/// with one tuple.
#[derive(Debug)]
pub struct Partition {
    spec_id: i32,
    tuple: Struct,
}

impl Partition {
    /// The partition spec the tuple is laid out by.
    pub fn spec_id(&self) -> i32 {
        self.spec_id
    }

    pub fn tuple(&self) -> &Struct {
        &self.tuple
    }

    /// The partition of `tuple`, primitive values and NULLs alone
    /// ([`holds_only_primitive_values`]), under the spec `spec_id`.
    pub(crate) fn new(spec_id: i32, tuple: Struct) -> Self {
        Self { spec_id, tuple }
    }

    /// An estimate of the memory the partition holds, itself included, as
    /// one shared allocation: its tuple.
    pub(crate) fn estimate_heap_bytes(&self) -> usize {
        let tuple = self.tuple.fields();
        measure_shared_allocation(size_of::<Self>())
            + measure_allocation(size_of_val(tuple))
            + tuple
                .iter()
                .map(|value| match value {
                    Some(Literal::Primitive(value)) => estimate_literal_heap_bytes(value),
                    _ => 0,
                })
                .sum::<usize>()
    }
}

/// Partitions are equal when their specs and the values of their tuples
/// are, each value as its `ValueClass` tells it.
impl PartialEq for Partition {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Partition {}

impl PartialOrd for Partition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// By spec, then value by value in the tuple's order, a NULL before any
/// value and a tuple before one it is the start of.
impl Ord for Partition {
    fn cmp(&self, other: &Self) -> Ordering {
        let spec_order = self.spec_id.cmp(&other.spec_id);
        if spec_order != Ordering::Equal {
            return spec_order;
        }
        let values = self.tuple.fields();
        let other_values = other.tuple.fields();
        for (value, other_value) in values.iter().zip(other_values) {
            let value_order = match (read_primitive(value), read_primitive(other_value)) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(value), Some(other_value)) => {
                    ValueClass::of(value).cmp(&ValueClass::of(other_value))
                }
            };
            if value_order != Ordering::Equal {
                return value_order;
            }
        }
        values.len().cmp(&other_values.len())
    }
}

/// What tells partitions apart: the spec, then each value of the tuple as
/// its `ValueClass` tells it.
impl Hash for Partition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.spec_id.hash(state);
        for value in self.tuple.fields() {
            match read_primitive(value) {
                None => state.write_u8(0),
                Some(value) => {
                    state.write_u8(1);
                    ValueClass::of(value).hash(state);
                }
            }
        }
    }
}

/// A value of a partition's tuple as partitions are told apart by it: what
/// the value is, whatever width it is stored in.
///
/// An `int` and a `long` of one number are one value, and so are a `float`
/// and the `double` it widens to exactly: a `float` 0.1 and a `double` 0.1
/// are two numbers, and stay two values. A decimal is its unscaled value at
/// any precision, which is how a tuple holds it. A NaN of the narrow width
/// keeps its sign and nothing else when widened
/// ([`widen_single_precision`]), so every narrow NaN of one sign
/// is one value, the `double` NaN of that sign with no payload. Nothing
/// else is one value in two forms: a date and a timestamp are an integer
/// each and are told apart by their number alone.
///
/// Floats order in the total order of their widened bits, which tells a
/// negative zero from a positive one; every other value as its type orders;
/// and values of two kinds by their kinds, an integer before a float.
enum ValueClass<'v> {
    Integer(i64),
    Float(f64),
    /// A value with one stored form.
    Other(&'v PrimitiveLiteral),
}

impl<'v> ValueClass<'v> {
    fn of(value: &'v PrimitiveLiteral) -> Self {
        match value {
            PrimitiveLiteral::Int(value) => Self::Integer(i64::from(*value)),
            PrimitiveLiteral::Long(value) => Self::Integer(*value),
            PrimitiveLiteral::Float(value) => Self::Float(widen_single_precision(value.0)),
            PrimitiveLiteral::Double(value) => Self::Float(value.0),
            other => Self::Other(other),
        }
    }

    /// Where the kind of the value stands among the kinds, which is what
    /// orders two values of different kinds.
    fn rank_kind(&self) -> u8 {
        match self {
            Self::Other(PrimitiveLiteral::Boolean(_)) => 0,
            Self::Integer(_) => 1,
            Self::Float(_) => 2,
            Self::Other(_) => 3,
        }
    }
}

impl Ord for ValueClass<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Integer(value), Self::Integer(other)) => value.cmp(other),
            (Self::Float(value), Self::Float(other)) => value.total_cmp(other),
            (Self::Other(value), Self::Other(other)) => value
                .partial_cmp(other)
                .expect("values that are not floats always order"),
            _ => self.rank_kind().cmp(&other.rank_kind()),
        }
    }
}

impl PartialOrd for ValueClass<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ValueClass<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ValueClass<'_> {}

impl Hash for ValueClass<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u8(self.rank_kind());
        match self {
            Self::Integer(value) => state.write_i64(*value),
            Self::Float(value) => state.write_u64(value.to_bits()),
            Self::Other(value) => value.hash(state),
        }
    }
}

/// Whether every value of `tuple` is a primitive or a NULL: what a
/// partition's tuple holds, and what its values are ordered as.
pub(crate) fn holds_only_primitive_values(tuple: &Struct) -> bool {
    let mut values = tuple.fields().iter();
    values.all(|value| matches!(value, None | Some(Literal::Primitive(_))))
}

/// The primitive `value`, a value of a partition's tuple, holds; `None`
/// for a NULL.
fn read_primitive(value: &Option<Literal>) -> Option<&PrimitiveLiteral> {
    match value {
        None => None,
        Some(Literal::Primitive(value)) => Some(value),
        Some(nested) => panic!("a partition's tuple holds the value {nested:?}, not a primitive"),
    }
}

/// What `literal` holds outside its own value: the bytes of a string or a
/// binary.
fn estimate_literal_heap_bytes(literal: &PrimitiveLiteral) -> usize {
    match literal {
        PrimitiveLiteral::String(value) => measure_allocation(value.len()),
        PrimitiveLiteral::Binary(value) => measure_allocation(value.len()),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{BuildHasher, RandomState};

    use super::*;

    /// The partition of the one value `value`, under the spec 0.
    fn partition_of(value: Literal) -> Partition {
        Partition::new(0, Struct::from_iter([Some(value)]))
    }

    /// Whether `left` and `right` are equal, order as equal and hash alike,
    /// which all three say together.
    fn are_one_partition(left: &Partition, right: &Partition) -> bool {
        let hasher = RandomState::new();
        let is_equal = left == right;
        let hashes_alike = hasher.hash_one(left) == hasher.hash_one(right);
        assert_eq!(is_equal, left.cmp(right) == Ordering::Equal);
        assert!(!is_equal || hashes_alike);
        is_equal
    }

    #[test]
    fn a_value_is_the_same_partition_in_either_width() {
        let integers = (
            partition_of(Literal::int(7)),
            partition_of(Literal::long(7)),
        );
        let floats = (
            partition_of(Literal::float(2.5f32)),
            partition_of(Literal::double(2.5)),
        );
        let negative_zeros = (
            partition_of(Literal::float(-0.0f32)),
            partition_of(Literal::double(-0.0)),
        );

        let are_same = [integers, floats, negative_zeros]
            .map(|(narrow, wide)| are_one_partition(&narrow, &wide));

        assert_eq!(are_same, [true; 3]);
    }

    #[test]
    fn values_that_differ_are_different_partitions_in_any_width() {
        let integers = (
            partition_of(Literal::int(7)),
            partition_of(Literal::long(8)),
        );
        let tenths = (
            partition_of(Literal::float(0.1f32)),
            partition_of(Literal::double(0.1)),
        );
        let zeros = (
            partition_of(Literal::float(-0.0f32)),
            partition_of(Literal::double(0.0)),
        );
        let kinds = (
            partition_of(Literal::long(1)),
            partition_of(Literal::double(1.0)),
        );

        let are_same =
            [integers, tenths, zeros, kinds].map(|(left, right)| are_one_partition(&left, &right));

        assert_eq!(are_same, [false; 4]);
    }

    #[test]
    fn a_narrow_nan_is_the_wide_nan_of_its_sign_whatever_else_its_bits_say() {
        let with_payload = partition_of(Literal::float(f32::from_bits(0x7fc0_0001)));
        let quiet = partition_of(Literal::float(f32::NAN));
        let wide = partition_of(Literal::double(f64::NAN));
        let negative = partition_of(Literal::float(-f32::NAN));

        let are_same = [
            are_one_partition(&with_payload, &quiet),
            are_one_partition(&quiet, &wide),
            are_one_partition(&negative, &wide),
        ];

        assert_eq!(are_same, [true, true, false]);
        assert!(negative < partition_of(Literal::double(f64::NEG_INFINITY)));
    }

    #[test]
    fn values_order_by_value_across_widths_and_by_kind_across_kinds() {
        let ascending = [
            partition_of(Literal::bool(true)),
            partition_of(Literal::long(-3)),
            partition_of(Literal::int(2)),
            partition_of(Literal::long(i64::from(i32::MAX) + 1)),
            partition_of(Literal::double(-1.5)),
            partition_of(Literal::float(-0.0f32)),
            partition_of(Literal::double(0.0)),
            partition_of(Literal::string("a")),
        ];

        let is_ascending = ascending.is_sorted_by(|left, right| left < right);

        assert!(is_ascending);
    }
}
