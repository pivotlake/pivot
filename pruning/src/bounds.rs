use arrow_array::{Array, ArrayRef, BooleanArray, Datum, Scalar};
use arrow_buffer::BooleanBuffer;
use arrow_ord::cmp;
use arrow_schema::ArrowError;

use crate::Comparison;

/// What metadata knows about some values in each object, one slot per object.
///
/// The order is Arrow's total order, which places NaNs outside the finite
/// range. A source whose bounds skip NaNs must leave null the sides it cannot
/// vouch for.
#[derive(Clone, Debug)]
pub struct Bounds {
    /// Every non-null value is at least this. Null where unknown.
    pub lower: ArrayRef,
    /// Every non-null value is at most this. Null where unknown.
    pub upper: ArrayRef,
    /// True where every value is NULL or the object is empty. False or null
    /// where that is not proven.
    pub all_null: BooleanArray,
}

impl Bounds {
    pub(crate) fn len(&self) -> usize {
        self.all_null.len()
    }

    /// The objects that may hold a value satisfying `<comparison> constant`.
    /// Bounds answer only a constant of their own type: a constant of another
    /// type reads the values differently (another cast of a VARIANT field), so
    /// neither the bounds nor `all_null` say anything about it.
    pub(crate) fn may_match(
        &self,
        comparison: Comparison,
        constant: &Scalar<ArrayRef>,
    ) -> BooleanBuffer {
        let data_type = self.lower.data_type();
        if constant.get().0.data_type() != data_type || data_type.is_nested() {
            return BooleanBuffer::new_set(self.len());
        }
        let proves = |bound: &ArrayRef, kernel: Kernel| {
            let holds = kernel(bound, constant).expect("arrays of one scalar type are comparable");
            proven(&holds)
        };
        let excluded = match comparison {
            Comparison::Equal => &proves(&self.lower, cmp::gt) | &proves(&self.upper, cmp::lt),
            // Every value equals the constant only when both bounds do.
            Comparison::NotEqual => &proves(&self.lower, cmp::eq) & &proves(&self.upper, cmp::eq),
            Comparison::Less => proves(&self.lower, cmp::gt_eq),
            Comparison::LessEqual => proves(&self.lower, cmp::gt),
            Comparison::Greater => proves(&self.upper, cmp::lt_eq),
            Comparison::GreaterEqual => proves(&self.upper, cmp::lt),
        };
        !&(&excluded | &proven(&self.all_null))
    }
}

type Kernel = fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>;

/// The slots that are true, counting null as not proven.
fn proven(mask: &BooleanArray) -> BooleanBuffer {
    match mask.nulls() {
        Some(nulls) => mask.values() & nulls.inner(),
        None => mask.values().clone(),
    }
}
