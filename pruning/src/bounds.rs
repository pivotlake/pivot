use arrow_arith::boolean::{and_kleene, or_kleene};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum, Scalar};
use arrow_buffer::NullBuffer;
use arrow_ord::cmp;
use arrow_schema::{ArrowError, DataType};
use arrow_select::filter::FilterPredicate;

use crate::Comparison;

/// Bounds for one column across a batch of objects. Null array entries mean
/// unknown statistics, not SQL NULL values. Each present array has the batch's
/// length. Floating-point bounds alone do not describe the presence of NaNs.
#[derive(Clone, Default)]
pub struct ColumnBounds {
    /// Whether these physical statistics cover the logical expression in each
    /// object. Absent means all are usable; false entries supply no proof.
    /// For example, a typed VARIANT leaf cannot describe values stored in an
    /// unshredded fallback. This masks all proofs, including `all_null`.
    pub validity: Option<NullBuffer>,
    pub lower: Option<ArrayRef>,
    pub upper: Option<ArrayRef>,
    /// True proves that every value is SQL NULL (or the object is empty).
    pub all_null: Option<BooleanArray>,
    /// True proves that the object contains no NaNs. Absent/null entries leave
    /// NaN presence unknown; bounds can then exclude only comparisons that no
    /// NaN can satisfy, such as equality with a non-NaN constant.
    pub nan_free: Option<BooleanArray>,
}

impl ColumnBounds {
    pub(crate) fn filter(&self, filter: &FilterPredicate) -> Result<Self, ArrowError> {
        let boolean =
            |array: &BooleanArray| filter.filter(array).map(|array| array.as_boolean().clone());
        Ok(Self {
            validity: self
                .validity
                .as_ref()
                .map(|validity| {
                    boolean(&BooleanArray::new(validity.inner().clone(), None))
                        .map(|array| NullBuffer::new(array.values().clone()))
                })
                .transpose()?,
            lower: self
                .lower
                .as_ref()
                .map(|array| filter.filter(array.as_ref()))
                .transpose()?,
            upper: self
                .upper
                .as_ref()
                .map(|array| filter.filter(array.as_ref()))
                .transpose()?,
            all_null: self.all_null.as_ref().map(boolean).transpose()?,
            nan_free: self.nan_free.as_ref().map(boolean).transpose()?,
        })
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        Self {
            validity: self
                .validity
                .as_ref()
                .map(|validity| validity.slice(offset, len)),
            lower: self.lower.as_ref().map(|array| array.slice(offset, len)),
            upper: self.upper.as_ref().map(|array| array.slice(offset, len)),
            all_null: self.all_null.as_ref().map(|array| array.slice(offset, len)),
            nan_free: self.nan_free.as_ref().map(|array| array.slice(offset, len)),
        }
    }

    /// Evaluate one comparison for every object. One-sided bounds can prove
    /// exclusions too; unknown and incompatible bounds contribute no proof.
    pub fn may_match(
        &self,
        len: usize,
        compare: Comparison,
        constant: &Scalar<ArrayRef>,
    ) -> Result<BooleanArray, ArrowError> {
        self.validate(len)?;
        self.compare(len, compare, constant)
    }

    pub(crate) fn validate(&self, len: usize) -> Result<(), ArrowError> {
        if self
            .validity
            .as_ref()
            .is_some_and(|validity| validity.len() != len)
        {
            return Err(ArrowError::InvalidArgumentError(
                "statistics validity has a different length".into(),
            ));
        }
        for array in [
            self.lower.as_deref(),
            self.upper.as_deref(),
            self.all_null.as_ref().map(|array| array as &dyn Array),
            self.nan_free.as_ref().map(|array| array as &dyn Array),
        ]
        .into_iter()
        .flatten()
        {
            if array.len() != len {
                return Err(ArrowError::InvalidArgumentError(
                    "statistics arrays have different lengths".into(),
                ));
            }
        }
        if let (Some(lower), Some(upper)) = (&self.lower, &self.upper)
            && lower.data_type() != upper.data_type()
        {
            return Err(ArrowError::InvalidArgumentError(
                "lower and upper statistics have different types".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn compare(
        &self,
        len: usize,
        compare: Comparison,
        constant: &Scalar<ArrayRef>,
    ) -> Result<BooleanArray, ArrowError> {
        let (value, _) = constant.get();
        if value.is_null(0) {
            return Ok(BooleanArray::from(vec![false; len]));
        }
        let test = |bound: &Option<ArrayRef>,
                    kernel: fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>|
         -> Result<BooleanArray, ArrowError> {
            match bound {
                Some(bound) if bound.data_type() == value.data_type() => {
                    let result = kernel(bound, constant)?;
                    // A NaN bound supplies no ordering information, even
                    // when separate counts prove the data has no NaNs.
                    Ok(match bound.data_type() {
                        DataType::Float32 => result
                            .iter()
                            .zip(bound.as_primitive::<Float32Type>().iter())
                            .map(|(excluded, value)| {
                                value.filter(|value| !value.is_nan()).and(excluded)
                            })
                            .collect(),
                        DataType::Float64 => result
                            .iter()
                            .zip(bound.as_primitive::<Float64Type>().iter())
                            .map(|(excluded, value)| {
                                value.filter(|value| !value.is_nan()).and(excluded)
                            })
                            .collect(),
                        _ => result,
                    })
                }
                _ => Ok(BooleanArray::new_null(len)),
            }
        };
        let excluded = match compare {
            Comparison::Equal => {
                or_kleene(&test(&self.lower, cmp::gt)?, &test(&self.upper, cmp::lt)?)?
            }
            Comparison::NotEqual => {
                and_kleene(&test(&self.lower, cmp::eq)?, &test(&self.upper, cmp::eq)?)?
            }
            Comparison::Less => test(&self.lower, cmp::gt_eq)?,
            Comparison::LessEqual => test(&self.lower, cmp::gt)?,
            Comparison::Greater => test(&self.upper, cmp::lt_eq)?,
            Comparison::GreaterEqual => test(&self.upper, cmp::lt)?,
        };
        let constant_is_nan = match value.data_type() {
            DataType::Float32 => Some(value.as_primitive::<Float32Type>().value(0).is_nan()),
            DataType::Float64 => Some(value.as_primitive::<Float64Type>().value(0).is_nan()),
            _ => None,
        };
        // Arrow's total ordering puts negative NaNs below finite values and
        // positive NaNs above them. Either sign can satisfy a range comparison;
        // neither can equal a non-NaN constant.
        let needs_nan_proof =
            constant_is_nan.is_some_and(|is_nan| is_nan || !matches!(compare, Comparison::Equal));
        let proven = |mask: &BooleanArray, index| mask.is_valid(index) && mask.value(index);
        Ok((0..len)
            .map(|index| {
                if self
                    .validity
                    .as_ref()
                    .is_some_and(|validity| !validity.is_valid(index))
                {
                    return true;
                }
                let all_null = self
                    .all_null
                    .as_ref()
                    .is_some_and(|mask| proven(mask, index));
                let usable_bounds = !needs_nan_proof
                    || self
                        .nan_free
                        .as_ref()
                        .is_some_and(|mask| proven(mask, index));
                !all_null && !(usable_bounds && proven(&excluded, index))
            })
            .collect())
    }
}
