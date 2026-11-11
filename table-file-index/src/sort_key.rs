//! A bound of a table's sort key in a file, as bytes that order as the
//! bounds do, so the stretches of the key two files cover are compared
//! without a typed value being decoded.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, new_null_array};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{ArrowError, DataType};

/// The stretch of a table's sort key a file's rows are bounded by, as bytes
/// that order as the keys do.
pub struct SortRange {
    /// The lower bound; a column without one orders first. Empty when the
    /// key has no ordered byte form.
    pub minimum: Arc<[u8]>,
    /// The upper bound; `None` when the key has no ordered byte form, or
    /// the column it leads with is not bounded by a value on both sides,
    /// so the stretch the file covers is unknown.
    pub maximum: Option<Arc<[u8]>>,
}

/// Encodes a bound of a table's sort key in a file as bytes that order as
/// the bounds do.
pub struct SortKeyEncoder {
    /// The Arrow type each sort column's bounds are encoded in, with the
    /// converter over those types; `None` for a table without a sort key,
    /// or one sorted by a column whose type has no ordered byte form.
    converter: Option<(Vec<DataType>, RowConverter)>,
}

impl SortKeyEncoder {
    /// The encoder for a table whose sort columns' bounds are of `types`,
    /// in the order the table is sorted by them.
    pub fn new(types: Vec<DataType>) -> Result<Self, ArrowError> {
        let fields: Vec<SortField> = types.iter().cloned().map(SortField::new).collect();
        let converter = if fields.is_empty() || !RowConverter::supports_fields(&fields) {
            None
        } else {
            Some((types, RowConverter::new(fields)?))
        };
        Ok(Self { converter })
    }

    /// Whether the sort key has an ordered byte form: a table without a
    /// sort key, or sorted by a column whose type has none, does not.
    pub fn has_ordered_form(&self) -> bool {
        self.converter.is_some()
    }

    /// One side of a file's bounds on the sort key as one row whose bytes
    /// order as the bounds do. `bound` hands out the file's bound on the
    /// sort column at a position, a single value; a column it has no bound
    /// for orders first. Empty when the key has no ordered byte form.
    pub fn encode_bound(
        &self,
        mut bound: impl FnMut(usize) -> Option<ArrayRef>,
    ) -> Result<Arc<[u8]>, ArrowError> {
        let Some((types, converter)) = &self.converter else {
            return Ok(Arc::from([]));
        };
        let arrays: Vec<ArrayRef> = types
            .iter()
            .enumerate()
            .map(|(position, data_type)| match bound(position) {
                Some(value) if value.len() == 1 && value.data_type() == data_type => Ok(value),
                Some(value) if value.len() == 1 => arrow_cast::cast(value.as_ref(), data_type),
                _ => Ok(new_null_array(data_type, 1)),
            })
            .collect::<Result<_, ArrowError>>()?;
        let rows = converter.convert_columns(&arrays)?;
        Ok(Arc::from(rows.row(0).as_ref()))
    }

    /// The stretch of the sort key `lower` and `upper` bound a file's rows
    /// by, each handing out the bound on the sort column at a position.
    pub fn encode_range(
        &self,
        mut lower: impl FnMut(usize) -> Option<ArrayRef>,
        mut upper: impl FnMut(usize) -> Option<ArrayRef>,
    ) -> Result<SortRange, ArrowError> {
        let is_known = |bound: Option<ArrayRef>| {
            bound.is_some_and(|bound| bound.len() == 1 && !bound.is_null(0))
        };
        let is_leading_bounded =
            self.has_ordered_form() && is_known(lower(0)) && is_known(upper(0));
        let minimum = self.encode_bound(&mut lower)?;
        let maximum = if is_leading_bounded {
            Some(self.encode_bound(&mut upper)?)
        } else {
            None
        };
        Ok(SortRange { minimum, maximum })
    }
}
