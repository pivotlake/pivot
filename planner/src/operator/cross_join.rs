//! [`CrossJoin`], the Cartesian product of two inputs.

use std::fmt;

use arrow_schema::Field;
use dispatch::RecordBatchOperatorSpec;

use crate::compile::Error;
use crate::types::{Type, physical_arrow_type};

/// Emits one row for every pair of left and right input rows.
#[derive(Debug)]
pub struct CrossJoin;

impl fmt::Display for CrossJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CrossJoin")
    }
}

impl CrossJoin {
    pub(crate) fn compile(
        &self,
        left: RecordBatchOperatorSpec,
        right: RecordBatchOperatorSpec,
        left_types: &[Type],
        left_nullability: &[bool],
        right_types: &[Type],
        right_nullability: &[bool],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let fields = |prefix: &str, types: &[Type], nullability: &[bool]| {
            types
                .iter()
                .zip(nullability)
                .enumerate()
                .map(|(index, (column_type, nullable))| {
                    Field::new(
                        format!("{prefix}_{index}"),
                        physical_arrow_type(column_type),
                        *nullable,
                    )
                })
                .collect()
        };
        Ok(left.cross_join(
            right,
            fields("left", left_types, left_nullability),
            fields("right", right_types, right_nullability),
        ))
    }
}
