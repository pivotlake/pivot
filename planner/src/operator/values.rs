//! Rows produced by a SQL `VALUES` clause.
//!
//! Literal rows are materialized once while the statement is planned. Prepared
//! rows retain a column-oriented batch of pointers into the Bind values, so an
//! Execute gathers each finished Arrow column directly instead of evaluating
//! one expression and constructing one one-row batch per cell.

use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, Scalar};
use arrow_schema::{Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec, values_input};

use crate::compile::{BoundParameters, Error};
use crate::expression::Expression;
use crate::types::{Type, physical_arrow_type};

/// One cell in a prepared VALUES batch.
#[derive(Debug, Clone)]
pub enum ValuePointer {
    /// Read this cell from the zero-based Bind parameter index.
    Parameter(usize),
    /// A literal mixed into an otherwise prepared VALUES list.
    Constant(Scalar<ArrayRef>),
}

#[derive(Debug)]
pub enum ValuesData {
    /// A fully materialized literal batch, cached as part of the plan.
    Literal {
        batch: RecordBatch,
        types: Vec<Type>,
    },
    /// Column-major pointers, one vector per output column and one entry per row.
    Pointers {
        columns: Vec<Vec<ValuePointer>>,
        types: Vec<Type>,
    },
    /// General VALUES expressions that are not plain literals/pointers.
    Expressions {
        rows: Vec<Vec<Expression>>,
        types: Vec<Type>,
    },
}

#[derive(Debug)]
pub struct Values {
    pub data: ValuesData,
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = match &self.data {
            ValuesData::Literal { batch, .. } => batch.num_rows(),
            ValuesData::Pointers { columns, .. } => columns.first().map_or(0, Vec::len),
            ValuesData::Expressions { rows, .. } => rows.len(),
        };
        write!(f, "Values(rows: {rows})")
    }
}

impl Values {
    pub fn output_types(&self) -> Vec<Type> {
        match &self.data {
            ValuesData::Literal { types, .. } => types.clone(),
            ValuesData::Pointers { types, .. } => types.clone(),
            ValuesData::Expressions { types, .. } => types.clone(),
        }
    }

    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        parameters: &BoundParameters,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let batch = match &self.data {
            // RecordBatch cloning only clones Arcs; the cached literal buffers
            // flow into every execution unchanged and zero-copy.
            ValuesData::Literal { batch, .. } => batch.clone(),
            ValuesData::Pointers { columns, types } => {
                build_pointer_batch(columns, types, parameters)?
            }
            ValuesData::Expressions { rows, .. } => {
                let rows = rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|expression| expression.compile(parameters))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(values_input(dispatcher, rows)
                    .map_each(|row| {
                        let empty = RecordBatch::try_new_with_options(
                            Arc::new(Schema::empty()),
                            Vec::new(),
                            &RecordBatchOptions::new().with_row_count(Some(1)),
                        )
                        .expect("one empty input row is valid");
                        let columns = row
                            .into_iter()
                            .map(|builder| builder()(&empty).into_array(1))
                            .collect::<Vec<_>>();
                        let fields = columns
                            .iter()
                            .map(|column| Field::new("", column.data_type().clone(), true))
                            .collect::<Vec<_>>();
                        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
                            .expect("VALUES expressions produce one row")
                    })
                    .record_batches());
            }
        };
        Ok(values_input(dispatcher, [batch]).record_batches())
    }
}

fn build_pointer_batch(
    pointers: &[Vec<ValuePointer>],
    types: &[Type],
    parameters: &BoundParameters,
) -> Result<RecordBatch, Error> {
    let mut columns = Vec::with_capacity(pointers.len());
    for (column, output_type) in pointers.iter().zip(types) {
        let output_type = physical_arrow_type(output_type);
        let mut values: Vec<ArrayRef> = Vec::with_capacity(column.len());
        for pointer in column {
            let array = match pointer {
                ValuePointer::Parameter(index) => parameters
                    .get(*index)
                    .ok_or(Error::MissingParameter(index + 1))?
                    .clone()
                    .into_inner(),
                ValuePointer::Constant(value) => value.clone().into_inner(),
            };
            values.push(array);
        }

        // The common prepared INSERT case has one source type throughout a
        // column: concatenate once, then cast the whole column if DuckDB added
        // a target cast. Mixed client types are uncommon; cast those length-one
        // inputs first so concat still receives homogeneous arrays.
        let all_same_type = values
            .first()
            .is_none_or(|first| values.iter().all(|v| v.data_type() == first.data_type()));
        let column = if all_same_type {
            let refs = values
                .iter()
                .map(|v| v.as_ref())
                .collect::<Vec<&dyn Array>>();
            let gathered = arrow::compute::concat(&refs)?;
            if gathered.data_type() == &output_type {
                gathered
            } else {
                arrow::compute::cast(&gathered, &output_type)?
            }
        } else {
            let casted = values
                .iter()
                .map(|value| arrow::compute::cast(value, &output_type))
                .collect::<Result<Vec<_>, _>>()?;
            let refs = casted
                .iter()
                .map(|v| v.as_ref())
                .collect::<Vec<&dyn Array>>();
            arrow::compute::concat(&refs)?
        };
        columns.push(column);
    }

    let fields = columns
        .iter()
        .map(|column| Field::new("", column.data_type().clone(), true))
        .collect::<Vec<_>>();
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float64Array;
    use arrow_schema::DataType;

    #[test]
    fn literal_output_types_are_not_guessed_from_physical_types() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("", DataType::Float64, true)])),
            vec![Arc::new(Float64Array::from(vec![1.25])) as ArrayRef],
        )
        .unwrap();
        let values = Values {
            data: ValuesData::Literal {
                batch,
                types: vec![Type::Decimal],
            },
        };

        assert_eq!(values.output_types(), vec![Type::Decimal]);
    }
}
