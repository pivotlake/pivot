//! Rows produced by a SQL `VALUES` clause.
//!
//! The translation from the parsed cell expressions to an Arrow `RecordBatch`
//! happens at **plan build time** ([`super::super::build`]): a literal `VALUES`
//! is materialized straight into a [`Values::Literal`] batch. A *prepared*
//! `VALUES` (its cells are bound parameters, e.g. `INSERT INTO t VALUES ($1,$2)`)
//! can't be materialized until the values are known, so it is kept as a
//! [`Values::Prepared`] "pointer" batch: for each output column, the parameter
//! index feeding each row. It is resolved to a real batch at compile time by
//! gathering the bound `params`. No per-cell expression evaluation happens at
//! run time, and there is no special-cased INSERT path. Non-parameterized,
//! non-literal cells retain the normal per-execution expression path.

use std::fmt;
use std::sync::Arc;

use arrow::compute::{cast, concat};
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, Scalar};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec, values_input};

use crate::compile::{Error, ExprFn};
use crate::expression::Expression;
use crate::types::{Type, type_from_physical};

#[derive(Debug)]
pub enum Values {
    /// Rows fully known at plan time, materialized to one batch during build.
    Literal(RecordBatch),
    /// Non-parameterized expressions that must be evaluated for each execution.
    Dynamic(Vec<Vec<Expression>>),
    /// Rows referencing bound parameters; resolved to a batch at compile time
    /// from the bound `params`.
    Prepared(PreparedValues),
}

/// The "pointer batch" for a prepared `VALUES`: per output column, which
/// parameter feeds each of its rows. `VALUES ($1,$2),($3,$4)` yields column 0 =
/// params `[0, 2]` and column 1 = params `[1, 3]`.
#[derive(Debug)]
pub struct PreparedValues {
    columns: Vec<PreparedColumn>,
    row_count: usize,
}

#[derive(Debug)]
pub struct PreparedColumn {
    /// The parameter index feeding each row of this column, in row order.
    param_indexes: Vec<usize>,
    /// The declared type of each referenced parameter, parallel to
    /// `param_indexes`, for the statement's parameter description.
    param_types: Vec<Type>,
    /// The Arrow type this column must emit (the cell's result type). Equal to
    /// the parameter's type unless the cell cast the parameter to a column type.
    output_type: DataType,
}

impl PreparedValues {
    pub(crate) fn new(columns: Vec<PreparedColumn>, row_count: usize) -> Self {
        Self { columns, row_count }
    }

    /// Gather the bound `params` into the real batch this `VALUES` emits: each
    /// column's rows are the parameter values it points at, in order.
    fn bind(&self, params: &[Scalar<ArrayRef>]) -> Result<RecordBatch, Error> {
        let mut fields = Vec::with_capacity(self.columns.len());
        let mut columns = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let mut cells: Vec<ArrayRef> = column
                .param_indexes
                .iter()
                .map(|&index| {
                    params
                        .get(index)
                        .map(|scalar| scalar.clone().into_inner())
                        .ok_or(Error::UnboundParameter {
                            index,
                            len: params.len(),
                        })
                })
                .collect::<Result<_, _>>()?;
            let cells_share_type = cells.first().is_none_or(|first| {
                cells
                    .iter()
                    .all(|cell| cell.data_type() == first.data_type())
            });
            if !cells_share_type {
                cells = cells
                    .into_iter()
                    .map(|cell| {
                        cast(&cell, &column.output_type)
                            .map_err(|error| Error::ValuesBatch(error.to_string()))
                    })
                    .collect::<Result<_, _>>()?;
            }
            let refs: Vec<&dyn Array> = cells.iter().map(AsRef::as_ref).collect();
            let gathered = concat(&refs).map_err(|e| Error::ValuesBatch(e.to_string()))?;
            // Coerce to the cell's declared type only when the bound parameter's
            // type differs (e.g. an int4 parameter into a bigint column).
            let array = if gathered.data_type() == &column.output_type {
                gathered
            } else {
                cast(&gathered, &column.output_type)
                    .map_err(|e| Error::ValuesBatch(e.to_string()))?
            };
            // Column names carry no meaning: the insert matches VALUES columns to
            // the table positionally.
            fields.push(Field::new("", column.output_type.clone(), true));
            columns.push(array);
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .map_err(|e| Error::ValuesBatch(e.to_string()))
    }
}

impl Values {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        params: &[Scalar<ArrayRef>],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        match self {
            Values::Literal(batch) => {
                Ok(values_input(dispatcher, [batch.clone()]).record_batches())
            }
            Values::Dynamic(rows) => compile_dynamic(dispatcher, rows),
            Values::Prepared(prepared) => {
                Ok(values_input(dispatcher, [prepared.bind(params)?]).record_batches())
            }
        }
    }

    /// The pivot [`Type`] of each emitted column, in order.
    pub(crate) fn output_types(&self) -> Result<Vec<Type>, Error> {
        match self {
            Values::Literal(batch) => batch
                .schema_ref()
                .fields()
                .iter()
                .map(|field| field.data_type())
                .map(|data_type| {
                    type_from_physical(data_type)
                        .ok_or_else(|| Error::UnsupportedValuesType(data_type.clone()))
                })
                .collect(),
            Values::Dynamic(rows) => rows
                .first()
                .into_iter()
                .flatten()
                .map(Expression::result_type)
                .collect(),
            Values::Prepared(prepared) => prepared
                .columns
                .iter()
                .map(|column| {
                    type_from_physical(&column.output_type)
                        .ok_or_else(|| Error::UnsupportedValuesType(column.output_type.clone()))
                })
                .collect(),
        }
    }

    /// The number of rows this `VALUES` emits.
    fn row_count(&self) -> usize {
        match self {
            Values::Literal(batch) => batch.num_rows(),
            Values::Dynamic(rows) => rows.len(),
            Values::Prepared(prepared) => prepared.row_count,
        }
    }

    /// Collect the `(index, type)` of each parameter this `VALUES` references.
    /// A prepared `VALUES` parameter is always resolved from its target column.
    pub(crate) fn collect_params(&self, out: &mut Vec<(usize, Option<Type>)>) {
        if let Values::Prepared(prepared) = self {
            for column in &prepared.columns {
                for (&index, param_type) in column.param_indexes.iter().zip(&column.param_types) {
                    out.push((index, Some(param_type.clone())));
                }
            }
        }
    }
}

fn compile_dynamic(
    dispatcher: &DataFlowDispatcher,
    rows: &[Vec<Expression>],
) -> Result<RecordBatchOperatorSpec, Error> {
    let rows: Vec<Vec<Arc<ExprFn>>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|expression| expression.compile().map(Arc::new))
                .collect()
        })
        .collect::<Result<_, _>>()?;
    let empty_row = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .expect("one empty row is a valid batch");

    Ok(values_input(dispatcher, rows)
        .map_each(move |row: Vec<Arc<ExprFn>>| {
            let columns: Vec<ArrayRef> = row
                .iter()
                .map(|builder| builder()(&empty_row).into_array(1))
                .collect();
            let fields = columns
                .iter()
                .map(|column| Field::new("", column.data_type().clone(), true))
                .collect::<Vec<_>>();
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
                .expect("VALUES expressions in one row have equal length")
        })
        .record_batches())
}

impl PreparedColumn {
    pub(crate) fn new(
        param_indexes: Vec<usize>,
        param_types: Vec<Type>,
        output_type: DataType,
    ) -> Self {
        Self {
            param_indexes,
            param_types,
            output_type,
        }
    }
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Values(rows: {})", self.row_count())
    }
}
