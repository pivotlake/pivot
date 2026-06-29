//! [`Insert`]: `INSERT INTO <table> ...`.
//!
//! The child plan produces the rows to insert (a `VALUES` list is an
//! [`ExpressionGet`](super::ExpressionGet), a sub-`SELECT` an ordinary plan).
//! `Insert` reshapes those rows into the table's schema, reordering columns to
//! the table's storage order and casting each to its declared type (DuckDB's
//! coercing casts are unwrapped during translation, so we re-apply them here),
//! then hands them to [`Catalog::insert`], which compiles the write: encode to
//! Parquet (respecting the table's partitioning/sort) and commit each file. The
//! statement yields no rows and only returns once the data is durable.

use crate::catalog::{Catalog, Column};
use crate::compile::Error;
use crate::types::{Type, type_from_logical};
use arrow::compute::cast;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use duckdb_planner::operator as duckdb_operator;
use std::fmt;
use std::sync::Arc;

/// `INSERT INTO <table> ...` with the rows produced by its single child.
#[derive(Debug)]
pub struct Insert {
    pub table: String,
    /// The table's columns in storage order.
    pub columns: Vec<Column>,
    /// Per table column, the child-output column that supplies it (empty = the
    /// child already matches the table's column order; `-1` = the statement
    /// omitted that column).
    pub column_index_map: Vec<i64>,
}

impl TryFrom<duckdb_operator::Insert> for Insert {
    type Error = super::Error;

    fn try_from(insert: duckdb_operator::Insert) -> Result<Self, Self::Error> {
        let columns = insert
            .columns
            .into_iter()
            .map(|column| {
                Ok(Column {
                    name: column.name,
                    col_type: type_from_logical(column.col_type)?,
                })
            })
            .collect::<Result<Vec<_>, super::Error>>()?;
        Ok(Self {
            table: insert.table,
            columns,
            column_index_map: insert.column_index_map,
        })
    }
}

impl fmt::Display for Insert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cols = self
            .columns
            .iter()
            .map(|c| format!("{}:{}", c.name, c.col_type))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Insert({}, [{cols}])", self.table)
    }
}

impl Insert {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        dispatcher: &DataFlowDispatcher,
        catalog: &Arc<dyn Catalog>,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // For each table column, the child-output column that supplies it. An
        // empty map means the child already matches the table's column order.
        let sources: Vec<usize> = if self.column_index_map.is_empty() {
            (0..self.columns.len()).collect()
        } else {
            self.columns
                .iter()
                .enumerate()
                .map(|(table_col, column)| match self.column_index_map[table_col] {
                    i if i >= 0 => Ok(i as usize),
                    // Omitted column. The engine has no column defaults, so its
                    // value would be NULL, which the writer can't store.
                    _ => Err(Error::InsertMissingColumn(column.name.clone())),
                })
                .collect::<Result<_, _>>()?
        };

        let targets: Vec<DataType> = self
            .columns
            .iter()
            .map(|column| insert_arrow_type(&column.col_type))
            .collect::<Result<_, _>>()?;

        // The output schema is the table's: its column names, declared types, and
        // (nullability left open so a stray NULL surfaces as the writer's
        // clean "non-null column" error rather than a batch-build panic).
        let schema = Arc::new(Schema::new(
            self.columns
                .iter()
                .zip(&targets)
                .map(|(column, data_type)| Field::new(&column.name, data_type.clone(), true))
                .collect::<Vec<_>>(),
        ));

        // DuckDB type-checks the values against the column types at bind time, so
        // every cast here is one it already proved valid (hence `expect`). An
        // out-of-range value safe-casts to NULL, which the writer then rejects.
        let projected = input.project(move || {
            let sources = sources.clone();
            let targets = targets.clone();
            let schema = schema.clone();
            move |batch: RecordBatch| {
                let columns: Vec<ArrayRef> = sources
                    .iter()
                    .zip(&targets)
                    .map(|(&source, data_type)| {
                        cast(batch.column(source), data_type)
                            .expect("INSERT value casts to the column's declared type")
                    })
                    .collect();
                RecordBatch::try_new(schema.clone(), columns)
                    .expect("reordered INSERT batch matches the table schema")
            }
        });

        catalog
            .insert(self.table.clone(), projected, dispatcher)
            .map_err(Error::Insert)
    }
}

/// The arrow type an inserted column is cast to: the storage type the scan
/// produces for the table's declared [`Type`]. Limited to the types the Parquet
/// writer can encode; others are rejected with a clear error rather than failing
/// deep in the encoder.
fn insert_arrow_type(col_type: &Type) -> Result<DataType, Error> {
    Ok(match col_type {
        Type::Int32 => DataType::Int32,
        Type::Int64 => DataType::Int64,
        Type::Float64 => DataType::Float64,
        Type::Utf8 => DataType::Utf8View,
        other => return Err(Error::InsertUnsupportedColumnType(other.clone())),
    })
}
