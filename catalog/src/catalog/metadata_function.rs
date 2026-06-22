//! The `metadata('table')` table function: one row per row group of a table,
//! read from its parquet footers (no data pages).
//!
//! It lives here, not in the planner, because row groups are a parquet/catalog
//! concept the generic planner knows nothing about. The planner only exposes the
//! generic [`TableFunction`] trait; this implements it, and
//! [`ParquetCatalog::table_function`](super::ParquetCatalog::table_function)
//! hands it back so the planner's `TableFunctionScan` can run it.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use dispatch::io::FileLocation;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use planner::ScalarValue;
use planner::catalog::{Column, QueryContext};
use planner::compile::Error;
use planner::types::Type;
use planner::{TableFunction, TableFunctionSignature};

use super::ParquetQueryContext;
use crate::manifest::PartitionEqFilter;
use crate::parquet::types::table::ParquetTable;

/// The output columns, in declared order. The single source of truth for both
/// the binding signature (handed to DuckDB through the bridge) and the emitted
/// batch schema; every column is `BIGINT`.
const COLUMN_NAMES: [&str; 5] = [
    "file_index",
    "row_group_index",
    "num_rows",
    "num_columns",
    "compressed_bytes",
];

pub(super) struct MetadataTableFunction;

impl TableFunction for MetadataTableFunction {
    fn name(&self) -> &str {
        "metadata"
    }

    fn signature(&self) -> TableFunctionSignature {
        TableFunctionSignature {
            arguments: vec![Type::Utf8],
            columns: COLUMN_NAMES
                .iter()
                .map(|name| Column {
                    name: name.to_string(),
                    col_type: Type::Int64,
                })
                .collect(),
        }
    }

    fn compile(
        &self,
        args: &[ScalarValue],
        dispatcher: &DataFlowDispatcher,
        ctx: &dyn QueryContext,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let table_name = match args {
            [name] => name.raw_value.as_str(),
            _ => {
                return Err(invalid(format!(
                    "expected a single table name, got {} arguments",
                    args.len()
                )));
            }
        };

        // The context resolves any committed table by name; metadata wants every
        // file, so no partition filters are applied.
        let parquet = ctx
            .as_any()
            .downcast_ref::<ParquetQueryContext>()
            .ok_or_else(|| invalid("metadata() requires a parquet-backed catalog".to_string()))?
            .parquet(table_name, std::iter::empty::<PartitionEqFilter>())
            .map_err(|_| invalid(format!("table '{table_name}' does not exist")))?;

        let rows = row_group_rows(&parquet);
        let i64_column = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
        let columns: Vec<ArrayRef> = vec![
            i64_column(rows.iter().map(|r| r.file_index).collect()),
            i64_column(rows.iter().map(|r| r.row_group_index).collect()),
            i64_column(rows.iter().map(|r| r.num_rows).collect()),
            i64_column(rows.iter().map(|r| r.num_columns).collect()),
            i64_column(rows.iter().map(|r| r.compressed_bytes).collect()),
        ];
        let schema = Arc::new(Schema::new(
            COLUMN_NAMES
                .iter()
                .map(|name| Field::new(*name, DataType::Int64, false))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema, columns)
            .expect("metadata columns are equal-length single arrays");
        Ok(dispatch::values_input(dispatcher, [batch]).record_batches())
    }
}

/// One output row of `metadata()`, in the full declared column order.
struct RowGroupRow {
    file_index: i64,
    row_group_index: i64,
    num_rows: i64,
    num_columns: i64,
    compressed_bytes: i64,
}

/// One [`RowGroupRow`] per row group of `parquet`, all from the footers the
/// context already holds (no data pages read). Files are numbered in the order
/// they first appear: row groups of one file share a [`FileLocation`], so a map
/// from location to its assigned index groups them. `row_group_index` is the
/// position *within its file* (so `(file_index, row_group_index)` identifies a
/// row group), matching DuckDB's `parquet_metadata`.
fn row_group_rows(parquet: &ParquetTable) -> Vec<RowGroupRow> {
    let mut file_index_by_location: HashMap<FileLocation, i64> = HashMap::new();
    parquet
        .row_groups()
        .iter()
        .map(|row_group| {
            let next = file_index_by_location.len() as i64;
            let file_index = *file_index_by_location
                .entry(row_group.location.clone())
                .or_insert(next);
            let compressed_bytes = row_group
                .columns
                .iter()
                .map(|column| column.total_compressed_size)
                .sum();
            RowGroupRow {
                file_index,
                row_group_index: row_group.file_row_group_idx as i64,
                num_rows: row_group.num_rows,
                num_columns: row_group.columns.len() as i64,
                compressed_bytes,
            }
        })
        .collect()
}

fn invalid(message: String) -> Error {
    Error::InvalidTableFunctionArgument {
        function: "metadata".to_string(),
        message,
    }
}
