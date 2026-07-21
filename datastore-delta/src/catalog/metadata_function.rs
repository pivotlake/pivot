//! The `metadata('table')` table function: one row per row group of a table,
//! read from its parquet footers (no data pages).
//!
//! It lives here, not in the planner, because row groups are a parquet/catalog
//! concept the generic planner knows nothing about. The planner only exposes the
//! generic [`TableFunction`] trait; this implements it, and
//! `ParquetTransaction::table_function` hands it back so the planner's
//! `TableFunctionScan` can run it against the query's transaction.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use planner::ScalarValue;
use planner::catalog::{CatalogTransaction, Column};
use planner::compile::Error;
use planner::types::Type;
use planner::{TableFunction, TableFunctionSignature};

use crate::parquet::RowGroupMetadata;

/// The output columns, in declared order. The single source of truth for both
/// the binding signature (handed to DuckDB through the bridge) and the emitted
/// batch schema. All `BIGINT` except `file_name` (`VARCHAR`), which names the
/// file each row group belongs to so callers can filter by it.
const COLUMNS: [(&str, Type); 6] = [
    ("file_index", Type::Int64),
    ("row_group_index", Type::Int64),
    ("num_rows", Type::Int64),
    ("num_columns", Type::Int64),
    ("compressed_bytes", Type::Int64),
    ("file_name", Type::Utf8),
];

pub(super) struct MetadataTableFunction {
    /// The datastore this function was resolved from — indexes its snapshot in a
    /// multi-datastore `PivotTransaction` (in the `catalog` crate).
    catalog_name: String,
}

impl MetadataTableFunction {
    pub(super) fn new(catalog_name: String) -> Self {
        Self { catalog_name }
    }
}

impl TableFunction for MetadataTableFunction {
    fn name(&self) -> &str {
        "metadata"
    }

    fn signature(&self) -> TableFunctionSignature {
        TableFunctionSignature {
            arguments: vec![Type::Utf8],
            columns: COLUMNS
                .iter()
                .map(|(name, col_type)| Column {
                    name: name.to_string(),
                    col_type: col_type.clone(),
                })
                .collect(),
        }
    }

    fn compile(
        &self,
        args: &[ScalarValue],
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let table_name = match args {
            [ScalarValue::Utf8(name)] => name.as_str(),
            [_] => {
                return Err(invalid(
                    "metadata() expects a string table name".to_string(),
                ));
            }
            _ => {
                return Err(invalid(format!(
                    "expected a single table name, got {} arguments",
                    args.len()
                )));
            }
        };

        // Read the per-file row groups straight from the transaction's frozen
        // snapshot (every footer is already materialized there), so each row
        // group's `file_name` is its real manifest path - no positional guessing.
        // Routed to this function's datastore, exactly as a table binding is.
        let snapshot = super::snapshot_for(transaction, &self.catalog_name)
            .map_err(|_| invalid("metadata() requires a parquet-backed catalog".to_string()))?;
        let file_row_groups = snapshot
            .file_row_groups(table_name)
            .ok_or_else(|| invalid(format!("table '{table_name}' does not exist")))?;
        let rows = row_group_rows(&file_row_groups);
        let i64_column = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
        let columns: Vec<ArrayRef> = vec![
            i64_column(rows.iter().map(|r| r.file_index).collect()),
            i64_column(rows.iter().map(|r| r.row_group_index).collect()),
            i64_column(rows.iter().map(|r| r.num_rows).collect()),
            i64_column(rows.iter().map(|r| r.num_columns).collect()),
            i64_column(rows.iter().map(|r| r.compressed_bytes).collect()),
            Arc::new(StringViewArray::from(
                rows.iter()
                    .map(|r| r.file_name.as_str())
                    .collect::<Vec<_>>(),
            )),
        ];
        let schema = Arc::new(Schema::new(
            COLUMNS
                .iter()
                .map(|(name, col_type)| {
                    let data_type = match col_type {
                        Type::Utf8 => DataType::Utf8View,
                        _ => DataType::Int64,
                    };
                    Field::new(*name, data_type, false)
                })
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
    file_name: String,
}

/// One [`RowGroupRow`] per row group, built straight from each file's manifest
/// path and its row groups (in manifest order). `file_index` is the file's
/// position; `row_group_index` is the position *within its file* (so
/// `(file_index, row_group_index)` identifies a row group), matching DuckDB's
/// `parquet_metadata`. `file_name` is the file's real manifest path.
fn row_group_rows(files: &[(String, Vec<Arc<RowGroupMetadata>>)]) -> Vec<RowGroupRow> {
    files
        .iter()
        .enumerate()
        .flat_map(|(file_index, (file_name, row_groups))| {
            row_groups.iter().map(move |row_group| RowGroupRow {
                file_index: file_index as i64,
                row_group_index: row_group.file_row_group_idx as i64,
                num_rows: row_group.num_rows,
                num_columns: row_group.columns.len() as i64,
                compressed_bytes: row_group
                    .columns
                    .iter()
                    .map(|column| column.total_compressed_size)
                    .sum(),
                file_name: file_name.clone(),
            })
        })
        .collect()
}

fn invalid(message: String) -> Error {
    Error::InvalidTableFunctionArgument {
        function: "metadata".to_string(),
        message,
    }
}
