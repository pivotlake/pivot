//! The `metadata('table')` table function: one row per row group of a table,
//! read from its parquet footers (no data pages).
//!
//! It lives here, not in the planner, because row groups are a parquet/catalog
//! concept the generic planner knows nothing about. The planner only exposes the
//! generic [`TableFunction`] trait; this implements it, and
//! `ParquetCatalog::table_function`
//! hands it back so the planner's `TableFunctionScan` can run it.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use planner::ScalarValue;
use planner::catalog::{Column, QueryContext};
use planner::compile::Error;
use planner::types::Type;
use planner::{TableFunction, TableFunctionSignature};

use super::ParquetQueryContext;
use crate::delta::PartitionEqFilter;
use crate::parquet::RowGroupMetadata;
use thiserror::Error as ThisError;

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

pub(super) struct MetadataTableFunction;

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
        ctx: &dyn QueryContext,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let table_name = match args {
            [ScalarValue::Utf8(name)] => name.as_str(),
            [_] => return Err(MetadataError::TableNameNotAString.into()),
            _ => return Err(MetadataError::WrongArgumentCount(args.len()).into()),
        };

        let parquet_ctx = ctx
            .as_any()
            .downcast_ref::<ParquetQueryContext>()
            .ok_or(MetadataError::NotAParquetCatalog)?;
        // Warm the table (reload to latest + load every file's footer); metadata
        // wants every file, so no partition filters are applied.
        parquet_ctx
            .parquet(table_name, std::iter::empty::<PartitionEqFilter>())
            .map_err(|source| MetadataError::ResolveTable {
                table: table_name.to_string(),
                source,
            })?;

        // Build straight from the table's per-file row groups, so each row
        // group's `file_name` is its real manifest path - no positional guessing.
        let table =
            parquet_ctx
                .table(table_name)
                .map_err(|source| MetadataError::ResolveTable {
                    table: table_name.to_string(),
                    source,
                })?;
        let rows = row_group_rows(&table.file_row_groups());
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

/// Why a `metadata()` call cannot compile. Wrapped in
/// [`Error::TableFunction`] on its way out to the generic planner.
#[derive(Debug, ThisError)]
enum MetadataError {
    #[error("expected a single table-name argument, got {0} arguments")]
    WrongArgumentCount(usize),
    #[error("the table name argument must be a string")]
    TableNameNotAString,
    #[error("requires a parquet-backed catalog")]
    NotAParquetCatalog,
    #[error("resolving table `{table}`: {source}")]
    ResolveTable {
        table: String,
        #[source]
        source: planner::catalog::Error,
    },
}

impl From<MetadataError> for Error {
    fn from(source: MetadataError) -> Self {
        Error::TableFunction {
            function: "metadata".to_string(),
            source: Box::new(source),
        }
    }
}
