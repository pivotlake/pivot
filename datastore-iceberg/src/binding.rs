//! The per-query binding of one loaded table: the snapshot the query bound
//! plus the filters pushed into it, compiled into the Parquet scan; or, for
//! an INSERT, the write into that table, committed on whatever the catalog
//! holds when the transaction commits.

use std::sync::Arc;

use catalog::CommitQueue;
use dispatch::RecordBatchOperatorSpec;
use iceberg_tables::Table as LoadedTable;
use parquet_engine::{ParquetBackedTable, ParquetTableBinding, ScanPlan};
use planner::catalog::{Column, Result as CatalogResult, TableRevision};
use planner::expression::Expression;

use crate::IcebergDatastore;
use crate::write::build_insert_spec;

/// One reference to a table in a query, with the filters pushed into it.
pub(crate) type IcebergTableBinding = ParquetTableBinding<BoundIcebergTable>;

/// What an [`IcebergTableBinding`] is bound to: the table as the query's
/// snapshot of the catalog loaded it, and the queue of the transaction that
/// resolved it.
#[derive(Clone)]
pub(crate) struct BoundIcebergTable {
    table: Arc<LoadedTable>,
    /// The transaction's staging queue the INSERT's commit action lands on.
    commit_actions: CommitQueue<IcebergDatastore>,
}

impl BoundIcebergTable {
    pub(crate) fn new(
        table: Arc<LoadedTable>,
        commit_actions: CommitQueue<IcebergDatastore>,
    ) -> Self {
        Self {
            table,
            commit_actions,
        }
    }
}

impl std::fmt::Debug for BoundIcebergTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundIcebergTable")
            .field("metadata_location", &self.table.metadata_location())
            .finish_non_exhaustive()
    }
}

impl ParquetBackedTable for BoundIcebergTable {
    fn revision(&self) -> TableRevision {
        self.table.revision()
    }

    fn columns(&self) -> Vec<Column> {
        self.table.columns().to_vec()
    }

    fn list_never_null_columns(&self) -> Vec<bool> {
        self.table.list_never_null_columns()
    }

    fn plan_scan(&self, filters: &[Expression]) -> CatalogResult<ScanPlan> {
        Ok(self.table.plan_scan(filters).map_err(crate::Error::from)?)
    }

    /// The data files' rows less those their deletion vectors delete, read
    /// from the manifests.
    fn count_rows(&self) -> Option<i64> {
        self.table.row_count()
    }

    /// The manifest list's count, less what its delete manifests count.
    fn estimate_row_count(&self) -> Option<u64> {
        self.table.estimated_row_count().map(|rows| rows as u64)
    }

    /// A pushed variant extract stays an expression above the scan.
    fn applies_variant_extracts(&self) -> bool {
        false
    }

    /// The write takes the table's shape and store from the snapshot this
    /// query bound. The files land on a queue of this statement's own as the
    /// dataflow runs; the append staged here hands that queue to the
    /// transaction's commit.
    fn compile_insert(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(build_insert_spec(
            &self.table,
            self.commit_actions.clone(),
            input,
        )?)
    }
}
