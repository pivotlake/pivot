//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! transaction resolve derives from its snapshot, so a query's filter pushdown
//! prunes its own view without affecting anyone else.

use catalog::CommitQueue;
use dispatch::RecordBatchOperatorSpec;
use parquet_engine::{ParquetBackedTable, ParquetTableBinding, ScanPlan};
use planner::catalog::{Column, Error as CatalogError, Result as CatalogResult, TableRevision};
use planner::expression::Expression;

use super::insert_sink::build_insert_spec;
use super::{CatalogTable, PivotlakeDatastore};

/// A catalog table resolved from one transaction, with this query's
/// pushed-down filters. Because the snapshot's frozen copy of the table
/// travels with the binding, every compile-time method resolves its file set
/// with no transaction handle.
pub type TableBinding = ParquetTableBinding<BoundCatalogTable>;

/// What a [`TableBinding`] is bound to: the snapshot's copy of a table and
/// the queue of the transaction that resolved it.
#[derive(Clone)]
pub struct BoundCatalogTable {
    /// The snapshot's frozen copy of this table: its schema and every
    /// committed file at the version the transaction opened. Reads plan
    /// their scan straight from it; a write resolves the live table by its
    /// durable id ([`CatalogTable::id`]) at commit, so it survives a
    /// concurrent rename.
    table: CatalogTable,
    /// The transaction's staging queue, shared with the
    /// [`PivotlakeTransaction`](super::PivotlakeTransaction) that produced this
    /// binding, so the append an INSERT compiled by this binding stages as it
    /// runs is applied by that transaction's commit.
    commit_actions: CommitQueue<PivotlakeDatastore>,
}

impl BoundCatalogTable {
    pub(super) fn new(
        table: CatalogTable,
        commit_actions: CommitQueue<PivotlakeDatastore>,
    ) -> Self {
        Self {
            table,
            commit_actions,
        }
    }
}

impl std::fmt::Debug for BoundCatalogTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundCatalogTable")
            .field("id", &self.table.id())
            .finish_non_exhaustive()
    }
}

impl ParquetBackedTable for BoundCatalogTable {
    fn revision(&self) -> TableRevision {
        TableRevision {
            identity: self.table.id().to_string(),
            version: self.table.version().to_string(),
        }
    }

    fn columns(&self) -> Vec<Column> {
        self.table.columns()
    }

    fn list_never_null_columns(&self) -> Vec<bool> {
        self.table.list_never_null_columns()
    }

    fn plan_scan(&self, filters: &[Expression]) -> CatalogResult<ScanPlan> {
        self.table
            .load_scan_plan(filters)
            .map_err(|e| CatalogError::Other(Box::new(e)))
    }

    /// What the captured snapshot's metadata counts; a count it cannot give
    /// means "scan", which also reports what is wrong.
    fn count_rows(&self) -> Option<i64> {
        let rows = self.table.count_rows().ok()??;
        i64::try_from(rows).ok()
    }

    /// The rows of every live file, as the captured snapshot's metadata
    /// counts them.
    fn estimate_row_count(&self) -> Option<u64> {
        self.table.total_rows().ok()
    }

    /// Each row group resolves a pushed path against its own shredding layout,
    /// so the scan reads only the leaves the path needs.
    fn applies_variant_extracts(&self) -> bool {
        true
    }

    /// The captured snapshot copy stamps the durable schema onto every
    /// written file. Compiling stages nothing: the dataflow stages its
    /// append on the transaction's queue as it starts, and the transaction's
    /// commit resolves the live table by its durable id.
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
