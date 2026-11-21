//! What a per-query **binding** is bound to: the independently-mutable
//! [`ParquetTableBinding`](parquet_engine::ParquetTableBinding) every
//! transaction resolve derives from its snapshot, so a query's filter pushdown
//! prunes its own view without affecting anyone else, is over a
//! [`BoundCatalogTable`].

use catalog::CommitQueue;
use dispatch::RecordBatchOperatorSpec;
use parquet_engine::{ParquetBackedTable, ScanSelection};
use planner::catalog::{Column, Error as CatalogError, Result as CatalogResult, TableRevision};
use planner::expression::Expression;

use super::insert_sink::build_insert_spec;
use super::{CatalogTable, PivotlakeDatastore};

/// What the binding of a catalog table resolved from one transaction is
/// bound to: the snapshot's copy of the table and the queue of the
/// transaction that resolved it. Because the snapshot's frozen copy of the
/// table travels with the binding, every compile-time method resolves its
/// file set with no transaction handle.
pub struct BoundCatalogTable {
    /// The snapshot's frozen copy of this table: its schema and every
    /// committed file at the version the transaction opened. Reads
    /// select their scan straight from it; a write resolves the live table by its
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

    fn declared_nullability(&self) -> Vec<bool> {
        self.table.declared_nullability()
    }

    fn select_scan(&self, filters: &[Expression]) -> CatalogResult<ScanSelection> {
        self.table
            .load_scan_selection(filters)
            .map_err(|e| CatalogError::Other(Box::new(e)))
    }

    /// What the captured snapshot's metadata counts; a count it cannot give
    /// means "scan", which also reports what is wrong.
    fn count_rows(&self) -> Option<i64> {
        let rows = self.table.count_live_rows()?;
        i64::try_from(rows).ok()
    }

    /// The rows of every live file, as the captured snapshot's metadata
    /// counts them.
    fn estimate_row_count(&self) -> Option<u64> {
        self.table.count_stored_rows().ok()
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
