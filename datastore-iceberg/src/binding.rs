//! The per-query binding of one loaded table: the snapshot the query bound
//! plus the predicates pushed into it, compiled into the Parquet scan.

use std::sync::Arc;

use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use parquet_engine::{
    ParquetTable, PushedPredicate, equality_predicates, materialize, prune_parquet_row_groups,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Result as CatalogResult, TableReference,
    TableRevision,
};
use planner::expression::TableFilter;

use crate::table::LoadedTable;

/// A table bound by one query. The loaded table is shared with every other
/// binding of it in the same query; the predicates are this binding's own, so
/// each query prunes its own view.
#[derive(Clone)]
pub(crate) struct IcebergTableBinding {
    reference: TableReference,
    table: Arc<LoadedTable>,
    predicates: Vec<PushedPredicate>,
}

impl std::fmt::Debug for IcebergTableBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergTableBinding")
            .field("reference", &self.reference)
            .field("metadata_location", &self.table.metadata_location)
            .field("predicates", &self.predicates)
            .finish()
    }
}

impl IcebergTableBinding {
    pub(crate) fn new(reference: TableReference, table: Arc<LoadedTable>) -> Self {
        Self {
            reference,
            table,
            predicates: Vec::new(),
        }
    }

    /// The row groups this binding's predicates leave: the files the
    /// manifests' partition summaries, partition values and column bounds
    /// cannot rule out, fetched from their footers, then narrowed to the row
    /// groups the footer statistics cannot rule out. The very view a scan
    /// reads and a late materialize must rebuild, so it is a pure function of
    /// the pinned snapshot and the recorded predicates and comes out identical
    /// each time it is built.
    fn pruned(&self) -> CatalogResult<Arc<ParquetTable>> {
        let parquet = self.table.fetch_pruned_parquet(&self.predicates)?;
        Ok(Arc::new(prune_parquet_row_groups(
            &parquet,
            &self.predicates,
        )))
    }

    /// `answer`, a statistic read from the manifests, or `None` when they
    /// could not be read: the planner then plans without it, and the scan
    /// that follows reads the same manifests and reports what failed.
    fn take_statistic<T>(&self, answer: crate::Result<T>) -> Option<T> {
        match answer {
            Ok(value) => Some(value),
            Err(error) => {
                tracing::warn!(table = %self.table.name, %error, "planning without manifest statistics");
                None
            }
        }
    }
}

impl BoundTable for IcebergTableBinding {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.table.revision()
    }

    fn supports_late_materialization(&self) -> bool {
        true
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        let eq_predicates = equality_predicates(&self.predicates);
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &self.pruned()?,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(eq_predicates),
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.table.columns().to_vec()
    }

    fn nullability(&self) -> Vec<bool> {
        (0..self.table.columns().len())
            .map(|column| self.table.column_may_hold_nulls(column))
            .collect()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(materialize(input, self.pruned()?, projection))
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> CatalogResult<bool> {
        // Recorded for file and row-group pruning at compile time. The query's
        // own `Filter` stays above the scan, so this only ever skips work.
        self.predicates.extend(PushedPredicate::from_filter(filter));
        Ok(false)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed predicate means
        // the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        self.take_statistic(self.table.column_min_max(column))
            .flatten()
    }

    fn row_count(&self) -> Option<i64> {
        if !self.predicates.is_empty() {
            return None;
        }
        self.take_statistic(self.table.total_rows())
    }

    fn estimate_row_count(&self) -> Option<u64> {
        self.take_statistic(self.table.estimated_rows())
            .map(|rows| rows as u64)
    }
}
