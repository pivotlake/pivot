//! The per-query binding of one loaded table: the snapshot the query bound
//! plus the predicates pushed into it, compiled into the Parquet scan.

use std::sync::Arc;

use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use parquet_engine::{
    ParquetTable, PushedPredicate, equality_predicates, materialize, prune_parquet,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Result as CatalogResult, TableReference,
    TableRevision,
};
use planner::expression::Expression;

use crate::table::LoadedTable;

/// A table bound by one query. The loaded table is shared with every other
/// binding of it in the same query; the scan view and the predicates are this
/// binding's own, so each query prunes its own view.
#[derive(Clone)]
pub(crate) struct IcebergTableBinding {
    reference: TableReference,
    table: Arc<LoadedTable>,
    /// Every row group of the table, the view the predicates prune.
    parquet: Arc<ParquetTable>,
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
        let parquet = Arc::new(table.parquet_table());
        Self {
            reference,
            table,
            parquet,
            predicates: Vec::new(),
        }
    }

    /// The row groups that survive this binding's pushed predicates: what a
    /// scan reads, and the very view a late materialize must rebuild, since a
    /// row reference is a position in it.
    fn prune(&self) -> Arc<ParquetTable> {
        Arc::new(prune_parquet(&self.parquet, &self.predicates))
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
            &self.prune(),
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
        self.table.nullability.clone()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(materialize(input, self.prune(), projection))
    }

    fn pushdown_filter(&mut self, filter: Expression) -> CatalogResult<bool> {
        // Recorded for row-group pruning at compile time. The query's own
        // `Filter` stays above the scan, so this only ever skips work.
        self.predicates
            .extend(PushedPredicate::from_filter(&filter));
        Ok(false)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed predicate means
        // the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        self.parquet.column_min_max(column)
    }

    fn row_count(&self) -> Option<i64> {
        self.predicates
            .is_empty()
            .then(|| self.parquet.total_rows())
    }

    fn estimate_row_count(&self) -> Option<u64> {
        Some(self.parquet.total_rows() as u64)
    }
}
