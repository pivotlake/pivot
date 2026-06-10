//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! [`Catalog::table`](planner::catalog::Catalog::table) resolve derives from
//! the master entry, so a query's filter pushdown prunes its own view without
//! affecting anyone else.

use std::sync::Arc;

use crate::parquet::{
    ParquetTable, ScanEqualityPredicate, materialize, row_group_eliminated, row_group_filter_from,
    scan_order_from, table_input_with_filter_and_eq_predicates,
};
use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{Column, DynamicScanPredicate, Result as CatalogResult, Table};
use planner::expression::{CompareType, Expression, TableFilter};

/// A single-column constant comparison (`col <cmp> const`) pushed down by
/// DuckDB during binding. Recorded as-is; applied at [`compile`](Table::compile)
/// time after the row-group metadata exists — min/max stats prune row groups,
/// and equality additionally prunes by dictionary contents in the decoder. The
/// upstream `Filter` always runs, so this is a pure optimization.
#[derive(Clone, Debug)]
struct PushedPredicate {
    column_idx: usize,
    compare_type: CompareType,
    value: Scalar<ArrayRef>,
}

/// A catalog table backed by a directory of Parquet files. Cloned per-binding
/// so each query accumulates its own pushed-down predicates via
/// [`Table::pushdown_filter`] without affecting others.
#[derive(Clone, Debug)]
pub struct TableBinding {
    pub columns: Vec<Column>,
    /// Where this table's Parquet data lives (a directory/key prefix in the
    /// database store). Kept so a future `refresh()` can re-read the latest files.
    pub location: String,
    /// The table's row-group metadata, read once when the table was defined and
    /// shared across every binding/query (cheap `Arc` clone). Pruning a binding's
    /// predicates clones the row-group `Vec` and filters it — no footer re-read.
    pub parquet: Arc<ParquetTable>,
    /// Single-column predicates pushed down for this binding (recorded here
    /// because the `Table` trait gives no channel from `pushdown_filter` to
    /// `compile`); applied as a filter when the scan is compiled.
    predicates: Vec<PushedPredicate>,
}

impl TableBinding {
    /// A binding over one state's scan view, with no predicates pushed yet.
    pub(super) fn new(columns: Vec<Column>, location: String, parquet: Arc<ParquetTable>) -> Self {
        Self {
            columns,
            location,
            parquet,
            predicates: Vec::new(),
        }
    }
}

impl Table for TableBinding {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Equality predicates additionally let the decoder skip row groups whose
        // dictionary for that column excludes the constant.
        let eq_predicates: Vec<ScanEqualityPredicate> = self
            .predicates
            .iter()
            .filter(|p| matches!(p.compare_type, CompareType::Equal))
            .map(|p| ScanEqualityPredicate {
                column_idx: p.column_idx,
                value: p.value.clone(),
            })
            .collect();

        // Prune the (already-materialized) row groups by the pushed-down
        // predicates' stats. No footer re-read — the row groups were built once
        // when the table was defined.
        let parquet = Arc::new(self.pruned_parquet());
        // Order the scan by the Top-N's key so its boundary tightens after the
        // first row group and the rest get pruned, instead of racing file order.
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(eq_predicates),
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn clone_box(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> RecordBatchOperatorSpec {
        // Late materialization re-reads rows by their *global* row-group index,
        // so it uses the full table, not the pruned scan view.
        materialize(input, self.parquet.clone(), projection)
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> CatalogResult<bool> {
        let TableFilter::Expression(expr) = filter else {
            return Ok(false);
        };
        let Expression::Compare(compare) = expr.as_ref() else {
            return Ok(false);
        };
        let (reference, constant) = match (compare.left.as_ref(), compare.right.as_ref()) {
            (Expression::Ref(r), Expression::Constant(k))
            | (Expression::Constant(k), Expression::Ref(r)) => (r, k),
            _ => return Ok(false),
        };

        // Just record it. The actual pruning (min/max row-group elimination and
        // equality/dictionary pruning) happens in `compile`, once the row-group
        // metadata exists. The upstream `Filter` is kept (we return `Ok(false)`),
        // so this is purely an optimization and never affects correctness.
        self.predicates.push(PushedPredicate {
            column_idx: reference.column_idx,
            compare_type: compare.compare_type,
            value: constant.clone(),
        });

        Ok(false)
    }
}

impl TableBinding {
    /// Clone this table's row groups and keep only those that survive this
    /// binding's pushed-down predicates — i.e. what [`Table::compile`] actually
    /// scans. A min/max stat that proves no row in a group can match drops it; a
    /// stats-comparison error means "can't prune" (kept) — never wrong, just
    /// unoptimized. No footer I/O: the row groups were materialized once when the
    /// table was defined. Exposed so pruning can be asserted directly.
    pub fn pruned_parquet(&self) -> ParquetTable {
        let mut parquet = (*self.parquet).clone();
        parquet.row_groups_mut().retain(|rg| {
            !self.predicates.iter().any(|p| {
                row_group_eliminated(rg.as_ref(), p.column_idx, p.compare_type, &p.value)
                    .unwrap_or(false)
            })
        });
        parquet
    }
}
