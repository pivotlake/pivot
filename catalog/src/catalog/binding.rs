//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! [`Catalog::table`](planner::catalog::Catalog::table) resolve derives from
//! the master entry, so a query's filter pushdown prunes its own view without
//! affecting anyone else.

use std::sync::Arc;

use crate::parquet::{
    ParquetTable, ScanEqualityPredicate, materialize, row_group_eliminated, row_group_filter_from,
    scan_order_from, table_input_with_filter_and_eq_predicates,
};
use arrow_array::{Array, ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    Column, DynamicScanPredicate, Error as CatalogError, QueryContext, Result as CatalogResult,
    Table,
};
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

/// A catalog table, resolved as a **live handle**: it carries the table's name,
/// schema, and this query's pushed-down predicates, but *not* a file set. It
/// pulls the current row groups from the query's [`QueryContext`] every time it
/// compiles, so a reused (cached) plan always scans the latest committed files.
/// Cloned per-binding so each query accumulates its own predicates.
#[derive(Clone, Debug)]
pub struct TableBinding {
    /// The catalog name resolved, used to fetch this table's data from the
    /// query cache at compile time.
    name: String,
    pub columns: Vec<Column>,
    /// Single-column predicates pushed down for this binding (recorded here
    /// because the `Table` trait gives no channel from `pushdown_filter` to
    /// `compile`); applied as a filter when the scan is compiled.
    predicates: Vec<PushedPredicate>,
}

impl TableBinding {
    /// A binding over `name`, with no predicates pushed yet.
    pub(super) fn new(name: String, columns: Vec<Column>) -> Self {
        Self {
            name,
            columns,
            predicates: Vec::new(),
        }
    }

    /// This table's current committed row groups, from the query context (which
    /// reloads to the latest version and pins). The context is always our own
    /// [`ParquetQueryContext`](super::ParquetQueryContext) — a `ParquetCatalog`
    /// only ever compiles its own bindings — so the downcast is an invariant;
    /// failing it, or the table having been dropped since planning, is an error,
    /// never a silent empty scan.
    fn resolve_files(&self, ctx: &dyn QueryContext) -> CatalogResult<Arc<ParquetTable>> {
        let ctx = ctx
            .as_any()
            .downcast_ref::<super::ParquetQueryContext>()
            .ok_or_else(|| {
                CatalogError::Other("query context is not a ParquetQueryContext".into())
            })?;
        ctx.parquet(&self.name)
    }
}

impl Table for TableBinding {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
        ctx: &dyn QueryContext,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // The query context hands back this table's latest committed files (it
        // reloads once and pins), so a reused cached plan scans data committed
        // since it was planned.
        let current = self.resolve_files(ctx)?;

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

        // Prune the row groups by the pushed-down predicates' stats. No footer
        // re-read — the cache already holds the materialized row groups.
        let parquet = Arc::new(self.pruned_parquet(&current));
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
        ctx: &dyn QueryContext,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Same `ctx` as the scan, so this reads the *same* pinned snapshot —
        // their global row-group indices must line up. Late materialization
        // re-reads rows by that global index, so it uses the full table, not the
        // pruned scan view.
        Ok(materialize(input, self.resolve_files(ctx)?, projection))
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

    fn column_min_max(
        &self,
        column: usize,
        ctx: &dyn QueryContext,
    ) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // Read this binding's current committed files, the same snapshot a scan
        // would see; an unresolvable context (e.g. table dropped) means "scan".
        let parquet = self.resolve_files(ctx).ok()?;
        let row_groups = parquet.row_groups();
        if row_groups.is_empty() {
            return None;
        }
        // Every row group must carry both bounds; fold them with the arrow
        // comparison kernels (stats come back typed as the physical column).
        let mut min: Option<Scalar<ArrayRef>> = None;
        let mut max: Option<Scalar<ArrayRef>> = None;
        for rg in row_groups {
            let stats = rg.column_statistics(column)?;
            let (rg_min, rg_max) = (stats.min.as_ref()?, stats.max.as_ref()?);
            min = Some(match min {
                Some(m) if scalar_lt(&m, rg_min) => m,
                _ => rg_min.clone(),
            });
            max = Some(match max {
                Some(m) if scalar_lt(rg_max, &m) => m,
                _ => rg_max.clone(),
            });
        }
        Some((min?, max?))
    }

    fn row_count(&self, ctx: &dyn QueryContext) -> Option<i64> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // The parquet footer carries each row group's exact row count, so the
        // table's count is their sum, with no data pages read.
        let parquet = self.resolve_files(ctx).ok()?;
        Some(parquet.row_groups().iter().map(|rg| rg.num_rows).sum())
    }
}

/// `a < b` over two single-value scalars of the same physical type. A null,
/// type mismatch, or kernel error reads as `false`. In [`column_min_max`] every
/// comparison is between two same-typed integer/temporal stat bounds, where the
/// kernel never errors and the bounds are non-null. (Mirrors the stricter,
/// private `scalar_lt` in `parquet::reading::fetching`; worth consolidating.)
///
/// [`column_min_max`]: TableBinding::column_min_max
fn scalar_lt(a: &Scalar<ArrayRef>, b: &Scalar<ArrayRef>) -> bool {
    arrow_ord::cmp::lt(a, b).is_ok_and(|r| r.len() == 1 && r.is_valid(0) && r.value(0))
}

impl TableBinding {
    /// Clone `parquet`'s row groups and keep only those that survive this
    /// binding's pushed-down predicates — i.e. what [`Table::compile`] actually
    /// scans over the table's current files. A min/max stat that proves no row in
    /// a group can match drops it; a stats-comparison error means "can't prune"
    /// (kept) — never wrong, just unoptimized. No footer I/O. Exposed so pruning
    /// can be asserted directly.
    pub fn pruned_parquet(&self, parquet: &ParquetTable) -> ParquetTable {
        let mut parquet = parquet.clone();
        parquet.row_groups_mut().retain(|rg| {
            !self.predicates.iter().any(|p| {
                row_group_eliminated(rg.as_ref(), p.column_idx, p.compare_type, &p.value)
                    .unwrap_or(false)
            })
        });
        parquet
    }
}
