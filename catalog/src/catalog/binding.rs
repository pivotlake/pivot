//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! transaction resolve derives from its snapshot, so a query's filter pushdown
//! prunes its own view without affecting anyone else.

use std::sync::Arc;

use crate::manifest::PartitionEqFilter;
use crate::parquet::types::leaves::{first_leaf, variant_shredded_leaves};
use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::{
    ParquetTable, ScanEqualityPredicate, materialize, row_group_eliminated, row_group_filter_from,
    scan_order_from, table_input_with_filter_and_eq_predicates,
};
use arrow_array::{Array, ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    CatalogTransaction, Column, DynamicScanPredicate, Error as CatalogError,
    Result as CatalogResult, Table,
};
use planner::expression::{CompareType, Expression, Function, JsonPath, TableFilter};

use super::ParquetTransaction;
use super::insert_sink::build_insert_spec;

/// A single-column constant comparison (`col <cmp> const`) pushed down by
/// DuckDB during binding. Recorded as-is; applied at [`compile_scan`](Table::compile_scan)
/// time after the row-group metadata exists — min/max stats prune row groups,
/// and equality additionally prunes by dictionary contents in the decoder. The
/// upstream `Filter` always runs, so this is a pure optimization.
#[derive(Clone, Debug)]
struct PushedPredicate {
    /// The top-level column the comparison reads. For a variant path this is
    /// the variant column; the predicate prunes against the path's shredded
    /// leaf.
    column_idx: usize,
    /// The path inside the variant column (`CAST(col->'a'->'b' AS T) <cmp>
    /// const`), empty for a plain column comparison.
    path: JsonPath,
    compare_type: CompareType,
    value: Scalar<ArrayRef>,
}

impl PushedPredicate {
    /// The column-chunk index this predicate's statistics live on in `rg`: the
    /// column's own leaf for a plain predicate, or the shredded typed leaf for
    /// a variant path. `None` means pruning isn't sound for this row group
    /// (always safe): the path isn't shredded in this file, or some rows may
    /// hold the path's value in a binary `value` fallback along the path,
    /// where the typed leaf's statistics can't see them. The spec only allows
    /// stats-based skipping when every such fallback is all-null.
    fn get_leaf_for_row_group(&self, rg: &RowGroupMetadata) -> Option<usize> {
        let fields = rg.schema.fields();
        if self.path.is_empty() {
            return Some(first_leaf(fields, self.column_idx));
        }
        let leaves = variant_shredded_leaves(fields, self.column_idx, &self.path)?;
        let all_fallbacks_null = leaves.value_fallbacks.iter().all(|&leaf| {
            rg.leaf_statistics(leaf)
                .and_then(|stats| stats.null_count)
                .is_some_and(|null_count| null_count == rg.num_rows)
        });
        all_fallbacks_null.then_some(leaves.typed_value)
    }
}

/// Returns the column and optional variant path that can use row-group stats.
///
/// Plain columns use an empty path. Typed variant reads use the corresponding
/// shredded leaf. Untyped variant reads cannot be compared and are ignored.
fn get_prunable_column_and_json_path(expr: &Expression) -> Option<(usize, JsonPath)> {
    match expr {
        Expression::Ref(r) => Some((r.column_idx, Vec::new())),
        Expression::Function(Function::VariantGet(read)) if read.as_type.is_some() => {
            match read.input.as_ref() {
                Expression::Ref(r) => Some((r.column_idx, read.path.clone())),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A catalog table resolved from one transaction: it carries the table's durable
/// id, schema, and this query's pushed-down predicates, but no catalog state. The
/// file set is read from the transaction handed to the compile-time methods, so
/// the binding is a plain value and the transaction stays the single owner of the
/// frozen view. Cloned per-binding so each query accumulates its own predicates.
#[derive(Clone, Debug)]
pub struct TableBinding {
    /// The table's durable id, resolved once at binding time. Reads pin their
    /// view by the frozen snapshot regardless, but a write resolves the live
    /// table by this id at commit, so it survives a concurrent rename.
    id: uuid::Uuid,
    pub columns: Vec<Column>,
    /// Single-column predicates pushed down for this binding (recorded here
    /// because the `Table` trait gives no channel from `pushdown_filter` to
    /// `compile`); applied as a filter when the scan is compiled.
    predicates: Vec<PushedPredicate>,
}

impl TableBinding {
    /// A binding over `id`, with no predicates pushed yet.
    pub(super) fn new(id: uuid::Uuid, columns: Vec<Column>) -> Self {
        Self {
            id,
            columns,
            predicates: Vec::new(),
        }
    }

    /// This table's row groups in `transaction`'s snapshot, partition-pruned by
    /// the pushed equality predicates. Pure in-memory; the snapshot is
    /// immutable, so a scan and its late materialize (same filters) build
    /// identical views addressing the same global row-group indices. The
    /// transaction is always our own [`ParquetTransaction`] (a `ParquetCatalog`
    /// only ever compiles its own bindings), so the downcast is an invariant;
    /// failing it is an error, never a silent empty scan.
    fn resolve_files(
        &self,
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<Arc<ParquetTable>> {
        let transaction = transaction
            .as_any()
            .downcast_ref::<ParquetTransaction>()
            .ok_or_else(|| CatalogError::Other("transaction is not a ParquetTransaction".into()))?;
        // Resolve by the durable id, same as the write path. The snapshot is
        // frozen, so id and name would resolve the same table here; keying both
        // paths off the id keeps `name` a pure label. It's still the label in the
        // error, since a table missing from the snapshot has no id to look up.
        let table = transaction
            .snapshot
            .catalog_table_by_id(&self.id)
            .ok_or_else(|| {
                CatalogError::Other(format!("table {} is not in this snapshot", self.id).into())
            })?;
        let filters: Vec<PartitionEqFilter> = self.partition_filter_candidates().collect();
        table
            .build_scan_view(&filters)
            .map_err(|e| CatalogError::Other(Box::new(e)))
    }

    /// This binding's pushed equality predicates as partition-filter candidates:
    /// the column name and the typed scalar. The catalog intersects these with
    /// the table's partition columns, so yielding every equality predicate (not
    /// just ones on partition columns, which the binding can't tell apart) is fine
    /// — a non-partition column prunes no files.
    fn partition_filter_candidates(&self) -> impl Iterator<Item = PartitionEqFilter> + '_ {
        self.predicates
            .iter()
            .filter(|p| matches!(p.compare_type, CompareType::Equal))
            .filter_map(|p| {
                Some(PartitionEqFilter {
                    column: self.columns.get(p.column_idx)?.name.clone(),
                    value: p.value.clone(),
                })
            })
    }
}

impl Table for TableBinding {
    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // The transaction's snapshot hands back the file set this query was
        // bound against, partition-pruned. All in-memory: the background
        // refresh already materialized every footer.
        let current = self.resolve_files(transaction)?;

        // Dictionary pruning currently supports plain columns only. Shredded
        // leaves have file-specific positions and still use min/max pruning.
        let eq_predicates: Vec<ScanEqualityPredicate> = self
            .predicates
            .iter()
            .filter(|p| p.path.is_empty() && matches!(p.compare_type, CompareType::Equal))
            .map(|p| ScanEqualityPredicate {
                column_idx: p.column_idx,
                value: p.value.clone(),
            })
            .collect();

        // Prune the row groups by the pushed-down predicates' stats.
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

    fn compile_insert(
        &self,
        input: RecordBatchOperatorSpec,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        let transaction = transaction
            .as_any()
            .downcast_ref::<ParquetTransaction>()
            .ok_or_else(|| CatalogError::Other("transaction is not a ParquetTransaction".into()))?;
        // Resolve the live-view table by its durable id (not name), so the write
        // targets this exact table even if it was renamed since binding.
        let table = transaction
            .snapshot
            .catalog_table_by_id(&self.id)
            .ok_or_else(|| {
                CatalogError::Other(format!("table {} is not in this snapshot", self.id).into())
            })?;
        Ok(build_insert_spec(
            &table,
            transaction.uploaded_files.clone(),
            input,
            dispatcher,
        )?)
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
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Same transaction as the scan and its snapshot is immutable, so this
        // reads an identical view and their global row-group indices line up.
        // Late materialization re-reads rows by that global index, so it uses
        // the full table, not the pruned scan view.
        Ok(materialize(
            input,
            self.resolve_files(transaction)?,
            projection,
        ))
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> CatalogResult<bool> {
        let TableFilter::Expression(expr) = filter else {
            return Ok(false);
        };
        let Expression::Compare(compare) = expr.as_ref() else {
            return Ok(false);
        };
        let ((column_idx, path), constant) = match (compare.left.as_ref(), compare.right.as_ref()) {
            (column, Expression::Constant(k)) | (Expression::Constant(k), column) => {
                let Some(prunable) = get_prunable_column_and_json_path(column) else {
                    return Ok(false);
                };
                (prunable, k)
            }
            _ => return Ok(false),
        };

        // Just record it. The actual pruning (min/max row-group elimination and
        // equality/dictionary pruning) happens in `compile`, once the row-group
        // metadata exists. The upstream `Filter` is kept (we return `Ok(false)`),
        // so this is purely an optimization and never affects correctness.
        self.predicates.push(PushedPredicate {
            column_idx,
            path,
            compare_type: compare.compare_type,
            value: constant.clone(),
        });

        Ok(false)
    }

    fn column_min_max(
        &self,
        column: usize,
        transaction: &dyn CatalogTransaction,
    ) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // Read the transaction's snapshot files, the same view a scan would
        // see; an unresolvable snapshot means "scan".
        let parquet = self.resolve_files(transaction).ok()?;
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

    fn row_count(&self, transaction: &dyn CatalogTransaction) -> Option<i64> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // The parquet footer carries each row group's exact row count, so the
        // table's count is their sum, with no data pages read.
        let parquet = self.resolve_files(transaction).ok()?;
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
    /// binding's pushed-down predicates — i.e. what [`Table::compile_scan`] actually
    /// scans over the table's current files. A min/max stat that proves no row in
    /// a group can match drops it; a stats-comparison error means "can't prune"
    /// (kept) — never wrong, just unoptimized. No footer I/O. Exposed so pruning
    /// can be asserted directly.
    pub fn pruned_parquet(&self, parquet: &ParquetTable) -> ParquetTable {
        let mut parquet = parquet.clone();
        parquet.row_groups_mut().retain(|rg| {
            !self.predicates.iter().any(|p| {
                p.get_leaf_for_row_group(rg.as_ref()).is_some_and(|leaf| {
                    row_group_eliminated(rg.as_ref(), leaf, p.compare_type, &p.value)
                        .unwrap_or(false)
                })
            })
        });
        parquet
    }
}
