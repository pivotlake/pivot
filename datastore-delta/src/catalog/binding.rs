//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! transaction resolve derives from its snapshot, so a query's filter pushdown
//! prunes its own view without affecting anyone else.

use std::sync::Arc;

use crate::manifest::{ColumnStatFilter, PartitionEqFilter};
use crate::parquet::types::leaves::{first_leaf, variant_shredded_leaves};
use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::{
    AppliedPredicate, ParquetTable, ScanEqualityPredicate, materialize, row_group_eliminated,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use arrow_array::{Array, ArrayRef, Scalar};
use crossbeam_deque::Injector;
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Error as CatalogError, Result as CatalogResult,
    TableReference, TableRevision,
};
use planner::expression::{CompareType, Expression, Function, JsonPath, TableFilter};
use planner::types::physical_arrow_type;

use super::CatalogTable;
use super::insert_sink::{UploadedFile, build_insert_spec};

/// A single-column constant comparison (`col <cmp> const`) pushed down by
/// DuckDB during binding. Recorded as-is; applied at [`compile_scan`](BoundTable::compile_scan)
/// time after the row-group metadata exists: min/max stats prune row groups,
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
    /// hold the path's value in an untyped `value` leaf along the path, where
    /// the typed leaf's statistics can't see them. The spec only allows
    /// stats-based skipping when every such value leaf is all-null.
    fn get_leaf_for_row_group(&self, rg: &RowGroupMetadata) -> Option<usize> {
        let fields = rg.schema.fields();
        if self.path.is_empty() {
            return Some(first_leaf(fields, self.column_idx));
        }
        let leaves = variant_shredded_leaves(fields, self.column_idx, &self.path)?;
        let all_value_leaves_null = leaves.value_leaves.iter().all(|&leaf| {
            rg.leaf_statistics(leaf)
                .and_then(|stats| stats.null_count)
                .is_some_and(|null_count| null_count == rg.num_rows)
        });
        all_value_leaves_null.then_some(leaves.typed_leaf)
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

/// A catalog table resolved from one transaction: it captures the snapshot's own
/// copy of the [`CatalogTable`] (schema, files, and their row groups), this
/// query's pushed-down predicates, and the transaction's uploaded-files injector
/// for the write path. Because the frozen snapshot copy travels with the binding,
/// every compile-time method resolves its file set with no transaction handle.
/// Cloned per-binding so each query accumulates its own predicates.
#[derive(Clone)]
pub struct TableBinding {
    /// The qualified name this binding was resolved under, as the catalog routed
    /// it. Recorded from the bind call rather than read back off `table`, so a
    /// later rename cannot drift it away from the name the query bound.
    reference: TableReference,
    /// The snapshot's frozen copy of this table: its schema and the row groups of
    /// every committed file at the version the transaction opened. Reads build
    /// their scan view straight from it; a write resolves the live table by its
    /// durable id ([`CatalogTable::id`]) at commit, so it survives a concurrent
    /// rename.
    table: CatalogTable,
    pub columns: Vec<Column>,
    /// Per-column footer-derived nullability, in `columns` order (see
    /// [`CatalogTable::nullability`](super::table::CatalogTable)).
    nullability: Vec<bool>,
    /// Single-column predicates pushed down for this binding (recorded here
    /// because the `BoundTable` trait gives no channel from `pushdown_filter` to
    /// `compile`); applied as a filter when the scan is compiled.
    predicates: Vec<PushedPredicate>,
    /// The subset of `predicates` this binding reported as fully handled, so
    /// the plan carries no `Filter` for them and the scan must apply each one
    /// exactly to every row it emits.
    owned_predicates: Vec<PushedPredicate>,
    /// The transaction's shared queue of finished INSERT files. Shared (`Arc`)
    /// with the [`DeltaTransaction`](super::DeltaTransaction) that produced this
    /// binding, so a file this binding's [`compile_insert`](BoundTable::compile_insert)
    /// pushes is drained by that transaction's commit.
    uploaded_files: Arc<Injector<UploadedFile>>,
}

impl std::fmt::Debug for TableBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableBinding")
            .field("reference", &self.reference)
            .field("id", &self.table.id())
            .field("predicates", &self.predicates)
            .finish_non_exhaustive()
    }
}

impl TableBinding {
    /// A binding over the snapshot's `table` copy, sharing the transaction's
    /// `uploaded_files` injector, with no predicates pushed yet.
    pub(super) fn new(
        reference: TableReference,
        table: CatalogTable,
        uploaded_files: Arc<Injector<UploadedFile>>,
    ) -> Self {
        let columns = table.columns();
        let nullability = table.nullability();
        Self {
            reference,
            table,
            columns,
            nullability,
            predicates: Vec::new(),
            owned_predicates: Vec::new(),
            uploaded_files,
        }
    }

    /// This table's row groups in the captured snapshot, partition-pruned by the
    /// pushed equality predicates. Pure in-memory; the snapshot copy is
    /// immutable, so a scan and its late materialize (same filters) build
    /// identical views addressing the same global row-group indices.
    fn resolve_files(&self) -> CatalogResult<Arc<ParquetTable>> {
        let partition_filters: Vec<PartitionEqFilter> =
            self.partition_filter_candidates().collect();
        let stat_filters: Vec<ColumnStatFilter> = self.stat_filter_candidates().collect();
        self.table
            .build_scan_view(&partition_filters, &stat_filters)
            .map_err(|e| CatalogError::Other(Box::new(e)))
    }

    /// This binding's pushed equality predicates as partition-filter candidates:
    /// the column name and the typed scalar. The catalog intersects these with
    /// the table's partition columns, so yielding every equality predicate (not
    /// just ones on partition columns, which the binding can't tell apart) is fine;
    /// a non-partition column prunes no files.
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

    /// This binding's pushed predicates as file-level stat-filter candidates: a
    /// plain top-level column comparison (no variant path) paired with its typed
    /// constant. Variant paths are excluded — their stats live in a shredded leaf,
    /// pruned per row group, not in the file's column stats. A column the file's
    /// stats don't bound simply prunes no files.
    fn stat_filter_candidates(&self) -> impl Iterator<Item = ColumnStatFilter> + '_ {
        self.predicates
            .iter()
            .filter(|p| p.path.is_empty())
            .filter_map(|p| {
                Some(ColumnStatFilter {
                    column: self.columns.get(p.column_idx)?.name.clone(),
                    compare_type: p.compare_type,
                    value: p.value.clone(),
                })
            })
    }
}

impl BoundTable for TableBinding {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: self.table.id().to_string(),
            version: self.table.version(),
        }
    }

    /// Each row group resolves a pushed path against its own shredding layout,
    /// so the scan reads only the leaves the path needs.
    fn applies_variant_extracts(&self) -> bool {
        true
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // The captured snapshot copy hands back the file set this query was
        // bound against, partition-pruned. All in-memory: the background
        // refresh already materialized every footer.
        let current = self.resolve_files()?;

        // A variant path predicate reaches its shredded typed leaf only in the
        // files that shred it, so each row group resolves the path itself.
        let eq_predicates: Vec<ScanEqualityPredicate> = self
            .predicates
            .iter()
            .filter(|p| matches!(p.compare_type, CompareType::Equal))
            .map(|p| ScanEqualityPredicate {
                column_idx: p.column_idx,
                path: p.path.clone(),
                value: p.value.clone(),
            })
            .collect();

        // A claimed predicate has no `Filter` above it any more, so the scan
        // must read its column even when the projection dropped it (a column
        // nothing else selects, e.g. `COUNT(*) WHERE col <> 'x'`). Extra
        // columns go on the end, and the decoder emits only the leading
        // `output_columns` so the batch keeps the schema the plan expects.
        let output_columns = projection.column_indices.len();
        let mut projection = projection;
        let mut applied: Vec<AppliedPredicate> = Vec::new();
        for predicate in &self.owned_predicates {
            let position = projection
                .column_indices
                .iter()
                .position(|&c| c == predicate.column_idx)
                .unwrap_or_else(|| {
                    projection.column_indices.push(predicate.column_idx);
                    if !projection.extracts.is_empty() {
                        projection.extracts.push(None);
                    }
                    projection.column_indices.len() - 1
                });
            applied.push(AppliedPredicate {
                output_idx: position,
                compare_type: predicate.compare_type,
                value: predicate.value.clone(),
            });
        }

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
            Arc::new(applied),
            output_columns,
        ))
    }

    fn compile_insert(
        &self,
        input: RecordBatchOperatorSpec,
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // The captured snapshot copy stamps the durable schema and target table
        // id onto every written file; the shared injector hands each finished
        // file to the transaction's commit, which resolves the live table by
        // that id (surviving a concurrent rename).
        Ok(build_insert_spec(
            &self.table,
            self.uploaded_files.clone(),
            input,
            dispatcher,
        )?)
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        self.nullability.clone()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // A row reference is a position in the *scanning* view's flat row-group
        // list, so this has to build the identical view: same captured
        // snapshot, same partition pruning, and the same stats pruning
        // [`compile_scan`](BoundTable::compile_scan) applies. Re-reading from a
        // view that kept even one row group the scan dropped shifts every later
        // index and silently returns another row group's rows.
        let current = self.resolve_files()?;
        Ok(materialize(
            input,
            Arc::new(self.pruned_parquet(&current)),
            projection,
        ))
    }

    fn pushdown_filter(
        &mut self,
        filter: TableFilter,
        scan_column_count: usize,
    ) -> CatalogResult<bool> {
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

        let predicate = PushedPredicate {
            column_idx,
            path,
            compare_type: compare.compare_type,
            value: constant.clone(),
        };

        // Record it either way: min/max row-group elimination and
        // equality/dictionary pruning read `predicates` when the scan is
        // compiled, whoever ends up applying the comparison per row.
        self.predicates.push(predicate.clone());

        // Claiming it removes the plan's `Filter` for this condition, so the
        // scan alone decides which rows survive and the column it reads need
        // not be projected at all. Only shapes the scan can evaluate exactly
        // are claimed; everything else keeps the `Filter` above.
        if self.can_own_predicate(&predicate, scan_column_count) {
            self.owned_predicates.push(predicate);
            return Ok(true);
        }

        Ok(false)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // Read the captured snapshot files, the same view a scan would see; an
        // unresolvable view means "scan".
        let parquet = self.resolve_files().ok()?;
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

    fn row_count(&self) -> Option<i64> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // The parquet footer carries each row group's exact row count, so the
        // table's count is their sum, with no data pages read.
        let parquet = self.resolve_files().ok()?;
        Some(parquet.row_groups().iter().map(|rg| rg.num_rows).sum())
    }

    fn estimate_row_count(&self) -> Option<u64> {
        // A planning estimate wants the base table's full size: pushed
        // predicates and partition pruning deliberately don't apply, since the
        // cost model accounts for filter selectivity itself. Build the whole
        // (unfiltered) view over the captured snapshot and sum the footers'
        // row-group counts.
        let parquet = self.table.build_scan_view(&[], &[]).ok()?;
        Some(
            parquet
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows as u64)
                .sum(),
        )
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
    fn can_own_predicate(&self, predicate: &PushedPredicate, scan_column_count: usize) -> bool {
        // Owning the comparison means the scan drops the failing rows itself,
        // which gathers every column the batch carries. The plan's `Filter`
        // does that far better - it keeps a selection vector and only
        // materializes when a cost model says it pays - so owning is only
        // worth it when there is next to nothing to gather. One column is the
        // shape that wins: the comparison's own, read for nothing else, which
        // projection pushdown then drops entirely once the condition is gone.
        if scan_column_count > 1 {
            return false;
        }
        // A variant path reads a shredded leaf that only some files carry, so
        // the scan cannot promise to evaluate it everywhere.
        if !predicate.path.is_empty() {
            return false;
        }
        // Equality is the whole vocabulary here: an ordering comparison would
        // be just as easy to apply, but keeping the claim narrow keeps the set
        // of shapes the scan must get exactly right small.
        if !matches!(
            predicate.compare_type,
            CompareType::Equal | CompareType::NotEqual
        ) {
            return false;
        }
        // The comparison runs as an Arrow kernel against the decoded column, so
        // the constant has to already be the column's physical type; a mismatch
        // would error at scan time, with no `Filter` left to fall back on.
        let Some(column) = self.columns.get(predicate.column_idx) else {
            return false;
        };
        let arrow_type = physical_arrow_type(&column.col_type);
        if arrow_array::Datum::get(&predicate.value).0.data_type() != &arrow_type {
            return false;
        }
        // Owning only pays when the comparison can be answered from the
        // chunk's dictionary keys, which needs every data page dictionary
        // encoded. A column whose values are too varied for a dictionary (a
        // user id, a hash) falls back to plain pages, and then owning buys
        // nothing over the `Filter` while still costing it a gather.
        self.column_is_all_dictionary(predicate.column_idx)
    }

    /// Whether every row group encodes `column_idx` entirely with its
    /// dictionary, so the scan can answer a comparison from the keys. An
    /// unreadable file set or a missing chunk reads as `false`.
    fn column_is_all_dictionary(&self, column_idx: usize) -> bool {
        let Ok(parquet) = self.resolve_files() else {
            return false;
        };
        let row_groups = parquet.row_groups();
        !row_groups.is_empty()
            && row_groups.iter().all(|rg| {
                let leaf = first_leaf(rg.schema.fields(), column_idx);
                rg.columns
                    .get(leaf)
                    .is_some_and(|chunk| chunk.data_pages_all_dictionary)
            })
    }

    /// Clone `parquet`'s row groups and keep only those that survive this
    /// binding's pushed-down predicates — i.e. what [`BoundTable::compile_scan`] actually
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
