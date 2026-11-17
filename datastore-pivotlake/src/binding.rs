//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! transaction resolve derives from its snapshot, so a query's filter pushdown
//! prunes its own view without affecting anyone else.

use std::sync::Arc;

use crate::manifest::{ColumnStatFilter, PartitionEqFilter};
use crate::text_index::{TextIndexRowSelector, TokenPredicate};
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use crossbeam_deque::Injector;
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec, RowDelivery};
use parquet_engine::{
    ParquetTable, PushedPredicate, equality_predicates, materialize, prune_parquet,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
    table_input_with_row_selector,
};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Error as CatalogError, Result as CatalogResult,
    TableReference, TableRevision,
};
use planner::expression::{CompareType, Expression, Function, HasTokens, Ref};
use planner::operator::Filter;
use planner::types::Type;

use super::insert_sink::build_insert_spec;
use super::{CatalogTable, CommitAction};

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
    /// Token predicates (`has_token` and its siblings) pushed down for this
    /// binding, each on a plain text column some file has a text index for.
    /// The scan reads only the rows those indexes say hold the tokens, unless
    /// it feeds a Top-N, which checks them on the text.
    token_predicates: Vec<TokenPredicate>,
    /// The transaction's staging queue, shared (`Arc`) with the
    /// [`PivotlakeTransaction`](super::PivotlakeTransaction) that produced this
    /// binding, so the append an INSERT compiled by this binding stages as it
    /// runs is applied by that transaction's commit.
    commit_actions: Arc<Injector<Box<dyn CommitAction>>>,
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
    /// staging queue, with no predicates pushed yet.
    pub(super) fn new(
        reference: TableReference,
        table: CatalogTable,
        commit_actions: Arc<Injector<Box<dyn CommitAction>>>,
    ) -> Self {
        let columns = table.columns();
        let nullability = table.nullability();
        Self {
            reference,
            table,
            columns,
            nullability,
            predicates: Vec::new(),
            token_predicates: Vec::new(),
            commit_actions,
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
            version: self.table.version().to_string(),
        }
    }

    fn supports_late_materialization(&self) -> bool {
        true
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
        let eq_predicates = equality_predicates(&self.predicates);

        // Prune the row groups by the pushed-down predicates' stats.
        let parquet = Arc::new(self.pruned_parquet(&current));
        // Order the scan by the Top-N's key so its boundary tightens after the
        // first row group and the rest get pruned, instead of racing file order.
        let scan_order = scan_order_from(&dynamic_filters);
        // The filter is consulted as a row group is handed out, so it does
        // not shift the row-group positions a late materialize addresses.
        let filter = row_group_filter_from(dynamic_filters);

        // A token predicate is answered from the text indexes: a stage ahead
        // of the scan's reads looks each file up in its index, and only the
        // rows holding the tokens are read.
        if !self.token_predicates.is_empty() && scan_order.is_none() {
            return Ok(table_input_with_row_selector(
                dispatcher,
                &parquet,
                projection,
                emit_row_group_metadata,
                filter,
                Arc::new(eq_predicates),
                TextIndexRowSelector::new(&self.token_predicates),
            ));
        }

        // A scan feeding a Top-N is read in the order of its key and under
        // its throttle, and uses no index. The token predicates the plan
        // keeps no filter for are checked here, on the text: the scan reads
        // their columns too, where the query did not ask for them, and drops
        // those again once the rows are checked.
        let requested_columns = projection.column_indices.len();
        let mut scanned = projection;
        let mut conditions = Vec::new();
        for predicate in &self.token_predicates {
            if !predicate.is_answered_by_the_scan {
                continue;
            }
            let read_whole_at = (0..scanned.column_indices.len()).find(|&output| {
                scanned.column_indices[output] == predicate.column_idx
                    && scanned.extract_at(output).is_none()
            });
            let position = match read_whole_at {
                Some(position) => position,
                None => {
                    scanned.column_indices.push(predicate.column_idx);
                    if !scanned.extracts.is_empty() {
                        scanned.extracts.push(None);
                    }
                    scanned.column_indices.len() - 1
                }
            };
            conditions.push(Expression::Function(Function::HasTokens(HasTokens {
                input: Box::new(Expression::Ref(Ref {
                    column_idx: position,
                    return_type: Type::Utf8,
                    name: Some(self.columns[predicate.column_idx].name.clone()),
                })),
                ..predicate.call.clone()
            })));
        }
        let added_columns = requested_columns..scanned.column_indices.len();
        let scan = table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            scanned,
            emit_row_group_metadata,
            filter,
            scan_order,
            Arc::new(eq_predicates),
        );
        if conditions.is_empty() {
            return Ok(scan);
        }
        let checked = Filter {
            conditions,
            // What a filter below a Top-N delivers.
            delivery: RowDelivery::Immediate,
        }
        .compile(scan)
        .map_err(|error| CatalogError::Other(Box::new(error)))?;
        if added_columns.is_empty() {
            return Ok(checked);
        }
        // The added columns follow the requested ones, ahead of any row-group
        // metadata columns.
        Ok(RecordBatchOperatorSpec::from_spec(checked.map(move || {
            let added_columns = added_columns.clone();
            move |batch: RecordBatch| {
                let kept: Vec<usize> = (0..batch.num_columns())
                    .filter(|column| !added_columns.contains(column))
                    .collect();
                Ok(batch.project(&kept)?)
            }
        })))
    }

    fn compile_insert(
        &self,
        input: RecordBatchOperatorSpec,
        _dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // The captured snapshot copy stamps the durable schema onto every
        // written file. Compiling stages nothing: the dataflow stages its
        // append on the transaction's queue as it starts, and the transaction's
        // commit resolves the live table by its durable id (surviving a
        // concurrent rename).
        Ok(build_insert_spec(
            &self.table,
            self.commit_actions.clone(),
            input,
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

    fn pushdown_filter(&mut self, filter: Expression) -> CatalogResult<bool> {
        // A comparison is just recorded. The actual pruning (min/max row-group
        // elimination and equality/dictionary pruning) happens in `compile`,
        // once the row-group metadata exists, and the upstream `Filter` is
        // kept, so it is purely an optimization.
        self.predicates
            .extend(PushedPredicate::from_filter(&filter));

        // A token function over a plain text column is answered from the
        // column's text indexes, for the tokens an index stores.
        let Expression::Function(Function::HasTokens(has_tokens)) = &filter else {
            return Ok(false);
        };
        let Expression::Ref(column) = has_tokens.input.as_ref() else {
            return Ok(false);
        };
        let query = has_tokens.query();
        if !query.is_answerable_from_index() {
            return Ok(false);
        }
        // The same call offered again gets the same answer.
        let already_pushed = self
            .token_predicates
            .iter()
            .find(|pushed| pushed.column_idx == column.column_idx && *pushed.query == query);
        if let Some(pushed) = already_pushed {
            return Ok(pushed.is_answered_by_the_scan);
        }
        let indexes = self
            .table
            .column_text_indexes(&self.columns[column.column_idx].name)?;
        if indexes.is_empty() {
            return Ok(false);
        }
        // The scan reads exactly the matching rows of every file that has an
        // index, and every row of a file that has none. So the filter is
        // needed above the scan only when some file has none.
        let is_answered_by_the_scan = indexes.covers_every_file();
        self.token_predicates.push(TokenPredicate {
            call: has_tokens.clone(),
            column_idx: column.column_idx,
            query: Arc::new(query),
            indexes,
            is_answered_by_the_scan,
        });
        Ok(is_answered_by_the_scan)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() || !self.token_predicates.is_empty() {
            return None;
        }
        // Read the captured snapshot files, the same view a scan would see; an
        // unresolvable view means "scan".
        self.resolve_files().ok()?.column_min_max(column)
    }

    fn row_count(&self) -> Option<i64> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() || !self.token_predicates.is_empty() {
            return None;
        }
        // The parquet footer carries each row group's exact row count, so the
        // table's count is their sum, with no data pages read.
        let parquet = self.resolve_files().ok()?;
        Some(parquet.total_rows())
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

impl TableBinding {
    /// Clone `parquet`'s row groups and keep only those that survive this
    /// binding's pushed-down predicates — i.e. what [`BoundTable::compile_scan`] actually
    /// scans over the table's current files. A min/max stat that proves no row in
    /// a group can match drops it; a stats-comparison error means "can't prune"
    /// (kept) — never wrong, just unoptimized. No footer I/O. Exposed so pruning
    /// can be asserted directly.
    pub fn pruned_parquet(&self, parquet: &ParquetTable) -> ParquetTable {
        prune_parquet(parquet, &self.predicates)
    }
}
