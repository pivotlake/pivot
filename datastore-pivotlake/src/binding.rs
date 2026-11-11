//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! transaction resolve derives from its snapshot, so a query's filter pushdown
//! prunes its own view without affecting anyone else.

use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Scalar};
use catalog::CommitQueue;
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use parquet_engine::{
    ParquetTable, PushedPredicate, equality_predicates, materialize, prune_parquet,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Error as CatalogError, Result as CatalogResult,
    TableReference, TableRevision,
};
use planner::expression::Expression;

use super::insert_sink::build_insert_spec;
use super::{CatalogTable, PivotlakeDatastore};

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
    /// The filters pushed down for this binding. The table's format selects
    /// files by them whole; the single-column comparisons among them prune
    /// row groups and feed the scan's equality checks.
    filters: Vec<Expression>,
    /// The row groups a scan of this binding reads, once something asked
    /// for them: statistics, scanning and late materialization, the last
    /// compiled from a clone of this binding, reuse the one load. The table
    /// itself holds no file's footer, so this is where a plan kept for a
    /// later execution keeps the footers it reads.
    cached_pruned_parquet: Arc<Mutex<Option<Arc<ParquetTable>>>>,
    /// The transaction's staging queue, shared (`Arc`) with the
    /// [`PivotlakeTransaction`](super::PivotlakeTransaction) that produced this
    /// binding, so the append an INSERT compiled by this binding stages as it
    /// runs is applied by that transaction's commit.
    commit_actions: CommitQueue<PivotlakeDatastore>,
}

impl std::fmt::Debug for TableBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableBinding")
            .field("reference", &self.reference)
            .field("id", &self.table.id())
            .field("filters", &self.filters)
            .finish_non_exhaustive()
    }
}

impl TableBinding {
    /// A binding over the snapshot's `table` copy, sharing the transaction's
    /// staging queue, with no predicates pushed yet.
    pub(super) fn new(
        reference: TableReference,
        table: CatalogTable,
        commit_actions: CommitQueue<PivotlakeDatastore>,
    ) -> CatalogResult<Self> {
        let columns = table.columns();
        let nullability = table
            .nullability()
            .map_err(|e| CatalogError::Other(Box::new(e)))?;
        Ok(Self {
            reference,
            table,
            columns,
            nullability,
            filters: Vec::new(),
            cached_pruned_parquet: Arc::default(),
            commit_actions,
        })
    }

    /// The row groups a scan of this binding reads: those of the files the
    /// pushed filters cannot rule out in the captured snapshot, less the
    /// ones their own statistics rule out. Planned by whatever asks first
    /// and kept on a successful load, so a scan and its late materialize
    /// address the same global row-group indices and no footer is read
    /// twice.
    fn load_pruned_parquet(&self) -> CatalogResult<Arc<ParquetTable>> {
        let mut cached = self.cached_pruned_parquet.lock().unwrap();
        if let Some(parquet) = cached.as_ref() {
            return Ok(Arc::clone(parquet));
        }
        let selected = self
            .table
            .plan_scan(&self.filters)
            .map_err(|e| CatalogError::Other(Box::new(e)))?;
        let parquet = Arc::new(self.pruned_parquet(&selected));
        *cached = Some(Arc::clone(&parquet));
        Ok(parquet)
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
        // The file set this query was bound against, less what the pushed
        // predicates rule out by partition, by file bounds and by row-group
        // statistics.
        let parquet = self.load_pruned_parquet()?;

        // A variant path predicate reaches its shredded typed leaf only in the
        // files that shred it, so each row group resolves the path itself.
        let eq_predicates = equality_predicates(&self.pushed_predicates());

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
        Ok(materialize(input, self.load_pruned_parquet()?, projection))
    }

    fn pushdown_filter(&mut self, filter: Expression) -> CatalogResult<bool> {
        // Just record it. The actual pruning (min/max row-group elimination and
        // equality/dictionary pruning) happens in `compile`, once the row-group
        // metadata exists. The upstream `Filter` is kept (we return `Ok(false)`),
        // so this is purely an optimization and never affects correctness.
        self.filters.push(filter);
        // What was planned for fewer filters is not this binding's view.
        self.cached_pruned_parquet = Arc::default();

        Ok(false)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down filter
        // means the scan this binding stands for excludes rows.
        if !self.filters.is_empty() {
            return None;
        }
        // Exact extrema come from the row groups a scan would read; an
        // unresolvable view means "scan", which reports what is wrong.
        self.load_pruned_parquet().ok()?.column_min_max(column)
    }

    fn row_count(&self) -> Option<i64> {
        // Only sound for the whole, unfiltered table: a pushed-down filter
        // means the scan this binding stands for excludes rows.
        if !self.filters.is_empty() {
            return None;
        }
        // What the captured snapshot's metadata counts; a count it cannot
        // give means "scan", which also reports what is wrong.
        let rows = self.table.count_rows().ok()??;
        i64::try_from(rows).ok()
    }

    fn estimate_row_count(&self) -> Option<u64> {
        // A planning estimate wants the base table's full size: pushed
        // predicates and partition pruning deliberately don't apply, since the
        // cost model accounts for filter selectivity itself. The captured
        // snapshot's metadata has the count.
        self.table.total_rows().ok()
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
        prune_parquet(parquet, &self.pushed_predicates())
    }

    /// The single-column constant comparisons among the pushed filters: what
    /// a row group's statistics can be tested against.
    fn pushed_predicates(&self) -> Vec<PushedPredicate> {
        self.filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect()
    }
}
