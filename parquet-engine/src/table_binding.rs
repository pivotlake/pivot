//! The binding of one query to a catalog table kept in Parquet files: the
//! filters pushed into it, and what a scan of the table reads once they are
//! applied, planned once and shared by everything the query asks of the
//! binding.
//!
//! A datastore says how its table is planned, counted and written
//! ([`ParquetBackedTable`]); the binding ([`ParquetTableBinding`]) is the same
//! for all of them.

use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Result as CatalogResult, TableReference,
    TableRevision,
};
use planner::expression::Expression;

use crate::{
    ParquetTable, PushedPredicate, equality_predicates, materialize, prune_parquet,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};

/// What a scan over some filters reads of a table version.
pub struct ScanPlan {
    /// The row groups the scan reads, in an order the version and the
    /// filters fix.
    pub parquet: Arc<ParquetTable>,
    /// Whether each column (in the table's column order) can hold a NULL in
    /// the files the scan reads, as the data shows it, not as the schema
    /// declares it. A column is reported free of NULLs only when that is
    /// proven for every one of those files, so a table may answer for more
    /// files than the scan reads. A column
    /// [`ParquetBackedTable::list_never_null_columns`] lists is free of NULLs;
    /// any other is reported nullable by a scan that reads no file.
    pub nullability: Vec<bool>,
}

/// One version of a catalog table as a [`ParquetTableBinding`] reads and
/// writes it: what differs between the datastores whose tables are Parquet
/// files.
pub trait ParquetBackedTable: Clone + std::fmt::Debug + Send + Sync + 'static {
    /// The identity and version of the table a plan over it is cached under.
    fn revision(&self) -> TableRevision;

    fn columns(&self) -> Vec<Column>;

    /// Whether each column (in [`columns`](Self::columns) order) holds a
    /// NULL in no file of this version, as far as the metadata held proves
    /// it with nothing read.
    fn list_never_null_columns(&self) -> Vec<bool>;

    /// What a scan over `filters`, the predicates pushed into it, reads:
    /// the row groups of the files this version's metadata cannot rule out,
    /// in an order fixed by the version, so the same filters yield the same
    /// row-group positions for scanning and materialization.
    fn plan_scan(&self, filters: &[Expression]) -> CatalogResult<ScanPlan>;

    /// The rows a scan of the whole table returns, when this version's
    /// metadata counts them exactly; `None` when only a scan can.
    fn count_rows(&self) -> Option<i64>;

    /// The size of the whole table as the cost model takes it; `None` when
    /// the metadata gives none.
    fn estimate_row_count(&self) -> Option<u64>;

    /// Whether a scan of this table applies the variant field extracts
    /// pushed into its projection (see
    /// [`BoundTable::applies_variant_extracts`]).
    fn applies_variant_extracts(&self) -> bool;

    /// The dataflow writing the rows of `input` into the table (see
    /// [`BoundTable::compile_insert`]).
    fn compile_insert(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> CatalogResult<RecordBatchOperatorSpec>;
}

/// One reference to a table in a query: the version of the table the
/// query's transaction resolved, and the filters pushed into this
/// reference. Cloned per reference, so each accumulates its own filters.
#[derive(Clone)]
pub struct ParquetTableBinding<Table> {
    /// The qualified name this binding was resolved under, as the catalog
    /// routed it.
    reference: TableReference,
    table: Table,
    pub columns: Vec<Column>,
    /// The filters pushed down for this binding. The table selects files
    /// by them whole; the single-column comparisons among them prune row
    /// groups and feed the scan's equality checks.
    filters: Vec<Expression>,
    /// What a scan of this binding reads, once something asked for it:
    /// nullability, statistics, scanning and late materialization, the last
    /// compiled from a clone of this binding, reuse the one load. A plan
    /// kept for a later execution keeps the footers it reads here.
    cached_scan_plan: Arc<Mutex<Option<Arc<ScanPlan>>>>,
}

impl<Table: ParquetBackedTable> std::fmt::Debug for ParquetTableBinding<Table> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetTableBinding")
            .field("reference", &self.reference)
            .field("table", &self.table)
            .field("filters", &self.filters)
            .finish_non_exhaustive()
    }
}

impl<Table: ParquetBackedTable> ParquetTableBinding<Table> {
    /// A binding over `table` with no filters pushed yet. Nothing of the
    /// table is read.
    pub fn new(reference: TableReference, table: Table) -> Self {
        let columns = table.columns();
        Self {
            reference,
            table,
            columns,
            filters: Vec::new(),
            cached_scan_plan: Arc::default(),
        }
    }

    /// Clone `parquet`'s row groups and keep only those that survive this
    /// binding's pushed-down predicates. A min/max stat that proves no row
    /// in a group can match drops it; a stats-comparison error keeps the
    /// group. No footer I/O. Exposed so pruning can be asserted directly.
    pub fn pruned_parquet(&self, parquet: &ParquetTable) -> ParquetTable {
        prune_parquet(parquet, &self.list_pushed_predicates())
    }

    /// What a scan of this binding reads: the files the pushed filters
    /// cannot rule out in the bound version, their row groups less the ones
    /// their own statistics rule out. Planned by whatever asks first, under
    /// a lock the clones share, and kept on a successful load, so a scan
    /// and its late materialize address the same global row-group indices
    /// and no footer is read twice.
    fn load_scan_plan(&self) -> CatalogResult<Arc<ScanPlan>> {
        let mut cached = self.cached_scan_plan.lock().unwrap();
        if let Some(plan) = cached.as_ref() {
            return Ok(Arc::clone(plan));
        }
        let selected = self.table.plan_scan(&self.filters)?;
        let plan = Arc::new(ScanPlan {
            parquet: Arc::new(self.pruned_parquet(&selected.parquet)),
            nullability: selected.nullability,
        });
        *cached = Some(Arc::clone(&plan));
        Ok(plan)
    }

    /// The row groups a scan of this binding reads.
    fn load_pruned_parquet(&self) -> CatalogResult<Arc<ParquetTable>> {
        Ok(Arc::clone(&self.load_scan_plan()?.parquet))
    }

    /// The single-column constant comparisons among the pushed filters:
    /// what a row group's statistics can be tested against.
    fn list_pushed_predicates(&self) -> Vec<PushedPredicate> {
        self.filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect()
    }

    /// Whether the scan this binding stands for reads the whole table.
    fn is_unfiltered(&self) -> bool {
        self.filters.is_empty()
    }
}

impl<Table: ParquetBackedTable> BoundTable for ParquetTableBinding<Table> {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.table.revision()
    }

    fn supports_late_materialization(&self) -> bool {
        true
    }

    fn applies_variant_extracts(&self) -> bool {
        self.table.applies_variant_extracts()
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
        let eq_predicates = equality_predicates(&self.list_pushed_predicates());

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
        self.table.compile_insert(input)
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    /// The columns the table's metadata shows never hold a NULL are answered
    /// with nothing read; the others for the files the pushed filters keep.
    /// A scan that cannot be planned may hold a NULL in any of those;
    /// compiling it reports what is wrong.
    fn nullability(&self) -> Vec<bool> {
        let never_null = self.table.list_never_null_columns();
        if never_null.iter().all(|is_never_null| *is_never_null) {
            return vec![false; never_null.len()];
        }
        match self.load_scan_plan() {
            Ok(plan) => plan.nullability.clone(),
            Err(_) => never_null
                .iter()
                .map(|is_never_null| !is_never_null)
                .collect(),
        }
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
        // list, so this has to build the identical view: same bound version,
        // same file selection, and the same stats pruning
        // [`compile_scan`](BoundTable::compile_scan) applies. Re-reading from a
        // view that kept even one row group the scan dropped shifts every later
        // index and silently returns another row group's rows.
        Ok(materialize(input, self.load_pruned_parquet()?, projection))
    }

    fn pushdown_filter(&mut self, filter: Expression) -> CatalogResult<bool> {
        // Just record it: pruning happens when the scan is planned. The
        // upstream `Filter` is kept (`Ok(false)`), so pruning only excludes
        // work and never decides which rows the query returns.
        self.filters.push(filter);
        // What was planned for fewer filters is not this binding's view; it
        // stays the view of the clones made before this filter.
        self.cached_scan_plan = Arc::default();

        Ok(false)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down filter
        // means the scan this binding stands for excludes rows.
        if !self.is_unfiltered() {
            return None;
        }
        // Exact extrema come from the row groups a scan would read; a view
        // that cannot be planned means "scan", which reports what is wrong.
        self.load_pruned_parquet().ok()?.column_min_max(column)
    }

    fn row_count(&self) -> Option<i64> {
        // Only sound for the whole, unfiltered table: a pushed-down filter
        // means the scan this binding stands for excludes rows.
        if !self.is_unfiltered() {
            return None;
        }
        self.table.count_rows()
    }

    fn estimate_row_count(&self) -> Option<u64> {
        // A planning estimate wants the base table's full size: pushed
        // predicates deliberately don't apply, since the cost model accounts
        // for filter selectivity itself.
        self.table.estimate_row_count()
    }
}
