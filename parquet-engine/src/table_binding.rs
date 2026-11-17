//! The binding of one query to a catalog table kept in Parquet files: the
//! filters pushed into it, and what a scan of the table reads once they are
//! applied, selected once and shared by everything the query asks of the
//! binding.
//!
//! A datastore says how its table is scanned, counted and written
//! ([`ParquetBackedTable`]); the binding ([`ParquetTableBinding`]) is the same
//! for all of them.

use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, OneShotNullaryFactory, Projection, RecordBatchOperatorSpec};
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
pub struct ScanSelection {
    /// The row groups the scan reads, in an order the version and the
    /// filters fix.
    pub row_groups: Arc<ParquetTable>,
    /// Whether each column (in the table's column order) can hold a NULL in
    /// the files the scan reads, as the data shows it, not as the schema
    /// declares it. A column is reported free of NULLs only when that is
    /// proven for every one of those files, so a table may answer for more
    /// files than the scan reads. A column
    /// [`ParquetBackedTable::declared_nullability`] reports free of NULLs is
    /// free of them here; any other is reported nullable by a scan that
    /// reads no file.
    pub nullability: Vec<bool>,
}

/// One version of a catalog table as a [`ParquetTableBinding`] reads and
/// writes it: what differs between the datastores whose tables are Parquet
/// files.
pub trait ParquetBackedTable: std::fmt::Debug + Send + Sync {
    /// The identity and version of the table a plan over it is cached under.
    fn revision(&self) -> TableRevision;

    fn columns(&self) -> Vec<Column>;

    /// Whether each column (in [`columns`](Self::columns) order) may hold a
    /// NULL in a file of this version, as the table's metadata declares it
    /// with nothing read: `false` for a column it lets no file hold a NULL
    /// in.
    fn declared_nullability(&self) -> Vec<bool>;

    /// What a scan over `filters`, the predicates pushed into it, reads:
    /// the row groups of the files this version's metadata cannot rule out,
    /// in an order fixed by the version, so the same filters yield the same
    /// row-group positions for scanning and materialization.
    fn select_scan(&self, filters: &[Expression]) -> CatalogResult<ScanSelection>;

    /// The rows a scan of the whole table returns, when this version's
    /// metadata counts them exactly; `None` when only a scan can.
    fn count_rows(&self) -> Option<i64>;

    /// The size of the whole table as the cost model takes it; `None` when
    /// the metadata gives none.
    fn estimate_row_count(&self) -> Option<u64>;

    /// The dataflow writing the rows of `input` into the table (see
    /// [`BoundTable::compile_insert`]).
    fn compile_insert(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> CatalogResult<RecordBatchOperatorSpec>;

    /// The dataflow deleting the rows of the table every one of
    /// `conditions` is true of (see [`BoundTable::compile_delete`]).
    fn compile_delete(
        &self,
        conditions: &[Expression],
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec>;
}

/// The dataflow of a statement that changes a table with no row read or
/// written: it runs `stage`, which stages the change for the transaction's
/// commit, once as it starts, and emits one `BIGINT` row holding
/// `affected_rows`, the rows the change removes.
pub fn build_affected_rows_spec(
    dispatcher: &DataFlowDispatcher,
    affected_rows: u64,
    stage: Box<dyn FnOnce() + Send>,
) -> CatalogResult<RecordBatchOperatorSpec> {
    let affected_rows = i64::try_from(affected_rows)
        .map_err(|error| planner::catalog::Error::Other(Box::new(error)))?;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "Count",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![affected_rows]))],
    )
    .map_err(|error| planner::catalog::Error::Other(Box::new(error)))?;
    // One nullary per worker, of which only the first stages the change and
    // emits the row.
    let mut statement = Some((batch, stage));
    let factories: Vec<_> = (0..dispatcher.worker_count())
        .map(|_| {
            let carried = statement.take();
            OneShotNullaryFactory::new(move || {
                let (batch, stage) = carried?;
                stage();
                Some(batch)
            })
        })
        .collect();
    Ok(RecordBatchOperatorSpec::from_nullary(dispatcher, factories))
}

/// One reference to a table in a query: the version of the table the
/// query's transaction resolved, and the filters pushed into this
/// reference. Cloned per reference, so each accumulates its own filters.
#[derive(Clone)]
pub struct ParquetTableBinding {
    /// The qualified name this binding was resolved under, as the catalog
    /// routed it.
    reference: TableReference,
    /// The table as its datastore scans, counts and writes it, shared by
    /// the clones of this binding.
    table: Arc<dyn ParquetBackedTable>,
    pub columns: Vec<Column>,
    /// The filters pushed down for this binding. The table selects files
    /// by them whole; the single-column comparisons among them prune row
    /// groups and feed the scan's equality checks.
    filters: Vec<Expression>,
    /// What a scan of this binding reads, once something asked for it:
    /// nullability, statistics, scanning and late materialization, the last
    /// compiled from a clone of this binding, reuse the one load. A plan
    /// kept for a later execution keeps the footers it reads here.
    cached_scan_selection: Arc<Mutex<Option<Arc<ScanSelection>>>>,
}

impl std::fmt::Debug for ParquetTableBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetTableBinding")
            .field("reference", &self.reference)
            .field("table", &self.table)
            .field("filters", &self.filters)
            .finish_non_exhaustive()
    }
}

impl ParquetTableBinding {
    /// A binding over `table` with no filters pushed yet. Nothing of the
    /// table is read.
    pub fn new(reference: TableReference, table: impl ParquetBackedTable + 'static) -> Self {
        let columns = table.columns();
        Self {
            reference,
            table: Arc::new(table),
            columns,
            filters: Vec::new(),
            cached_scan_selection: Arc::default(),
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
    /// their own statistics rule out. Selected by whatever asks first, under
    /// a lock the clones share, and kept on a successful load, so a scan
    /// and its late materialize address the same global row-group indices
    /// and no footer is read twice.
    fn load_scan_selection(&self) -> CatalogResult<Arc<ScanSelection>> {
        let mut cached = self.cached_scan_selection.lock().unwrap();
        if let Some(selection) = cached.as_ref() {
            return Ok(Arc::clone(selection));
        }
        let selected = self.table.select_scan(&self.filters)?;
        let selection = Arc::new(ScanSelection {
            row_groups: Arc::new(self.pruned_parquet(&selected.row_groups)),
            nullability: selected.nullability,
        });
        *cached = Some(Arc::clone(&selection));
        Ok(selection)
    }

    /// The row groups a scan of this binding reads.
    fn load_pruned_parquet(&self) -> CatalogResult<Arc<ParquetTable>> {
        Ok(Arc::clone(&self.load_scan_selection()?.row_groups))
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

impl BoundTable for ParquetTableBinding {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.table.revision()
    }

    fn supports_late_materialization(&self) -> bool {
        true
    }

    /// Every table kept in Parquet files applies them: the reader resolves a
    /// pushed path against each row group's own layout, so a file reads the
    /// leaves the path needs whichever fields it shredded, and a file
    /// without the column yields no value for it. An extract is applied to
    /// every decoded row, those the table format deleted included, before
    /// they are dropped, so a deleted row whose value cannot be cast fails
    /// the scan.
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

    /// The table is handed the statement's own conditions: the filters
    /// pushed into this binding only narrow a scan, and need not be all of
    /// them.
    fn compile_delete(
        &self,
        conditions: &[Expression],
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        self.table.compile_delete(conditions, dispatcher)
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    /// The columns the table's metadata declares free of NULLs are answered
    /// with nothing read; the others for the files the pushed filters keep.
    /// A scan that cannot be selected may hold a NULL in any of those;
    /// compiling it reports what is wrong.
    fn nullability(&self) -> Vec<bool> {
        let declared_nullability = self.table.declared_nullability();
        if declared_nullability.iter().all(|is_nullable| !is_nullable) {
            return declared_nullability;
        }
        match self.load_scan_selection() {
            Ok(selection) => selection.nullability.clone(),
            Err(_) => declared_nullability,
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
        // Just record it: pruning happens when the scan is selected. The
        // upstream `Filter` is kept (`Ok(false)`), so pruning only excludes
        // work and never decides which rows the query returns.
        self.filters.push(filter);
        // What was selected for fewer filters is not this binding's view; it
        // stays the view of the clones made before this filter.
        self.cached_scan_selection = Arc::default();

        Ok(false)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down filter
        // means the scan this binding stands for excludes rows.
        if !self.is_unfiltered() {
            return None;
        }
        // Exact extrema come from the row groups a scan would read; a view
        // that cannot be selected means "scan", which reports what is wrong.
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
