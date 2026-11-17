//! The binding of one query to a catalog table kept in Parquet files: the
//! filters pushed into it, and what a scan of the table reads once they are
//! applied, selected once and shared by everything the query asks of the
//! binding.
//!
//! A datastore says how its table is scanned, counted and written
//! ([`ParquetBackedTable`]); the binding ([`ParquetTableBinding`]) is the same
//! for all of them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Scalar};
use dispatch::io::OpenFile;
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use object_storage::ObjectPath;
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Result as CatalogResult, TableReference,
    TableRevision,
};
use planner::expression::{Expression, Function};

use crate::{
    FileToSearch, OpenTextIndexFile, ParquetTable, PushedPredicate, TokenToFind,
    equality_predicates, materialize, prune_parquet, row_group_filter_from, scan_order_from,
    table_input_searching_text_indexes, table_input_with_filter_and_eq_predicates,
};

/// What a scan over some filters reads of a table version.
pub struct ScanSelection {
    /// The row groups the scan reads, in an order the version and the
    /// filters fix.
    pub row_groups: Arc<ParquetTable>,
    /// The path of each data file the scan reads, as its table names the
    /// file, by the file as its row groups hold it open. A table finds a
    /// file's text indexes by this path.
    pub data_file_paths: HashMap<OpenFile, ObjectPath>,
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

    /// For each of `data_files` (paths as a [`ScanSelection`] gives them),
    /// in their order, the index file that says which of the file's rows
    /// have a token in `column` (a position in [`columns`](Self::columns)),
    /// opened for reading: `None` for a file no text index of the column
    /// has an index file of. A table without text indexes has none for any
    /// file.
    fn open_text_index_files(
        &self,
        _column: usize,
        data_files: &[ObjectPath],
    ) -> CatalogResult<Vec<Option<OpenTextIndexFile>>> {
        Ok(vec![None; data_files.len()])
    }
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
            data_file_paths: selected.data_file_paths,
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

    /// The files of the scan with what their text indexes are asked: for
    /// every `has_token` of a column among the pushed filters, the token and
    /// the index file of the column in each file that has one. `None` when
    /// the filters look for no token, or no file of the scan has an index to
    /// find one in: the scan then reads its row groups as any other does.
    fn plan_text_index_search(
        &self,
        selection: &ScanSelection,
    ) -> CatalogResult<Option<Vec<FileToSearch>>> {
        let tokens_looked_for: Vec<(usize, &str)> = self
            .filters
            .iter()
            .filter_map(|filter| {
                let Expression::Function(Function::HasToken(has_token)) = filter else {
                    return None;
                };
                let Expression::Ref(column) = has_token.text.as_ref() else {
                    return None;
                };
                Some((column.column_idx, has_token.token.as_str()))
            })
            .collect();
        if tokens_looked_for.is_empty() {
            return Ok(None);
        }

        // The scan's files in the order its row groups are in, which holds
        // a file's row groups one after the other.
        let mut files: Vec<(&OpenFile, FileToSearch)> = Vec::new();
        for (row_group_index, row_group) in selection.row_groups.row_groups().iter().enumerate() {
            match files.last_mut() {
                Some((open_file, file)) if **open_file == row_group.open_file => {
                    file.row_groups.push(row_group_index);
                }
                _ => files.push((
                    &row_group.open_file,
                    FileToSearch {
                        row_groups: vec![row_group_index],
                        tokens: Vec::new(),
                    },
                )),
            }
        }
        // A file the selection gives no path of is one no index is found
        // for.
        let (positions_of_named_files, paths): (Vec<usize>, Vec<ObjectPath>) = files
            .iter()
            .enumerate()
            .filter_map(|(position, (open_file, _))| {
                Some((position, selection.data_file_paths.get(open_file)?.clone()))
            })
            .unzip();
        for (column, token) in tokens_looked_for {
            let index_files = self.table.open_text_index_files(column, &paths)?;
            for (&position, index_file) in positions_of_named_files.iter().zip(index_files) {
                if let Some(index_file) = index_file {
                    files[position]
                        .1
                        .tokens
                        .push(TokenToFind::new(token.as_bytes(), index_file));
                }
            }
        }
        let files: Vec<FileToSearch> = files.into_iter().map(|(_, file)| file).collect();
        if files.iter().all(|file| file.tokens.is_empty()) {
            return Ok(None);
        }
        Ok(Some(files))
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
        let selection = self.load_scan_selection()?;
        let parquet = Arc::clone(&selection.row_groups);

        // A scan that looks for tokens searches its files' text indexes
        // first and reads only the rows found, unless a Top-N above it is
        // tightening a boundary: such a scan reads its row groups in the
        // order of the Top-N's key and stops at the boundary, and a search
        // of every file ahead of that would read indexes of files it never
        // gets to. The filters stay above the scan either way, so the
        // search only decides what is read, never what is returned.
        if dynamic_filters.is_empty()
            && !projection.indices().is_empty()
            && let Some(files) = self.plan_text_index_search(&selection)?
        {
            return Ok(table_input_searching_text_indexes(
                dispatcher,
                &parquet,
                files,
                projection,
                emit_row_group_metadata,
            ));
        }

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
