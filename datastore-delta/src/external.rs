//! Reading Parquet files a query names directly — `read_parquet('s3://…')` —
//! rather than tables the catalog holds.
//!
//! What comes back is an ordinary [`BoundTable`]: the same scan pipeline, the
//! same row-group pruning, the same late materialization a catalog table's
//! binding gets, over row groups read from a location instead of from a Delta
//! snapshot. Nothing here is written to, and nothing is remembered between
//! queries: the location is listed and its footers are read each time one binds,
//! so a file added since the last query is part of the next one.
//!
//! The schema is the **first matched file's**, by name. Every other file must
//! agree with it, column for column and type for type: a location whose files
//! disagree is an error naming the file that broke, not a scan that silently
//! reads a column's rows from one file and NULLs from another.

use std::sync::Arc;

use arrow_array::{ArrayRef, Scalar};
use arrow_schema::Schema;
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Error as CatalogError, Result as CatalogResult,
    TableReference, TableRevision,
};
use planner::expression::TableFilter;
use planner::types::type_from_physical;

use crate::parquet::load_file_row_groups;
use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::{
    ParquetTable, materialize, row_group_filter_from, scan_order_from,
    table_input_with_filter_and_eq_predicates,
};
use crate::pushdown::{
    self, PushedPredicate, column_min_max, equality_predicates, location_revision, prune_row_groups,
};
use crate::store::{DataFile, ObjectStore, StoreError};

/// A location that cannot be read as a table.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no file matches `{location}`")]
    NoFilesMatch { location: String },
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("cannot read the footers of `{location}`: {source}")]
    Footers {
        location: String,
        #[source]
        source: dispatch::DataFlowError,
    },
    #[error(
        "column `{column}` of `{file}` has type `{data_type}`, which pivot cannot read from Parquet"
    )]
    UnsupportedColumnType {
        file: String,
        column: String,
        data_type: String,
    },
    #[error(
        "`{file}` does not have the columns `{first_file}` has, so `{location}` matches files of more than one schema: `{first_file}` has [{expected}], `{file}` has [{found}]"
    )]
    SchemaMismatch {
        location: String,
        first_file: String,
        file: String,
        expected: String,
        found: String,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The files `pattern` matches directly under `store`'s root, located for
/// reading. `location` is the location as written, for the error a pattern
/// matching nothing raises: a read of no files at all is a mistyped path far
/// more often than an intentionally empty scan.
pub fn list_external_parquet_files(
    store: &dyn ObjectStore,
    pattern: &str,
    location: &str,
) -> Result<Vec<DataFile>> {
    let matched = crate::store::list_matching(store, pattern)?;
    if matched.is_empty() {
        return Err(Error::NoFilesMatch {
            location: location.to_string(),
        });
    }
    matched
        .into_iter()
        .map(|file| {
            let source = store.source(&file.path)?;
            Ok(DataFile { file, source })
        })
        .collect()
}

/// Read `files`' footers over the worker pool and bind them as one table.
///
/// Drives the metadata-fetch dataflow, so it runs on the **coordinator** (the
/// thread holding `dispatcher`), which is where a query binds.
pub fn bind_external_parquet(
    dispatcher: &DataFlowDispatcher,
    location: String,
    files: Vec<DataFile>,
) -> Result<ExternalParquetBinding> {
    // Nothing is declared about these files, so each footer is read as it
    // stands and the first file's schema becomes the table's below.
    let mut loaded =
        load_file_row_groups(dispatcher, &files, Arc::from(Vec::new())).map_err(|source| {
            Error::Footers {
                location: location.clone(),
                source,
            }
        })?;
    // The workers finish in whatever order they steal, so order by name: which
    // file's schema the table takes, and the order the scan reads them in, are
    // then the same on every run.
    loaded.sort_by(|a, b| a.file.path.as_str().cmp(b.file.path.as_str()));

    // A file with no row groups carries no schema to reconcile or to read.
    let mut with_row_groups = loaded
        .iter()
        .filter(|file| !file.row_groups.is_empty())
        .peekable();
    let Some((first_file, schema)) = with_row_groups.peek().map(|file| {
        (
            file.file.path.to_string(),
            file.row_groups[0].schema.clone(),
        )
    }) else {
        return Err(Error::NoFilesMatch { location });
    };
    for file in with_row_groups {
        check_same_columns(
            &schema,
            &first_file,
            &file.row_groups[0].schema,
            &file.file.path.to_string(),
            &location,
        )?;
    }

    let row_groups: Vec<Arc<RowGroupMetadata>> = loaded
        .iter()
        .flat_map(|file| file.row_groups.iter().cloned())
        .collect();
    let columns = columns_of(&schema, &first_file)?;
    let nullability = columns
        .iter()
        .map(|column| column_may_hold_nulls(&row_groups, &column.name))
        .collect();

    Ok(ExternalParquetBinding {
        location,
        parquet: Arc::new(ParquetTable::new(row_groups)),
        columns,
        nullability,
        predicates: Vec::new(),
    })
}

/// The table columns `schema` describes, refusing a Parquet type pivot has no
/// SQL type for rather than binding a column no expression could read.
fn columns_of(schema: &Schema, file: &str) -> Result<Vec<Column>> {
    schema
        .fields()
        .iter()
        .map(|field| {
            let col_type = type_from_physical(field.data_type()).ok_or_else(|| {
                Error::UnsupportedColumnType {
                    file: file.to_string(),
                    column: field.name().clone(),
                    data_type: field.data_type().to_string(),
                }
            })?;
            Ok(Column {
                name: field.name().clone(),
                col_type,
            })
        })
        .collect()
}

/// Refuse `schema` unless it carries the same columns, in the same order and of
/// the same types, as the schema the table bound. Nullability is deliberately
/// not compared: a column marked `REQUIRED` in one file and `OPTIONAL` in
/// another is one column that may hold NULLs, which
/// [`column_may_hold_nulls`] already answers by reading every row group.
fn check_same_columns(
    expected: &Schema,
    first_file: &str,
    found: &Schema,
    file: &str,
    location: &str,
) -> Result<()> {
    let describe = |schema: &Schema| {
        schema
            .fields()
            .iter()
            .map(|field| format!("{} {}", field.name(), field.data_type()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let same = expected.fields().len() == found.fields().len()
        && expected
            .fields()
            .iter()
            .zip(found.fields())
            .all(|(a, b)| a.name() == b.name() && a.data_type() == b.data_type());
    if same {
        return Ok(());
    }
    Err(Error::SchemaMismatch {
        location: location.to_string(),
        first_file: first_file.to_string(),
        file: file.to_string(),
        expected: describe(expected),
        found: describe(found),
    })
}

/// Whether any row group can hold a NULL in `name`. A column that is `REQUIRED`
/// everywhere cannot; an `OPTIONAL` one (writers mark every column `OPTIONAL`
/// even when no value is ever NULL) is refined by the chunk's `null_count`
/// statistic when the schema is flat enough to map fields to leaves.
fn column_may_hold_nulls(row_groups: &[Arc<RowGroupMetadata>], name: &str) -> bool {
    row_groups.iter().any(|rg| {
        let Ok(field_idx) = rg.schema.index_of(name) else {
            return true;
        };
        if !rg.schema.field(field_idx).is_nullable() {
            return false;
        }
        // Nested fields (e.g. a shredded variant) span several leaves, so the
        // field-to-chunk mapping below does not hold; stay conservative for the
        // whole row group.
        if rg.columns.len() != rg.schema.fields().len() {
            return true;
        }
        rg.columns[field_idx]
            .statistics
            .as_ref()
            .and_then(|stats| stats.null_count)
            .is_none_or(|null_count| null_count > 0)
    })
}

/// One query's reading of a location: the row groups of every file it matched,
/// read once when the query bound, plus the predicates DuckDB pushed into this
/// scan. Independently mutable per query, exactly as a catalog table's binding
/// is, so one query's pushdown never narrows another's.
#[derive(Clone)]
pub struct ExternalParquetBinding {
    /// The location as the query wrote it: what the scan is named by, and what
    /// an error about these files points at.
    location: String,
    parquet: Arc<ParquetTable>,
    columns: Vec<Column>,
    nullability: Vec<bool>,
    predicates: Vec<PushedPredicate>,
}

impl std::fmt::Debug for ExternalParquetBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalParquetBinding")
            .field("location", &self.location)
            .field("predicates", &self.predicates)
            .finish_non_exhaustive()
    }
}

impl ExternalParquetBinding {
    /// The row groups this binding's pushed-down predicates leave: what a scan
    /// reads. A late materialize must build the identical view, since a row
    /// reference is a position within it.
    fn pruned(&self) -> Arc<ParquetTable> {
        Arc::new(prune_row_groups(&self.parquet, &self.predicates))
    }
}

impl BoundTable for ExternalParquetBinding {
    fn table_reference(&self) -> TableReference {
        // A location belongs to no datastore and no schema; the reference
        // exists so a plan can name what it scans.
        TableReference {
            datastore: String::new(),
            schema: String::new(),
            table: self.location.clone(),
        }
    }

    fn table_revision(&self) -> TableRevision {
        location_revision(&self.location)
    }

    /// The files this scan read were listed while it was planned. Keeping the
    /// plan would keep that listing, so the next query would read the files this
    /// one found rather than the ones there now.
    fn is_plan_cacheable(&self) -> bool {
        false
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
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &self.pruned(),
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(equality_predicates(&self.predicates)),
        ))
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // A row reference is a position in the *scanning* view's flat row-group
        // list, so this has to build the identical view — same files, same
        // pruning — or every index past the first dropped row group shifts.
        Ok(materialize(input, self.pruned(), projection))
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

    fn pushdown_filter(&mut self, filter: TableFilter) -> CatalogResult<bool> {
        pushdown::record_pushed_filter(filter, &mut self.predicates)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered read: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        column_min_max(&self.parquet, column)
    }

    fn row_count(&self) -> Option<i64> {
        if !self.predicates.is_empty() {
            return None;
        }
        // Every footer carries its row groups' exact row counts, so the total is
        // their sum with no data pages read.
        Some(self.parquet.row_groups().iter().map(|rg| rg.num_rows).sum())
    }

    fn estimate_row_count(&self) -> Option<u64> {
        // The cost model wants the base size and applies filter selectivity
        // itself, so this ignores the pushed predicates.
        Some(
            self.parquet
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows as u64)
                .sum(),
        )
    }
}

impl From<Error> for CatalogError {
    fn from(error: Error) -> Self {
        CatalogError::Other(Box::new(error))
    }
}
