//! The [`planner::catalog::Catalog`] implementation over a REST catalog.
//! Binding resolves a table's schema; each query's [`QueryContext`] then
//! resolves the table's *current* snapshot once and materializes its Parquet
//! row groups, shared by every scan of that table in the query. Resolving the
//! snapshot per query (not per bind) matters because the server caches plans
//! by SQL text: a reused plan must still see commits made since it was bound,
//! and a self-join's two bindings must read one consistent snapshot.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use catalog::TableFile;
use catalog::parquet::{ParquetTable, load_table_files, materialize, table_input};
use catalog::store::DataFile;
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    Catalog, Column, CreateTableRequest, DynamicScanPredicate, QueryContext,
    Result as CatalogResult, Table,
};
use planner::types::physical_arrow_type;

use crate::client::RestClient;
use crate::{Error, Result, manifest, warehouse::WarehouseReader};

/// How to reach the REST catalog. `uri` is the endpoint root (e.g.
/// `http://localhost:8181`); the rest is optional.
#[derive(Debug, Clone)]
pub struct IcebergRestConfig {
    pub uri: String,
    /// Warehouse identifier passed to the config endpoint, for servers that
    /// host several warehouses.
    pub warehouse: Option<String>,
    /// Bearer token sent on every catalog request.
    pub token: Option<String>,
    /// The namespace an unqualified table name resolves in. A dotted name
    /// (`ns.table`, `a.b.table`) names its namespace explicitly - note SQL
    /// must quote such a name as one identifier (`FROM "ns.table"`), since
    /// the planner's binder exposes a single schema.
    pub default_namespace: String,
}

impl IcebergRestConfig {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            warehouse: None,
            token: None,
            default_namespace: "default".to_string(),
        }
    }
}

/// A read-only [`Catalog`] over an existing Iceberg REST catalog. Resolving a
/// table loads its current metadata from the catalog service; scanning reads
/// its snapshot's Parquet files directly from the warehouse's object store,
/// through the engine's usual range-read path.
pub struct IcebergRestCatalog {
    client: Arc<RestClient>,
    reader: Arc<WarehouseReader>,
    default_namespace: Vec<String>,
    /// The pool footer loads run on, captured here because query contexts are
    /// created without one.
    dispatcher: DataFlowDispatcher,
}

impl std::fmt::Debug for IcebergRestCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergRestCatalog")
            .field("client", &self.client)
            .field("default_namespace", &self.default_namespace)
            .finish_non_exhaustive()
    }
}

impl IcebergRestCatalog {
    /// Connect to the catalog service (fetching `/v1/config`), failing fast on
    /// an unreachable or misconfigured endpoint.
    pub fn connect(config: IcebergRestConfig, dispatcher: DataFlowDispatcher) -> Result<Self> {
        let client = RestClient::connect(
            &config.uri,
            config.warehouse.as_deref(),
            config.token.clone(),
        )?;
        Ok(Self {
            client: Arc::new(client),
            reader: Arc::new(WarehouseReader::default()),
            default_namespace: split_name(&config.default_namespace),
            dispatcher,
        })
    }

    /// Resolve `name` to a table binding carrying its schema, `Ok(None)` when
    /// the catalog has no such table. The snapshot to scan is resolved later,
    /// per query, through the [`QueryContext`].
    fn resolve(&self, name: &str) -> Result<Option<IcebergTable>> {
        let mut parts = split_name(name);
        let table_name = parts.pop().expect("split_name yields at least one part");
        let namespace = if parts.is_empty() {
            self.default_namespace.clone()
        } else {
            parts
        };

        let Some(loaded) = self.client.load_table(&namespace, &table_name)? else {
            return Ok(None);
        };
        Ok(Some(IcebergTable {
            name: name.to_string(),
            columns: loaded.metadata.map_columns(name)?,
            namespace,
            table_name,
        }))
    }
}

/// Split a (table or namespace) name on `.` into its parts: `a.b.c` names
/// table `c` in namespace `[a, b]`. Always yields at least one part.
fn split_name(name: &str) -> Vec<String> {
    name.split('.').map(str::to_string).collect()
}

impl Catalog for IcebergRestCatalog {
    /// Resolve `name` against the REST catalog. Binding fetches the table's
    /// current schema; the snapshot to scan is re-resolved by every query's
    /// context, so even a cached plan reads the latest committed snapshot. A
    /// resolve *failure* (unreachable catalog, unsupported schema) is logged
    /// and binds as "no such table" - the trait offers no error channel here.
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        match self.resolve(name) {
            Ok(table) => table.map(|table| Box::new(table) as Box<dyn Table>),
            Err(e) => {
                tracing::error!(table = name, "failed to resolve iceberg table: {e}");
                None
            }
        }
    }

    fn query_context(&self) -> Box<dyn QueryContext> {
        Box::new(IcebergQueryContext {
            client: self.client.clone(),
            reader: self.reader.clone(),
            dispatcher: self.dispatcher.clone(),
            pinned: Mutex::new(HashMap::new()),
        })
    }

    fn create_table(
        &self,
        _request: CreateTableRequest,
        _dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Err(Error::CreateTableUnsupported.into())
    }
}

/// One query's context: the first ask for a table resolves its **current**
/// snapshot from the REST catalog and materializes it into row groups, pinned
/// (keyed by the binding's name) for the rest of the query - so a reused
/// (cached) plan sees data committed since it was bound, and the scan, a late
/// materialize, and a self-join's second binding all read one consistent
/// snapshot and agree on row-group indices.
struct IcebergQueryContext {
    client: Arc<RestClient>,
    reader: Arc<WarehouseReader>,
    dispatcher: DataFlowDispatcher,
    pinned: Mutex<HashMap<String, Arc<ParquetTable>>>,
}

impl QueryContext for IcebergQueryContext {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl IcebergQueryContext {
    /// `table`'s scan view for this query: resolve its current snapshot, walk
    /// the manifest list to the live data files, fetch their footers over the
    /// worker pool, and pin the result. A table dropped since planning is an
    /// error, never a silent empty scan.
    fn pin_scan_view(&self, table: &IcebergTable) -> Result<Arc<ParquetTable>> {
        if let Some(parquet) = self.pinned.lock().unwrap().get(&table.name) {
            return Ok(parquet.clone());
        }

        let loaded = self
            .client
            .load_table(&table.namespace, &table.table_name)?
            .ok_or_else(|| {
                Error::Metadata(format!(
                    "table `{}` no longer exists (dropped since planning?)",
                    table.name
                ))
            })?;
        let parquet = match loaded.metadata.find_current_manifest_list()? {
            Some(manifest_list) => self.build_scan_view(table, &manifest_list)?,
            // No snapshot yet: the table exists but holds no data.
            None => Arc::new(ParquetTable::new(Vec::new())),
        };
        self.pinned
            .lock()
            .unwrap()
            .insert(table.name.clone(), parquet.clone());
        Ok(parquet)
    }

    /// Materialize the snapshot behind `manifest_list` into a scannable
    /// [`ParquetTable`], validating every file against the binding's schema.
    fn build_scan_view(
        &self,
        table: &IcebergTable,
        manifest_list: &str,
    ) -> Result<Arc<ParquetTable>> {
        let manifests = manifest::parse_manifest_list(&self.reader.fetch(manifest_list)?)?;
        if let Some(deletes) = manifests.iter().find(|entry| entry.content != 0) {
            return Err(Error::Unsupported(format!(
                "`{}` is a delete manifest; iceberg row-level deletes are not supported",
                deletes.path
            )));
        }
        let data_files = self.fetch_data_files(&manifests)?;

        let table_files: Vec<TableFile> = load_table_files(&self.dispatcher, &data_files)?;
        for file in &table_files {
            validate_file_schema(table, file)?;
        }
        let row_groups = table_files
            .iter()
            .flat_map(|file| file.row_groups().iter().cloned())
            .collect();
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// Fetch and parse every (data) manifest, locating each live data file for
    /// reading. The manifests are independent object-store GETs, so they are
    /// fanned out over a few scoped threads instead of paying one blocking
    /// round-trip per manifest; results keep manifest-list order.
    fn fetch_data_files(&self, manifests: &[manifest::ManifestFile]) -> Result<Vec<DataFile>> {
        const FETCH_THREADS: usize = 8;

        let chunk_size = manifests.len().div_ceil(FETCH_THREADS).max(1);
        let chunk_results: Vec<Result<Vec<DataFile>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = manifests
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut files = Vec::new();
                        for entry in chunk {
                            for file in manifest::parse_manifest(&self.reader.fetch(&entry.path)?)?
                            {
                                files.push(self.reader.locate_data_file(
                                    &file.file_path,
                                    file.file_size_in_bytes as u64,
                                )?);
                            }
                        }
                        Ok(files)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("manifest fetch thread panicked"))
                .collect()
        });

        let mut data_files = Vec::new();
        for chunk in chunk_results {
            data_files.extend(chunk?);
        }
        Ok(data_files)
    }
}

/// Check that `file`'s physical columns line up with the binding's schema:
/// same count, and each position carries the expected name and arrow type. The
/// scan resolves columns **positionally**, so any drift - a column added,
/// dropped, renamed, reordered, or type-promoted after this file was written
/// (iceberg schema evolution, which resolves by field id) - would silently
/// read the wrong data; fail the query instead.
fn validate_file_schema(table: &IcebergTable, file: &TableFile) -> Result<()> {
    let mismatch = |detail: String| Error::DataFileSchemaMismatch {
        table: table.name.clone(),
        detail,
    };
    for row_group in file.row_groups() {
        let fields = row_group.schema.fields();
        if fields.len() != table.columns.len() {
            return Err(mismatch(format!(
                "the table has {} columns but a data file has {}",
                table.columns.len(),
                fields.len()
            )));
        }
        for (field, column) in fields.iter().zip(&table.columns) {
            let expected = physical_arrow_type(&column.col_type);
            if field.name() != &column.name || field.data_type() != &expected {
                return Err(mismatch(format!(
                    "expected column `{}` ({expected}), a data file has `{}` ({})",
                    column.name,
                    field.name(),
                    field.data_type()
                )));
            }
        }
    }
    Ok(())
}

/// One table binding: the name it was resolved as plus its schema. The
/// snapshot to scan is *not* part of the binding - it is resolved per query by
/// [`IcebergQueryContext::pin_scan_view`] at [`compile`](Table::compile) time,
/// so a cached plan always scans the latest committed snapshot and binding
/// stays cheap.
#[derive(Clone, Debug)]
struct IcebergTable {
    /// The name this binding was resolved as, e.g. `db.events` - the query
    /// context's pin key.
    name: String,
    columns: Vec<Column>,
    namespace: Vec<String>,
    table_name: String,
}

impl IcebergTable {
    /// This query's pinned row groups for the table, materialized through the
    /// query context. The context is always our own - an `IcebergRestCatalog`
    /// only compiles its own tables - so a foreign context is an error, never
    /// an empty scan.
    fn resolve_scan_view(&self, ctx: &dyn QueryContext) -> CatalogResult<Arc<ParquetTable>> {
        let ctx = ctx
            .as_any()
            .downcast_ref::<IcebergQueryContext>()
            .ok_or_else(|| {
                planner::catalog::Error::Other(
                    format!(
                        "query context for iceberg table `{}` is not an IcebergQueryContext",
                        self.name
                    )
                    .into(),
                )
            })?;
        Ok(ctx.pin_scan_view(self)?)
    }
}

impl Table for IcebergTable {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        _dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
        ctx: &dyn QueryContext,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Dynamic filters are a pure scan optimization; ignoring them is
        // always correct, just without row-group skipping.
        let parquet = self.resolve_scan_view(ctx)?;
        Ok(table_input(
            dispatcher,
            &parquet,
            projection,
            emit_row_group_metadata,
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
        // Same context as the scan, so both read the pinned snapshot and their
        // global row-group indices line up.
        Ok(materialize(input, self.resolve_scan_view(ctx)?, projection))
    }

    fn row_count(&self, ctx: &dyn QueryContext) -> Option<i64> {
        // No predicates are ever pushed into this table (pushdown_filter keeps
        // the default), so the whole pinned snapshot is what a scan reads and
        // its footer row counts answer exactly.
        let parquet = self.resolve_scan_view(ctx).ok()?;
        Some(parquet.row_groups().iter().map(|rg| rg.num_rows).sum())
    }
}
