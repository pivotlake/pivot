//! The [`planner::catalog::Catalog`] implementation over a REST catalog.
//! Binding resolves a table's schema; each query's [`QueryContext`] then
//! resolves the table's *current* snapshot once - via iceberg-rust's
//! `plan_files` scan planning - and materializes its Parquet row groups,
//! shared by every scan of that table in the query. Resolving the snapshot per
//! query (not per bind) matters because the server caches plans by SQL text: a
//! reused plan must still see commits made since it was bound, and a
//! self-join's two bindings must read one consistent snapshot.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use catalog::TableFile;
use catalog::parquet::{ParquetTable, load_table_files, materialize, table_input};
use catalog::store::DataFile;
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use futures::TryStreamExt;
use iceberg::Catalog as IcebergCatalog;
use iceberg::CatalogBuilder as _;
use iceberg::scan::FileScanTask;
use iceberg::spec::DataFileFormat;
use iceberg::{NamespaceIdent, TableIdent};
use iceberg_catalog_rest::{
    REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalog, RestCatalogBuilder,
};
use planner::catalog::{
    Catalog, Column, CreateTableRequest, DynamicScanPredicate, QueryContext,
    Result as CatalogResult, Table,
};
use planner::types::physical_arrow_type;

use crate::storage::PivotStorageFactory;
use crate::warehouse::WarehouseReader;
use crate::{Error, Result, schema};

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
    /// Extra catalog properties passed through to the REST client verbatim
    /// (e.g. OAuth settings). Storage credentials are *not* configured here:
    /// warehouse reads go through `catalog::store`, which resolves them from
    /// the environment like the rest of pivot.
    pub props: HashMap<String, String>,
}

impl IcebergRestConfig {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            warehouse: None,
            token: None,
            default_namespace: "default".to_string(),
            props: HashMap::new(),
        }
    }
}

/// The async iceberg client plus the small runtime that drives it, shared by
/// the catalog and its query contexts. All calls are `block_on` at the two
/// control-plane entry points (bind, per-query snapshot pin); the scan data
/// path never touches it.
struct RestHandle {
    runtime: tokio::runtime::Runtime,
    catalog: RestCatalog,
}

impl RestHandle {
    /// Load `ident`'s current table state, `Ok(None)` when the catalog has no
    /// such table (or namespace).
    fn load_table(&self, ident: &TableIdent) -> Result<Option<iceberg::table::Table>> {
        match self.runtime.block_on(self.catalog.load_table(ident)) {
            Ok(table) => Ok(Some(table)),
            Err(e)
                if matches!(
                    e.kind(),
                    iceberg::ErrorKind::TableNotFound | iceberg::ErrorKind::NamespaceNotFound
                ) =>
            {
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Plan `table`'s current snapshot: the live data files, resolved by the
    /// upstream implementation (field ids, delete files, format versions). A
    /// table with no snapshot yet plans to an empty list.
    fn plan_files(&self, table: &iceberg::table::Table) -> Result<Vec<FileScanTask>> {
        self.runtime.block_on(async {
            let scan = table.scan().select_all().build()?;
            Ok(scan.plan_files().await?.try_collect().await?)
        })
    }
}

/// A read-only [`Catalog`] over an existing Iceberg REST catalog. Resolving a
/// table loads its current metadata from the catalog service; scanning reads
/// its snapshot's Parquet files directly from the warehouse's object store,
/// through the engine's usual range-read path.
pub struct IcebergRestCatalog {
    handle: Arc<RestHandle>,
    reader: Arc<WarehouseReader>,
    default_namespace: Vec<String>,
    /// The pool footer loads run on, captured here because query contexts are
    /// created without one.
    dispatcher: DataFlowDispatcher,
}

impl std::fmt::Debug for IcebergRestCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergRestCatalog")
            .field("default_namespace", &self.default_namespace)
            .finish_non_exhaustive()
    }
}

impl IcebergRestCatalog {
    /// Connect to the catalog service, failing fast on an unreachable or
    /// misconfigured endpoint (the client is lazy, so reachability is probed
    /// with a namespace listing).
    pub fn connect(config: IcebergRestConfig, dispatcher: DataFlowDispatcher) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(Error::Runtime)?;

        let mut props = config.props;
        props.insert(REST_CATALOG_PROP_URI.to_string(), config.uri);
        if let Some(warehouse) = config.warehouse {
            props.insert(REST_CATALOG_PROP_WAREHOUSE.to_string(), warehouse);
        }
        if let Some(token) = config.token {
            props.insert("token".to_string(), token);
        }
        let catalog = runtime.block_on(
            RestCatalogBuilder::default()
                .with_storage_factory(Arc::new(PivotStorageFactory))
                .load("pivot", props),
        )?;
        runtime.block_on(catalog.list_namespaces(None))?;

        Ok(Self {
            handle: Arc::new(RestHandle { runtime, catalog }),
            reader: Arc::new(WarehouseReader::default()),
            default_namespace: split_name(&config.default_namespace),
            dispatcher,
        })
    }

    /// Resolve `name` to a table binding carrying its schema, `Ok(None)` when
    /// the catalog has no such table. The snapshot to scan is resolved later,
    /// per query, through the [`QueryContext`].
    fn resolve(&self, name: &str) -> Result<Option<IcebergTable>> {
        let ident = self.parse_ident(name);
        let Some(loaded) = self.handle.load_table(&ident)? else {
            return Ok(None);
        };
        Ok(Some(IcebergTable {
            name: name.to_string(),
            columns: schema::map_columns(loaded.metadata().current_schema(), name)?,
            ident,
        }))
    }

    /// `name` as an iceberg table identity: `a.b.c` names table `c` in
    /// namespace `[a, b]`; an unqualified name lives in the configured default
    /// namespace.
    fn parse_ident(&self, name: &str) -> TableIdent {
        let mut parts = split_name(name);
        let table_name = parts.pop().expect("split_name yields at least one part");
        let namespace = if parts.is_empty() {
            self.default_namespace.clone()
        } else {
            parts
        };
        TableIdent::new(
            NamespaceIdent::from_vec(namespace).expect("parts are non-empty"),
            table_name,
        )
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
            handle: self.handle.clone(),
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
    handle: Arc<RestHandle>,
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
    /// `table`'s scan view for this query: plan its current snapshot's live
    /// data files, fetch their footers over the worker pool, and pin the
    /// result. A table dropped since planning is an error, never a silent
    /// empty scan.
    fn pin_scan_view(&self, table: &IcebergTable) -> Result<Arc<ParquetTable>> {
        if let Some(parquet) = self.pinned.lock().unwrap().get(&table.name) {
            return Ok(parquet.clone());
        }

        let loaded = self.handle.load_table(&table.ident)?.ok_or_else(|| {
            Error::Unsupported(format!(
                "table `{}` no longer exists (dropped since planning?)",
                table.name
            ))
        })?;
        let tasks = self.handle.plan_files(&loaded)?;

        let mut data_files: Vec<DataFile> = Vec::with_capacity(tasks.len());
        for task in &tasks {
            if !task.deletes.is_empty() {
                return Err(Error::Unsupported(format!(
                    "`{}` carries row-level delete files; iceberg deletes are not supported",
                    task.data_file_path
                )));
            }
            if task.data_file_format != DataFileFormat::Parquet {
                return Err(Error::Unsupported(format!(
                    "data file `{}` has format {}; only PARQUET is supported",
                    task.data_file_path, task.data_file_format
                )));
            }
            data_files.push(
                self.reader
                    .locate_data_file(&task.data_file_path, task.file_size_in_bytes)?,
            );
        }

        let table_files: Vec<TableFile> = load_table_files(&self.dispatcher, &data_files)?;
        for file in &table_files {
            validate_file_schema(table, file)?;
        }
        let row_groups = table_files
            .iter()
            .flat_map(|file| file.row_groups().iter().cloned())
            .collect();
        let parquet = Arc::new(ParquetTable::new(row_groups));
        self.pinned
            .lock()
            .unwrap()
            .insert(table.name.clone(), parquet.clone());
        Ok(parquet)
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
    ident: TableIdent,
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
