//! Exposes Iceberg tables through Pivot's datastore and planner interfaces.
//!
//! This module translates the current Iceberg schema and snapshot tasks into
//! a read-only Parquet scan while enforcing the feature set supported in v1.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use catalog::datastore::{
    Datastore, DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata,
    DatastoreTransaction,
};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use futures::TryStreamExt;
use iceberg::spec::{DataFileFormat, PrimitiveType, Type as IcebergType};
use iceberg::table::Table;
use iceberg::{
    Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, Runtime as IcebergRuntime, TableIdent,
};
use iceberg_catalog_rest::{
    REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalog, RestCatalogBuilder,
};
use object_storage::DataFile;
use parquet_engine::{
    DeclaredColumn, ParquetTable, PushedPredicate, ScanEqualityPredicate, materialize,
    prune_parquet, row_group_filter_from, scan_order_from,
    table_input_with_filter_and_eq_predicates,
};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Error as PlannerError, Result as PlannerResult,
    SchemaQualifiedTableName, TableReference, TableRevision,
};
use planner::expression::TableFilter;
use planner::types::Type;

use crate::PivotStorageFactory;
use crate::rest::{LoadedTable, RestTableLoader};

#[derive(Clone)]
pub enum IcebergAuth {
    None,
    Bearer {
        token: String,
    },
    OAuth2ClientCredentials {
        client_id: String,
        client_secret: String,
        scope: Option<String>,
        token_endpoint: Option<String>,
    },
}

impl Debug for IcebergAuth {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => formatter.write_str("None"),
            Self::Bearer { .. } => formatter
                .debug_struct("Bearer")
                .field("token", &"redacted")
                .finish(),
            Self::OAuth2ClientCredentials {
                client_id,
                scope,
                token_endpoint,
                ..
            } => formatter
                .debug_struct("OAuth2ClientCredentials")
                .field("client_id", client_id)
                .field("client_secret", &"redacted")
                .field("scope", scope)
                .field("token_endpoint", token_endpoint)
                .finish(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IcebergConfig {
    pub catalog_uri: String,
    pub warehouse: Option<String>,
    pub auth: IcebergAuth,
}

impl IcebergConfig {
    pub fn new(catalog_uri: impl Into<String>) -> Self {
        Self {
            catalog_uri: catalog_uri.into(),
            warehouse: None,
            auth: IcebergAuth::None,
        }
    }
}

pub struct IcebergDatastore {
    config: IcebergConfig,
    catalog: RestCatalog,
    loader: RestTableLoader,
    runtime: tokio::runtime::Handle,
}

impl Debug for IcebergDatastore {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IcebergDatastore")
            .field("catalog_uri", &self.config.catalog_uri)
            .field("warehouse", &self.config.warehouse)
            .field("auth", &self.config.auth)
            .finish()
    }
}

impl IcebergDatastore {
    pub fn new(config: IcebergConfig, storage: Arc<PivotStorageFactory>) -> PlannerResult<Self> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|error| planner_error(format!("Iceberg requires a Tokio runtime: {error}")))?;
        let iceberg_runtime = IcebergRuntime::try_current().map_err(iceberg_error)?;
        let mut properties = HashMap::new();
        properties.insert(
            REST_CATALOG_PROP_URI.to_string(),
            config.catalog_uri.clone(),
        );
        if let Some(warehouse) = &config.warehouse {
            properties.insert(REST_CATALOG_PROP_WAREHOUSE.to_string(), warehouse.clone());
        }
        match &config.auth {
            IcebergAuth::None => {}
            IcebergAuth::Bearer { token } => {
                properties.insert("token".to_string(), token.clone());
            }
            IcebergAuth::OAuth2ClientCredentials {
                client_id,
                client_secret,
                scope,
                token_endpoint,
            } => {
                properties.insert(
                    "credential".to_string(),
                    format!("{client_id}:{client_secret}"),
                );
                if let Some(scope) = scope {
                    properties.insert("scope".to_string(), scope.clone());
                }
                if let Some(token_endpoint) = token_endpoint {
                    properties.insert("oauth2-server-uri".to_string(), token_endpoint.clone());
                }
            }
        }

        let loader = RestTableLoader::new(&config, storage.clone(), iceberg_runtime.clone());
        let catalog = block_on_runtime(
            &runtime,
            RestCatalogBuilder::default()
                .with_storage_factory(storage.clone())
                .with_runtime(iceberg_runtime)
                .load("pivot", properties),
        )?
        .map_err(iceberg_error)?;

        Ok(Self {
            config,
            catalog,
            loader,
            runtime,
        })
    }

    fn run<T>(
        &self,
        future: impl std::future::Future<Output = iceberg::Result<T>>,
    ) -> PlannerResult<T> {
        block_on_runtime(&self.runtime, future)?.map_err(iceberg_error)
    }

    fn load_table(&self, ident: &TableIdent) -> PlannerResult<Option<LoadedTable>> {
        match block_on_runtime(&self.runtime, self.loader.load_table(ident))? {
            Ok(table) => Ok(Some(table)),
            Err(error) if error.kind() == ErrorKind::TableNotFound => Ok(None),
            Err(error) => Err(iceberg_error(error)),
        }
    }
}

#[async_trait]
impl Datastore for IcebergDatastore {
    fn begin_transaction(self: Arc<Self>) -> Arc<dyn DatastoreTransaction> {
        Arc::new(IcebergTransaction {
            datastore: self,
            tables: Mutex::new(HashMap::new()),
        })
    }

    fn kind(&self) -> &'static str {
        "iceberg"
    }

    fn data_path(&self) -> String {
        self.config.catalog_uri.clone()
    }
}

struct FrozenTable {
    table: Arc<Table>,
    storage: Arc<PivotStorageFactory>,
    columns: Vec<Column>,
    field_ids: Vec<i32>,
    nullability: Vec<bool>,
    tasks: Vec<iceberg::scan::FileScanTask>,
    revision: TableRevision,
    row_count: u64,
    bytes: u64,
}

impl FrozenTable {
    fn load(datastore: &IcebergDatastore, loaded: LoadedTable) -> PlannerResult<Self> {
        let table = loaded.table;
        let (columns, field_ids, nullability) = table_schema(&table)?;
        let scan = table.scan().build().map_err(iceberg_error)?;
        let tasks: Vec<_> =
            datastore.run(async move { scan.plan_files().await?.try_collect().await })?;
        for task in &tasks {
            if !task.deletes.is_empty() {
                return Err(planner_error(format!(
                    "Iceberg table `{}` has delete files; position and equality deletes are not supported in read-only v1",
                    table.identifier()
                )));
            }
            if task.data_file_format != DataFileFormat::Parquet {
                return Err(planner_error(format!(
                    "Iceberg table `{}` references non-Parquet data file `{}`",
                    table.identifier(),
                    task.data_file_path
                )));
            }
            if task.start != 0 || task.length != task.file_size_in_bytes {
                return Err(planner_error(format!(
                    "Iceberg table `{}` produced a split scan task for `{}`; split files are not supported",
                    table.identifier(),
                    task.data_file_path
                )));
            }
        }

        let metadata_location = table.metadata_location().ok_or_else(|| {
            planner_error(format!(
                "Iceberg table `{}` has no committed metadata location",
                table.identifier()
            ))
        })?;
        let revision = TableRevision {
            identity: table.metadata().uuid().to_string(),
            version: metadata_location.to_string(),
        };
        let row_count = tasks.iter().filter_map(|task| task.record_count).sum();
        let bytes = tasks.iter().map(|task| task.file_size_in_bytes).sum();
        Ok(Self {
            table,
            storage: loaded.storage,
            columns,
            field_ids,
            nullability,
            tasks,
            revision,
            row_count,
            bytes,
        })
    }

    fn binding(self: &Arc<Self>, datastore: &str, name: &SchemaQualifiedTableName) -> TableBinding {
        TableBinding {
            reference: TableReference {
                datastore: datastore.to_string(),
                schema: name.schema.clone(),
                table: name.table.clone(),
            },
            frozen: self.clone(),
            parquet: Arc::new(OnceLock::new()),
            predicates: Vec::new(),
        }
    }
}

struct IcebergTransaction {
    datastore: Arc<IcebergDatastore>,
    tables: Mutex<HashMap<TableIdent, Arc<FrozenTable>>>,
}

impl Debug for IcebergTransaction {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("IcebergTransaction").finish()
    }
}

impl IcebergTransaction {
    fn ident(name: &SchemaQualifiedTableName) -> TableIdent {
        TableIdent::new(NamespaceIdent::new(name.schema.clone()), name.table.clone())
    }

    fn load_table_ident(&self, ident: TableIdent) -> PlannerResult<Option<Arc<FrozenTable>>> {
        if let Some(table) = self.tables.lock().unwrap().get(&ident) {
            return Ok(Some(table.clone()));
        }
        let Some(table) = self.datastore.load_table(&ident)? else {
            return Ok(None);
        };
        let frozen = Arc::new(FrozenTable::load(&self.datastore, table)?);
        let mut tables = self.tables.lock().unwrap();
        Ok(Some(
            tables
                .entry(ident)
                .or_insert_with(|| frozen.clone())
                .clone(),
        ))
    }

    fn table(&self, name: &SchemaQualifiedTableName) -> PlannerResult<Option<Arc<FrozenTable>>> {
        self.load_table_ident(Self::ident(name))
    }

    fn build_table_metadata(
        ident: &TableIdent,
        table: &FrozenTable,
    ) -> PlannerResult<DatastoreTableMetadata> {
        let [schema] = ident.namespace.as_ref().as_slice() else {
            return Err(planner_error(format!(
                "Iceberg v1 supports only one-level namespaces, got `{}`",
                ident.namespace
            )));
        };
        Ok(DatastoreTableMetadata {
            name: SchemaQualifiedTableName::new(schema, ident.name.clone()),
            id: table.revision.identity.clone(),
            columns: table
                .columns
                .iter()
                .enumerate()
                .map(|(position, column)| DatastoreColumnMetadata {
                    name: column.name.clone(),
                    column_type: column.col_type.clone(),
                    position,
                    bytes: 0,
                    bytes_uncompressed: 0,
                    is_partition_key: false,
                    is_sort_key: false,
                })
                .collect(),
            sort_by: Vec::new(),
            partition_by: Vec::new(),
            total_rows: table.row_count,
            bytes: table.bytes,
            bytes_uncompressed: 0,
            files: table
                .tasks
                .iter()
                .map(|task| DatastoreFileMetadata {
                    path: task.data_file_path.clone(),
                    bytes: task.file_size_in_bytes,
                    bytes_uncompressed: 0,
                    partition: String::new(),
                    min_max_stats: "{}".to_string(),
                })
                .collect(),
        })
    }
}

#[async_trait]
impl DatastoreTransaction for IcebergTransaction {
    fn does_schema_exist(&self, schema: &str) -> PlannerResult<bool> {
        let namespace = NamespaceIdent::new(schema.to_string());
        self.datastore
            .run(self.datastore.catalog.namespace_exists(&namespace))
    }

    fn bind_table(
        &self,
        datastore: &str,
        name: &SchemaQualifiedTableName,
    ) -> PlannerResult<Option<Box<dyn BoundTable>>> {
        Ok(self
            .table(name)?
            .map(|table| Box::new(table.binding(datastore, name)) as Box<dyn BoundTable>))
    }

    fn table_revision(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> PlannerResult<Option<TableRevision>> {
        Ok(self.table(name)?.map(|table| table.revision.clone()))
    }

    fn tables(&self) -> PlannerResult<Vec<DatastoreTableMetadata>> {
        let namespaces = self
            .datastore
            .run(self.datastore.catalog.list_namespaces(None))?;
        let mut identifiers = Vec::new();
        for namespace in namespaces {
            if namespace.len() != 1 {
                return Err(planner_error(format!(
                    "Iceberg v1 supports only one-level namespaces, got `{namespace}`"
                )));
            }
            identifiers.extend(
                self.datastore
                    .run(self.datastore.catalog.list_tables(&namespace))?,
            );
        }

        let mut metadata = Vec::with_capacity(identifiers.len());
        for ident in identifiers {
            let table = self.load_table_ident(ident.clone())?.ok_or_else(|| {
                planner_error(format!(
                    "Iceberg table `{ident}` disappeared while catalog inventory was loading"
                ))
            })?;
            metadata.push(Self::build_table_metadata(&ident, &table)?);
        }
        Ok(metadata)
    }
}

#[derive(Clone)]
struct TableBinding {
    reference: TableReference,
    frozen: Arc<FrozenTable>,
    parquet: Arc<OnceLock<std::result::Result<Arc<ParquetTable>, String>>>,
    predicates: Vec<PushedPredicate>,
}

impl Debug for TableBinding {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IcebergTableBinding")
            .field("reference", &self.reference)
            .field("revision", &self.frozen.revision)
            .field("predicates", &self.predicates)
            .finish_non_exhaustive()
    }
}

impl TableBinding {
    fn parquet(&self, dispatcher: &DataFlowDispatcher) -> PlannerResult<Arc<ParquetTable>> {
        let loaded = self.parquet.get_or_init(|| {
            let files: std::result::Result<Vec<DataFile>, String> = self
                .frozen
                .tasks
                .iter()
                .map(|task| {
                    self.frozen
                        .storage
                        .data_file(
                            &task.data_file_path,
                            task.file_size_in_bytes,
                            self.frozen.table.file_io().config(),
                        )
                        .map_err(|error| error.to_string())
                })
                .collect();
            let declared_columns = self
                .frozen
                .columns
                .iter()
                .cloned()
                .zip(self.frozen.field_ids.iter().copied())
                .map(|(column, field_id)| DeclaredColumn {
                    column,
                    field_id: Some(field_id),
                })
                .collect();
            ParquetTable::from_locations_with_field_ids(dispatcher, files?, declared_columns)
                .map(Arc::new)
                .map_err(|error| error.to_string())
        });
        loaded.clone().map_err(|message| {
            planner_error(format!("loading Iceberg Parquet metadata: {message}"))
        })
    }
}

impl BoundTable for TableBinding {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.frozen.revision.clone()
    }

    fn is_plan_cacheable(&self) -> bool {
        // Cached plans retain their bindings, including the Apache table and
        // temporary credentials. Bind these again for each transaction.
        false
    }

    fn supports_late_materialization(&self) -> bool {
        true
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> PlannerResult<RecordBatchOperatorSpec> {
        let loaded = self.parquet(dispatcher)?;
        let parquet = Arc::new(prune_parquet(loaded.as_ref(), &self.predicates));
        let equality_predicates = self
            .predicates
            .iter()
            .filter(|predicate| {
                matches!(
                    predicate.compare_type,
                    planner::expression::CompareType::Equal
                )
            })
            .map(|predicate| ScanEqualityPredicate {
                column_idx: predicate.column_idx,
                path: predicate.path.clone(),
                value: predicate.value.clone(),
            })
            .collect();
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(equality_predicates),
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.frozen.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        self.frozen.nullability.clone()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> PlannerResult<RecordBatchOperatorSpec> {
        let loaded = self
            .parquet
            .get()
            .ok_or_else(|| {
                planner_error("Iceberg materialize called before its scan was compiled")
            })?
            .clone()
            .map_err(planner_error)?;
        let parquet = Arc::new(prune_parquet(loaded.as_ref(), &self.predicates));
        Ok(materialize(input, parquet, projection))
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> PlannerResult<bool> {
        self.predicates.extend(PushedPredicate::from_filter(filter));
        Ok(false)
    }

    fn row_count(&self) -> Option<i64> {
        self.predicates
            .is_empty()
            .then(|| i64::try_from(self.frozen.row_count).ok())
            .flatten()
    }

    fn estimate_row_count(&self) -> Option<u64> {
        Some(self.frozen.row_count)
    }
}

fn table_schema(table: &Table) -> PlannerResult<(Vec<Column>, Vec<i32>, Vec<bool>)> {
    let mut columns = Vec::new();
    let mut field_ids = Vec::new();
    let mut nullability = Vec::new();
    for field in table.current_schema_ref().as_struct().fields() {
        columns.push(Column {
            name: field.name.clone(),
            col_type: pivot_type(&field.name, &field.field_type)?,
        });
        field_ids.push(field.id);
        nullability.push(!field.required);
    }
    Ok((columns, field_ids, nullability))
}

fn block_on_runtime<F>(runtime: &tokio::runtime::Handle, future: F) -> PlannerResult<F::Output>
where
    F: Future,
{
    match tokio::runtime::Handle::try_current() {
        Ok(current) if current.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            Ok(tokio::task::block_in_place(|| runtime.block_on(future)))
        }
        Ok(_) => Err(planner_error(
            "Iceberg requires a multi-thread Tokio runtime for synchronous catalog access",
        )),
        Err(_) => Ok(runtime.block_on(future)),
    }
}

fn pivot_type(column: &str, iceberg_type: &IcebergType) -> PlannerResult<Type> {
    let IcebergType::Primitive(primitive) = iceberg_type else {
        return Err(unsupported_type(column, iceberg_type));
    };
    match primitive {
        PrimitiveType::Boolean => Ok(Type::Boolean),
        PrimitiveType::Int => Ok(Type::Int32),
        PrimitiveType::Long => Ok(Type::Int64),
        PrimitiveType::Float => Ok(Type::Float32),
        PrimitiveType::Double => Ok(Type::Float64),
        PrimitiveType::Decimal { precision, scale } if *precision <= 38 => Ok(Type::Decimal {
            precision: u8::try_from(*precision)
                .map_err(|_| unsupported_type(column, iceberg_type))?,
            scale: i8::try_from(*scale).map_err(|_| unsupported_type(column, iceberg_type))?,
        }),
        PrimitiveType::Date => Ok(Type::Date),
        PrimitiveType::Timestamp => Ok(Type::Timestamp),
        PrimitiveType::Timestamptz => Ok(Type::TimestampTz),
        PrimitiveType::String => Ok(Type::Utf8),
        _ => Err(unsupported_type(column, iceberg_type)),
    }
}

fn unsupported_type(column: &str, iceberg_type: &IcebergType) -> PlannerError {
    planner_error(format!(
        "Iceberg column `{column}` has unsupported v1 type `{iceberg_type}`"
    ))
}

fn iceberg_error(error: iceberg::Error) -> PlannerError {
    PlannerError::Other(Box::new(error))
}

fn planner_error(message: impl Into<String>) -> PlannerError {
    PlannerError::Other(message.into().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap_primitive(primitive: PrimitiveType) -> IcebergType {
        IcebergType::Primitive(primitive)
    }

    #[test]
    fn supported_iceberg_primitive_types_map_exactly() {
        let cases = [
            (PrimitiveType::Boolean, Type::Boolean),
            (PrimitiveType::Int, Type::Int32),
            (PrimitiveType::Long, Type::Int64),
            (PrimitiveType::Float, Type::Float32),
            (PrimitiveType::Double, Type::Float64),
            (PrimitiveType::Date, Type::Date),
            (PrimitiveType::Timestamp, Type::Timestamp),
            (PrimitiveType::Timestamptz, Type::TimestampTz),
            (PrimitiveType::String, Type::Utf8),
        ];

        for (iceberg, expected) in cases {
            assert_eq!(
                pivot_type("value", &wrap_primitive(iceberg)).unwrap(),
                expected
            );
        }
        assert_eq!(
            pivot_type(
                "price",
                &wrap_primitive(PrimitiveType::Decimal {
                    precision: 38,
                    scale: 12,
                }),
            )
            .unwrap(),
            Type::Decimal {
                precision: 38,
                scale: 12,
            }
        );
    }

    #[test]
    fn unsupported_v1_primitive_types_fail_with_the_column_name() {
        let unsupported = [
            PrimitiveType::Time,
            PrimitiveType::TimestampNs,
            PrimitiveType::TimestamptzNs,
            PrimitiveType::Uuid,
            PrimitiveType::Fixed(16),
            PrimitiveType::Binary,
        ];

        for iceberg in unsupported {
            let error = pivot_type("unsupported_column", &wrap_primitive(iceberg)).unwrap_err();
            assert!(error.to_string().contains("unsupported_column"), "{error}");
        }
    }

    #[test]
    fn catalog_futures_can_block_from_a_tokio_worker() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap();

        let value = runtime.block_on(async {
            block_on_runtime(&tokio::runtime::Handle::current(), async { 42 }).unwrap()
        });

        assert_eq!(value, 42);
    }
}
