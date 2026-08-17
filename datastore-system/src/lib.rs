//! The global, read-only `system` datastore.
//!
//! Its content spans every configured datastore, so it belongs to none of them:
//! it is served as a datastore of its own rather than as a schema each datastore
//! appears to carry. A query reaches it the usual way, `system.tables`, since a
//! two-part name resolves its first part as a datastore before a schema.
//!
//! It has no storage, so it is not a [`datastore::Datastore`] the catalog holds
//! a handle to. Only its transaction is real, assembled by the composite from
//! the sub-transactions the query already opened, and from there the catalog
//! routes to it exactly as it routes to a stored datastore.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{Field, Schema};
use async_trait::async_trait;
use datastore::{DatastoreTableMetadata, DatastoreTransaction};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec, values_input};
use planner::catalog::{
    BoundTable, Column, CreateSchemaRequest, CreateTableRequest, DropTableRequest,
    DynamicScanPredicate, Error as CatalogError, Result, SchemaCreation, SchemaQualifiedTableName,
    TableCreation, TableDrop, TableReference, TableRevision,
};
use planner::types::{Type, physical_arrow_type};

pub const DATASTORE_NAME: &str = "system";
const TABLES_NAME: &str = "tables";
const TABLE_FILES_NAME: &str = "table_files";

/// A statement this datastore cannot serve. Its relations are assembled per
/// query from the datastores it is composed with, so there is nothing for a
/// write to change; each variant names what the statement targeted.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot create table `{}` in the read-only `{}` datastore", .0, DATASTORE_NAME)]
    CreateTable(SchemaQualifiedTableName),
    #[error("cannot drop table `{}` from the read-only `{}` datastore", .0, DATASTORE_NAME)]
    DropTable(SchemaQualifiedTableName),
    #[error("cannot create schema `{}` in the read-only `{}` datastore", .0, DATASTORE_NAME)]
    CreateSchema(String),
}

/// How one relation is built from the global inventory.
type BuildRelation = fn(TableReference, Vec<GlobalTableMetadata>) -> VirtualTableBinding;

/// Every relation this datastore serves. The one place a system table is
/// registered: binding one, and reporting it in the inventory, both read this.
const RELATIONS: [(&str, BuildRelation); 2] = [
    (TABLES_NAME, build_tables_relation),
    (TABLE_FILES_NAME, build_table_files_relation),
];

/// One table paired with the datastore that owns it. Datastores only know their
/// schema-qualified names; the inventory adds this final qualifier.
struct GlobalTableMetadata {
    datastore_name: String,
    table: DatastoreTableMetadata,
}

/// The `system` datastore's transaction: the sub-transaction of every other
/// datastore, since each of its relations is assembled from them all. The
/// composite opens them before building this, so a query that reads a datastore
/// directly and through `system.tables` reads one snapshot of it either way.
#[derive(Debug)]
pub struct SystemTransaction {
    datastores: Vec<(String, Arc<dyn DatastoreTransaction>)>,
}

impl SystemTransaction {
    pub fn new(datastores: Vec<(String, Arc<dyn DatastoreTransaction>)>) -> Self {
        Self { datastores }
    }

    /// Every table of every datastore, this one included: the inventory both
    /// relations are built from. Listing its own relations is what lets
    /// `system.tables` describe the datastore serving it.
    fn global_tables(&self) -> Vec<GlobalTableMetadata> {
        self.datastores
            .iter()
            .flat_map(|(datastore_name, transaction)| {
                transaction
                    .tables()
                    .into_iter()
                    .map(move |table| GlobalTableMetadata {
                        datastore_name: datastore_name.clone(),
                        table,
                    })
            })
            .chain(self.tables().into_iter().map(|table| GlobalTableMetadata {
                datastore_name: DATASTORE_NAME.to_string(),
                table,
            }))
            .collect()
    }
}

#[async_trait]
impl DatastoreTransaction for SystemTransaction {
    /// Every relation lives in the default schema, the one an unqualified
    /// `system.<table>` resolves to.
    fn does_schema_exist(&self, schema: &str) -> bool {
        schema == planner::DEFAULT_SCHEMA_NAME
    }

    fn bind_table(
        &self,
        datastore: &str,
        name: &SchemaQualifiedTableName,
    ) -> Option<Box<dyn BoundTable>> {
        if name.schema != planner::DEFAULT_SCHEMA_NAME {
            return None;
        }
        let reference = TableReference {
            datastore: datastore.to_string(),
            schema: name.schema.clone(),
            table: name.table.clone(),
        };
        let (_, build) = RELATIONS
            .iter()
            .find(|(relation, _)| *relation == name.table)?;
        Some(Box::new(build(reference, self.global_tables())))
    }

    /// Asked of the binding rather than of a second list of names, so the
    /// contract that every bindable table has a revision holds by construction.
    fn table_revision(&self, name: &SchemaQualifiedTableName) -> Option<TableRevision> {
        Some(self.bind_table(DATASTORE_NAME, name)?.table_revision())
    }

    /// The relations this datastore serves. None of them is stored, so none has
    /// files or a durable identifier; each is named by the qualified name that
    /// reaches it, which is unique and stable for as long as it is served.
    fn tables(&self) -> Vec<DatastoreTableMetadata> {
        RELATIONS
            .iter()
            .map(|(relation, _)| DatastoreTableMetadata {
                name: SchemaQualifiedTableName::new(planner::DEFAULT_SCHEMA_NAME, *relation),
                id: format!("{DATASTORE_NAME}.{relation}"),
                files: Vec::new(),
            })
            .collect()
    }

    fn bind_create_table(&self, request: CreateTableRequest) -> Result<Box<dyn TableCreation>> {
        Err(CatalogError::Other(Box::new(Error::CreateTable(
            request.schema_qualified_name(),
        ))))
    }

    fn bind_drop_table(&self, request: DropTableRequest) -> Result<Box<dyn TableDrop>> {
        Err(CatalogError::Other(Box::new(Error::DropTable(
            request.schema_qualified_name(),
        ))))
    }

    fn bind_create_schema(&self, request: CreateSchemaRequest) -> Result<Box<dyn SchemaCreation>> {
        Err(CatalogError::Other(Box::new(Error::CreateSchema(
            request.name,
        ))))
    }
}

/// Build the catalog-wide `system.tables` relation.
fn build_tables_relation(
    reference: TableReference,
    tables: Vec<GlobalTableMetadata>,
) -> VirtualTableBinding {
    let datastore_names: Vec<_> = tables
        .iter()
        .map(|entry| entry.datastore_name.as_str())
        .collect();
    let schema_names: Vec<_> = tables
        .iter()
        .map(|entry| entry.table.name.schema.as_str())
        .collect();
    let names: Vec<_> = tables
        .iter()
        .map(|entry| entry.table.name.table.as_str())
        .collect();
    let ids: Vec<_> = tables.iter().map(|entry| entry.table.id.as_str()).collect();

    build_table(
        reference,
        vec![
            string_column("datastore_name", datastore_names),
            string_column("schema_name", schema_names),
            string_column("name", names),
            string_column("id", ids),
        ],
    )
}

/// Build the catalog-wide `system.table_files` relation: one row per committed
/// data file, named by the table that holds it rather than by that table's
/// datastore, since a file's owning table is what `system.tables` joins on.
fn build_table_files_relation(
    reference: TableReference,
    tables: Vec<GlobalTableMetadata>,
) -> VirtualTableBinding {
    let files: Vec<_> = tables
        .into_iter()
        .flat_map(|entry| {
            let table_id = entry.table.id;
            entry
                .table
                .files
                .into_iter()
                .map(move |file| (table_id.clone(), file))
        })
        .collect();

    let table_ids: Vec<_> = files.iter().map(|(id, _)| id.as_str()).collect();
    let paths: Vec<_> = files.iter().map(|(_, file)| file.path.as_str()).collect();
    // A file is far smaller than an i64 holds, so the cast keeps every size and
    // spares clients an unsigned type they mostly lack.
    let sizes: Vec<_> = files
        .iter()
        .map(|(_, file)| file.size_bytes as i64)
        .collect();

    build_table(
        reference,
        vec![
            string_column("table_id", table_ids),
            string_column("path", paths),
            int_column("size_bytes", sizes),
        ],
    )
}

#[derive(Debug, Clone)]
struct VirtualTableBinding {
    reference: TableReference,
    columns: Vec<Column>,
    batch: Arc<RecordBatch>,
}

impl BoundTable for VirtualTableBinding {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    /// Names this binding, but dates nothing: the batch is rebuilt from the
    /// transaction's snapshot on every bind, so there is no version to advance.
    /// Only a cached plan reads a revision, and one of these is never cached.
    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: format!("{DATASTORE_NAME}.{}", self.reference.table),
            version: 0,
        }
    }

    fn is_plan_cacheable(&self) -> bool {
        false
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        _dynamic_filters: Vec<DynamicScanPredicate>,
        _emit_row_group_metadata: bool,
    ) -> Result<RecordBatchOperatorSpec> {
        let projected = self
            .batch
            .project(projection.indices())
            .map_err(|error| planner::catalog::Error::Other(Box::new(error)))?;
        Ok(values_input(dispatcher, [projected]).record_batches())
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        vec![false; self.columns.len()]
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn row_count(&self) -> Option<i64> {
        i64::try_from(self.batch.num_rows()).ok()
    }

    fn estimate_row_count(&self) -> Option<u64> {
        Some(self.batch.num_rows() as u64)
    }
}

/// Build the Arrow batch and its planner columns from the same ordered pairs,
/// keeping each virtual column's name, logical type, and array adjacent.
fn build_table(reference: TableReference, columns: Vec<(Column, ArrayRef)>) -> VirtualTableBinding {
    let fields: Vec<_> = columns
        .iter()
        .map(|(column, _)| Field::new(&column.name, physical_arrow_type(&column.col_type), false))
        .collect();
    let arrays: Vec<_> = columns.iter().map(|(_, array)| array.clone()).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .expect("system table columns have equal lengths and declared physical types");
    VirtualTableBinding {
        reference,
        columns: columns.into_iter().map(|(column, _)| column).collect(),
        batch: Arc::new(batch),
    }
}

fn string_column(name: &str, values: Vec<&str>) -> (Column, ArrayRef) {
    (
        Column {
            name: name.to_string(),
            col_type: Type::Utf8,
        },
        Arc::new(StringViewArray::from(values)),
    )
}

fn int_column(name: &str, values: Vec<i64>) -> (Column, ArrayRef) {
    (
        Column {
            name: name.to_string(),
            col_type: Type::Int64,
        },
        Arc::new(Int64Array::from(values)),
    )
}
