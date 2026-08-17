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

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{Field, Schema, SchemaRef};
use async_trait::async_trait;
use datastore::{DatastoreTableMetadata, DatastoreTransaction};
use dispatch::{
    DataFlowDispatcher, MemoryBlockState, MemoryBlockStatus, OneShotNullaryFactory, Projection,
    RecordBatchOperatorSpec, memory_ctx, values_input,
};
use planner::catalog::{
    BoundTable, Column, CreateSchemaRequest, CreateTableRequest, DropTableRequest,
    DynamicScanPredicate, Error as CatalogError, Result, SchemaCreation, SchemaQualifiedTableName,
    TableCreation, TableDrop, TableReference, TableRevision,
};
use planner::types::{Type, physical_arrow_type};

pub const DATASTORE_NAME: &str = "system";
const TABLES_NAME: &str = "tables";
const TABLE_FILES_NAME: &str = "table_files";
const MEMORY_BLOCKS_NAME: &str = "memory_blocks";

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

/// How one relation is built from the global inventory. What it returns decides
/// how the relation is served: rows the inventory already holds, or a scan that
/// reads them when the query runs.
type BuildRelation = fn(TableReference, Vec<GlobalTableMetadata>) -> Box<dyn BoundTable>;

/// Every relation this datastore serves. The one place a system table is
/// registered: binding one, and reporting it in the inventory, both read this.
const RELATIONS: [(&str, BuildRelation); 3] = [
    (TABLES_NAME, build_tables_relation),
    (TABLE_FILES_NAME, build_table_files_relation),
    (MEMORY_BLOCKS_NAME, build_memory_blocks_relation),
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
        Some(build(reference, self.global_tables()))
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
) -> Box<dyn BoundTable> {
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
) -> Box<dyn BoundTable> {
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

/// Build the `system.memory_blocks` relation: one row per block of the memory
/// ring, the fixed-size unit every cache and every operator allocates in.
///
/// Reporting the blocks themselves rather than a summary of them is what makes
/// the summary a query: blocks are all one size, so the share of memory in any
/// state is the share of rows in it, and no aggregate has to be anticipated
/// here. Nothing in the catalog describes memory, so the inventory the other
/// relations are built from is not read.
fn build_memory_blocks_relation(
    reference: TableReference,
    _tables: Vec<GlobalTableMetadata>,
) -> Box<dyn BoundTable> {
    build_scanned_table(
        reference,
        vec![
            declare_column("slot", Type::Int64),
            declare_column("node", Type::Int64),
            declare_column("state", Type::Utf8),
            declare_column("readers", Type::Int64),
            declare_column("size_bytes", Type::Int64),
        ],
        compile_memory_blocks_scan,
    )
}

/// Compile the scan of the memory ring.
///
/// A worker's handle to the memory it allocates from is thread-local, so the
/// read has to happen on a worker.
fn compile_memory_blocks_scan(
    dispatcher: &DataFlowDispatcher,
    schema: SchemaRef,
    projection: Projection,
) -> RecordBatchOperatorSpec {
    // The first worker to be built reads the ring; the rest emit nothing.
    let mut reader = Some((schema, projection));
    let factories: Vec<_> = (0..dispatcher.worker_count())
        .map(|_| {
            let staged = reader.take();
            OneShotNullaryFactory::new(move || {
                let (schema, projection) = staged?;
                let blocks = memory_ctx().read_all_blocks();
                let batch = RecordBatch::try_new(schema, memory_block_arrays(&blocks))
                    .expect("system table columns have equal lengths and declared physical types");
                Some(
                    batch
                        .project(&projection.column_indices)
                        .expect("the projection indexes this relation's own columns"),
                )
            })
        })
        .collect();
    RecordBatchOperatorSpec::from_nullary(dispatcher, factories)
}

/// One row per block of the ring.
fn memory_block_arrays(blocks: &[MemoryBlockStatus]) -> Vec<ArrayRef> {
    // Every block is the same size, so the column repeats one value; it is
    // carried anyway so that summing memory needs no knowledge of that size.
    let size_bytes = dispatch::block_size_bytes() as i64;

    vec![
        int_array(blocks.iter().map(|block| block.slot as i64).collect()),
        int_array(blocks.iter().map(|block| block.node as i64).collect()),
        string_array(blocks.iter().map(|block| state_name(block.state)).collect()),
        int_array(blocks.iter().map(|block| block.readers as i64).collect()),
        int_array(vec![size_bytes; blocks.len()]),
    ]
}

/// The name a block's state answers to in the `state` column. These are the
/// values queries compare against, so they are spelled here, with the relation
/// serving them, rather than by the memory subsystem being described.
fn state_name(state: MemoryBlockState) -> &'static str {
    match state {
        MemoryBlockState::Free => "free",
        MemoryBlockState::Pinned => "pinned",
        MemoryBlockState::CompressedCache => "compressed_cache",
        MemoryBlockState::DecompressedCache => "decompressed_cache",
    }
}

/// Compile a relation that reads itself when scanned, into the operator that
/// emits its projected rows.
type CompileScan = fn(&DataFlowDispatcher, SchemaRef, Projection) -> RecordBatchOperatorSpec;

/// What every relation here has, whichever way its rows are come by: the name
/// that reaches it and the columns it serves. Fixed when the relation is
/// registered, so binding always knows the shape.
#[derive(Debug, Clone)]
struct Relation {
    reference: TableReference,
    columns: Vec<Column>,
    schema: SchemaRef,
}

impl Relation {
    fn new(reference: TableReference, columns: Vec<Column>) -> Self {
        Self {
            reference,
            schema: build_schema(&columns),
            columns,
        }
    }

    /// Names a binding of this relation, but dates nothing: its rows are read
    /// afresh every time, so there is no version to advance. Only a cached plan
    /// reads a revision, and one of these is never cached.
    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: format!("{DATASTORE_NAME}.{}", self.reference.table),
            version: 0,
        }
    }
}

/// A relation whose rows the transaction already held when the table bound.
/// Scanning it is handing them over, and their count is known without doing so.
#[derive(Debug, Clone)]
struct BoundRelation {
    relation: Relation,
    arrays: Vec<ArrayRef>,
}

impl BoundRelation {
    /// Every relation declares at least one column, so the first array's length
    /// is the row count.
    fn held_row_count(&self) -> usize {
        self.arrays
            .first()
            .expect("a system relation declares at least one column")
            .len()
    }
}

impl BoundTable for BoundRelation {
    fn table_reference(&self) -> TableReference {
        self.relation.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.relation.table_revision()
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
        let batch = RecordBatch::try_new(self.relation.schema.clone(), self.arrays.clone())
            .expect("system table columns have equal lengths and declared physical types");
        let projected = batch
            .project(projection.indices())
            .map_err(|error| planner::catalog::Error::Other(Box::new(error)))?;
        Ok(values_input(dispatcher, [projected]).record_batches())
    }

    fn columns(&self) -> Vec<Column> {
        self.relation.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        vec![false; self.relation.columns.len()]
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    /// Exact, and answered without scanning: the rows are already here.
    fn row_count(&self) -> Option<i64> {
        i64::try_from(self.held_row_count()).ok()
    }

    fn estimate_row_count(&self) -> Option<u64> {
        Some(self.held_row_count() as u64)
    }
}

/// A relation that reads itself when scanned, because its rows describe the
/// running server rather than the catalog. Nothing has read them at bind time,
/// so it has no count to offer and is scanned to be counted.
#[derive(Clone)]
struct ScannedRelation {
    relation: Relation,
    compile_function: CompileScan,
}

/// Shows the relation, not a function pointer's address.
impl Debug for ScannedRelation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScannedRelation({:?})", self.relation)
    }
}

impl BoundTable for ScannedRelation {
    fn table_reference(&self) -> TableReference {
        self.relation.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.relation.table_revision()
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
        Ok((self.compile_function)(
            dispatcher,
            self.relation.schema.clone(),
            projection,
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.relation.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        vec![false; self.relation.columns.len()]
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn row_count(&self) -> Option<i64> {
        None
    }

    fn estimate_row_count(&self) -> Option<u64> {
        None
    }
}

/// Build a relation whose rows the transaction already holds, from ordered
/// pairs that keep each virtual column's name, logical type, and array
/// adjacent.
fn build_table(reference: TableReference, columns: Vec<(Column, ArrayRef)>) -> Box<dyn BoundTable> {
    let arrays: Vec<_> = columns.iter().map(|(_, array)| array.clone()).collect();
    let columns: Vec<_> = columns.into_iter().map(|(column, _)| column).collect();
    Box::new(BoundRelation {
        relation: Relation::new(reference, columns),
        arrays,
    })
}

/// Build a relation that reads itself when it is scanned, declaring its columns
/// now and the scan that will produce them then. The two must agree: the scan
/// emits one array per column, of that column's physical type.
fn build_scanned_table(
    reference: TableReference,
    columns: Vec<Column>,
    compile_function: CompileScan,
) -> Box<dyn BoundTable> {
    Box::new(ScannedRelation {
        relation: Relation::new(reference, columns),
        compile_function,
    })
}

/// The Arrow schema a relation's arrays are assembled against, one field per
/// declared column. Nothing a system table serves is null, so no field is.
fn build_schema(columns: &[Column]) -> SchemaRef {
    Arc::new(Schema::new(
        columns
            .iter()
            .map(|column| Field::new(&column.name, physical_arrow_type(&column.col_type), false))
            .collect::<Vec<_>>(),
    ))
}

fn string_column(name: &str, values: Vec<&str>) -> (Column, ArrayRef) {
    (declare_column(name, Type::Utf8), string_array(values))
}

fn int_column(name: &str, values: Vec<i64>) -> (Column, ArrayRef) {
    (declare_column(name, Type::Int64), int_array(values))
}

fn declare_column(name: &str, col_type: Type) -> Column {
    Column {
        name: name.to_string(),
        col_type,
    }
}

fn string_array(values: Vec<&str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

fn int_array(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}
