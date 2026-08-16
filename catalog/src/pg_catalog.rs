//! Read-only PostgreSQL catalog relations built from a datastore transaction.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Int16Array, Int32Array, Int64Array, RecordBatch,
    StringViewArray, UInt32Array, new_empty_array, new_null_array,
};
use arrow_schema::{Field, Schema};
use datastore::DatastoreTransaction;
use dispatch::{DataFlowDispatcher, OneShotNullaryFactory, Projection, RecordBatchOperatorSpec};
use metastore::{DEFAULT_USER_NAME, Metastore};
use planner::DEFAULT_SCHEMA_NAME;
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Result, TableReference, TableRevision,
};
use planner::pg_catalog::{
    PIVOT_OWNER_OID, VIRTUAL_OID_PREFIX, VISIBLE_RELATION_OID_FLAG, describe_type,
    mark_relation_oid_visible, type_modifier, type_oid,
};
use planner::types::{Type, physical_arrow_type};

pub const SCHEMA_NAME: &str = "pg_catalog";

const RELATIONS: &[&str] = &[
    "pg_am",
    "pg_attrdef",
    "pg_attribute",
    "pg_auth_members",
    "pg_class",
    "pg_collation",
    "pg_constraint",
    "pg_description",
    "pg_inherits",
    "pg_namespace",
    "pg_policy",
    "pg_publication",
    "pg_publication_namespace",
    "pg_publication_rel",
    "pg_roles",
    "pg_statistic_ext",
    "pg_type",
];

/// Process-local OIDs for the virtual catalog's schemas and relations.
///
/// One registry belongs to a [`crate::PivotCatalog`], so every query handled by
/// that catalog sees the same OID for an object. Nothing persists this map, and
/// a new server catalog starts with an empty allocation sequence.
#[derive(Debug, Default)]
pub(crate) struct OidRegistry {
    state: Mutex<OidRegistryState>,
}

#[derive(Debug, Default)]
struct OidRegistryState {
    oids_by_object: HashMap<OidObject, u32>,
    next_sequence: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum OidObject {
    Schema { datastore: String, schema: String },
    Relation { datastore: String, identity: String },
    Role { name: String },
}

impl OidRegistry {
    fn schema_oid(&self, datastore: &str, schema: &str) -> u32 {
        self.oid_for(OidObject::Schema {
            datastore: datastore.to_string(),
            schema: schema.to_string(),
        })
    }

    fn relation_oid(&self, datastore: &str, identity: &str, schema: &str) -> u32 {
        let oid = self.oid_for(OidObject::Relation {
            datastore: datastore.to_string(),
            identity: identity.to_string(),
        });
        if schema == DEFAULT_SCHEMA_NAME {
            mark_relation_oid_visible(oid)
        } else {
            oid
        }
    }

    fn role_oid(&self, name: &str) -> u32 {
        // Object ownership is deliberately synthetic rather than persisted on
        // schemas or tables. Keep its PostgreSQL-compatible bootstrap OID
        // stable while allocating every authentication-only role from this
        // process-local registry like the other virtual catalog objects.
        if name == DEFAULT_USER_NAME {
            PIVOT_OWNER_OID
        } else {
            self.oid_for(OidObject::Role {
                name: name.to_string(),
            })
        }
    }

    fn oid_for(&self, object: OidObject) -> u32 {
        let mut state = self.state.lock().unwrap();
        if let Some(oid) = state.oids_by_object.get(&object) {
            return *oid;
        }

        if state.next_sequence >= VISIBLE_RELATION_OID_FLAG {
            panic!("virtual PostgreSQL OID space exhausted");
        }
        let oid = VIRTUAL_OID_PREFIX | state.next_sequence;
        state.next_sequence += 1;
        state.oids_by_object.insert(object, oid);
        oid
    }
}

pub fn table_revision(datastore: &str, table: &str) -> Option<TableRevision> {
    RELATIONS.contains(&table).then(|| TableRevision {
        identity: format!("{datastore}.{SCHEMA_NAME}.{table}"),
        version: 0,
    })
}

pub fn bind_table(
    reference: TableReference,
    transaction: &dyn DatastoreTransaction,
    oid_registry: &OidRegistry,
    metastore: &dyn Metastore,
) -> Option<Box<dyn BoundTable>> {
    let table = match reference.table.as_str() {
        "pg_namespace" => build_pg_namespace(&reference, transaction, oid_registry),
        "pg_class" => build_pg_class(&reference, transaction, oid_registry),
        "pg_attribute" => build_pg_attribute(&reference, transaction, oid_registry),
        "pg_type" => build_pg_type(&reference, transaction, oid_registry),
        "pg_roles" => build_pg_roles(&reference, metastore, oid_registry),
        // Pivot has authentication users but no role-membership or privilege
        // model. Modern psql's `\du` still reads this catalog to render its
        // "Member of" column, so expose the PostgreSQL 16 shape as empty.
        "pg_auth_members" => build_empty(
            &reference,
            &[
                column("oid", Type::UInt32),
                column("roleid", Type::UInt32),
                column("member", Type::UInt32),
                column("grantor", Type::UInt32),
                column("admin_option", Type::Boolean),
                column("inherit_option", Type::Boolean),
                column("set_option", Type::Boolean),
            ],
        ),
        "pg_am" => build_empty(
            &reference,
            &[column("oid", Type::UInt32), column("amname", Type::Utf8)],
        ),
        "pg_attrdef" => build_empty(
            &reference,
            &[
                column("oid", Type::UInt32),
                column("adrelid", Type::UInt32),
                column("adnum", Type::Int16),
                column("adbin", Type::Utf8),
            ],
        ),
        "pg_collation" => build_empty(
            &reference,
            &[column("oid", Type::UInt32), column("collname", Type::Utf8)],
        ),
        "pg_constraint" => build_empty(
            &reference,
            &[
                column("oid", Type::UInt32),
                column("conname", Type::Utf8),
                column("connamespace", Type::UInt32),
                column("contype", Type::Utf8),
                column("condeferrable", Type::Boolean),
                column("condeferred", Type::Boolean),
                column("convalidated", Type::Boolean),
                column("conrelid", Type::UInt32),
                column("contypid", Type::UInt32),
                column("conindid", Type::UInt32),
                column("conparentid", Type::UInt32),
                column("confrelid", Type::UInt32),
                column("conislocal", Type::Boolean),
                column("coninhcount", Type::Int16),
                column("connoinherit", Type::Boolean),
                column("conkey", Type::List(Box::new(Type::Int16))),
                column("confkey", Type::List(Box::new(Type::Int16))),
                column("conbin", Type::Utf8),
            ],
        ),
        "pg_description" => build_empty(
            &reference,
            &[
                column("objoid", Type::UInt32),
                column("classoid", Type::UInt32),
                column("objsubid", Type::Int32),
                column("description", Type::Utf8),
            ],
        ),
        "pg_inherits" => build_empty(
            &reference,
            &[
                column("inhrelid", Type::UInt32),
                column("inhparent", Type::UInt32),
                column("inhseqno", Type::Int32),
                column("inhdetachpending", Type::Boolean),
            ],
        ),
        "pg_policy" => build_empty(
            &reference,
            &[
                column("polname", Type::Utf8),
                column("polpermissive", Type::Boolean),
                column("polroles", Type::List(Box::new(Type::UInt32))),
                column("polqual", Type::Utf8),
                column("polrelid", Type::UInt32),
                column("polwithcheck", Type::Utf8),
                column("polcmd", Type::Utf8),
            ],
        ),
        "pg_statistic_ext" => build_empty(
            &reference,
            &[
                column("oid", Type::UInt32),
                column("stxrelid", Type::UInt32),
                column("stxnamespace", Type::UInt32),
                column("stxname", Type::Utf8),
                column("stxkeys", Type::List(Box::new(Type::Int16))),
                column("stxkind", Type::List(Box::new(Type::Utf8))),
                column("stxstattarget", Type::Int32),
            ],
        ),
        "pg_publication" => build_empty(
            &reference,
            &[
                column("oid", Type::UInt32),
                column("pubname", Type::Utf8),
                column("puballtables", Type::Boolean),
            ],
        ),
        "pg_publication_namespace" => build_empty(
            &reference,
            &[
                column("oid", Type::UInt32),
                column("pnpubid", Type::UInt32),
                column("pnnspid", Type::UInt32),
            ],
        ),
        "pg_publication_rel" => build_empty(
            &reference,
            &[
                column("oid", Type::UInt32),
                column("prpubid", Type::UInt32),
                column("prrelid", Type::UInt32),
                column("prqual", Type::Utf8),
                column("prattrs", Type::List(Box::new(Type::Int16))),
            ],
        ),
        _ => return None,
    };
    Some(Box::new(table))
}

#[derive(Debug, Clone)]
struct VirtualTableBinding {
    reference: TableReference,
    revision: TableRevision,
    columns: Vec<Column>,
    batch: Arc<RecordBatch>,
}

impl BoundTable for VirtualTableBinding {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.revision.clone()
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
        let mut projected = Some(projected);
        let factories: Vec<_> = (0..dispatcher.worker_count())
            .map(|_| {
                let batch = projected.take();
                OneShotNullaryFactory::new(move || batch)
            })
            .collect();
        Ok(RecordBatchOperatorSpec::from_nullary(dispatcher, factories))
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        self.batch
            .columns()
            .iter()
            .map(|array| array.null_count() > 0)
            .collect()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn estimate_row_count(&self) -> Option<u64> {
        Some(self.batch.num_rows() as u64)
    }

    fn row_count(&self) -> Option<i64> {
        i64::try_from(self.batch.num_rows()).ok()
    }
}

fn build_pg_namespace(
    reference: &TableReference,
    transaction: &dyn DatastoreTransaction,
    oid_registry: &OidRegistry,
) -> VirtualTableBinding {
    let mut schema_names = transaction.schema_names();
    schema_names.push(SCHEMA_NAME.to_string());
    schema_names.sort();
    schema_names.dedup();
    let oids: Vec<_> = schema_names
        .iter()
        .map(|schema| oid_registry.schema_oid(&reference.datastore, schema))
        .collect();
    let row_count = schema_names.len();

    build_table(
        reference,
        vec![
            (
                column("oid", Type::UInt32),
                Arc::new(UInt32Array::from(oids)),
            ),
            (
                column("nspname", Type::Utf8),
                Arc::new(StringViewArray::from_iter_values(
                    schema_names.iter().map(String::as_str),
                )),
            ),
            (
                column("nspowner", Type::UInt32),
                Arc::new(UInt32Array::from(vec![PIVOT_OWNER_OID; row_count])),
            ),
            (
                column("nspacl", Type::Utf8),
                Arc::new(StringViewArray::from(vec![None::<&str>; row_count])),
            ),
        ],
    )
}

fn build_pg_class(
    reference: &TableReference,
    transaction: &dyn DatastoreTransaction,
    oid_registry: &OidRegistry,
) -> VirtualTableBinding {
    let tables = transaction.tables();
    let row_count = tables.len();
    let relation_oids: Vec<_> = tables
        .iter()
        .map(|table| {
            oid_registry.relation_oid(
                &reference.datastore,
                &table.revision.identity,
                &table.name.schema,
            )
        })
        .collect();
    let names: Vec<_> = tables
        .iter()
        .map(|table| table.name.table.as_str())
        .collect();
    let schema_oids: Vec<_> = tables
        .iter()
        .map(|table| oid_registry.schema_oid(&reference.datastore, &table.name.schema))
        .collect();
    let attribute_counts: Vec<_> = tables
        .iter()
        .map(|table| {
            i16::try_from(table.columns.len())
                .expect("PostgreSQL attribute numbers must fit in a signed 16-bit integer")
        })
        .collect();

    build_table(
        reference,
        vec![
            (
                column("oid", Type::UInt32),
                Arc::new(UInt32Array::from(relation_oids)),
            ),
            (
                column("relname", Type::Utf8),
                Arc::new(StringViewArray::from(names)),
            ),
            (
                column("relnamespace", Type::UInt32),
                Arc::new(UInt32Array::from(schema_oids)),
            ),
            oid_column("reltype", row_count, 0),
            oid_column("reloftype", row_count, 0),
            oid_column("relowner", row_count, PIVOT_OWNER_OID),
            oid_column("relam", row_count, 0),
            oid_column("relfilenode", row_count, 0),
            oid_column("reltablespace", row_count, 0),
            int32_column("relpages", row_count, 0),
            (
                column("reltuples", Type::Float32),
                Arc::new(Float32Array::from(vec![0.0; row_count])),
            ),
            int32_column("relallvisible", row_count, 0),
            oid_column("reltoastrelid", row_count, 0),
            bool_column("relhasindex", row_count, false),
            bool_column("relisshared", row_count, false),
            string_column("relpersistence", row_count, "p"),
            string_column("relkind", row_count, "r"),
            (
                column("relnatts", Type::Int16),
                Arc::new(Int16Array::from(attribute_counts)),
            ),
            int16_column("relchecks", row_count, 0),
            bool_column("relhasoids", row_count, false),
            bool_column("relhaspkey", row_count, false),
            bool_column("relhasrules", row_count, false),
            bool_column("relhastriggers", row_count, false),
            bool_column("relhassubclass", row_count, false),
            bool_column("relrowsecurity", row_count, false),
            bool_column("relforcerowsecurity", row_count, false),
            bool_column("relispopulated", row_count, true),
            string_column("relreplident", row_count, "d"),
            bool_column("relispartition", row_count, false),
            (
                column("reloptions", Type::Utf8),
                Arc::new(StringViewArray::from(vec![None::<&str>; row_count])),
            ),
            (
                column("relpartbound", Type::Utf8),
                Arc::new(StringViewArray::from(vec![None::<&str>; row_count])),
            ),
        ],
    )
}

fn build_pg_attribute(
    reference: &TableReference,
    transaction: &dyn DatastoreTransaction,
    oid_registry: &OidRegistry,
) -> VirtualTableBinding {
    struct Attribute<'a> {
        relation_oid: u32,
        position: i16,
        column: &'a Column,
    }

    let tables = transaction.tables();
    let attributes: Vec<_> = tables
        .iter()
        .flat_map(|table| {
            let oid = oid_registry.relation_oid(
                &reference.datastore,
                &table.revision.identity,
                &table.name.schema,
            );
            table
                .columns
                .iter()
                .enumerate()
                .map(move |(position, column)| Attribute {
                    relation_oid: oid,
                    position: i16::try_from(position + 1)
                        .expect("PostgreSQL attribute numbers must fit in a signed 16-bit integer"),
                    column,
                })
        })
        .collect();
    let row_count = attributes.len();

    build_table(
        reference,
        vec![
            (
                column("attrelid", Type::UInt32),
                Arc::new(UInt32Array::from(
                    attributes
                        .iter()
                        .map(|attribute| attribute.relation_oid)
                        .collect::<Vec<_>>(),
                )),
            ),
            (
                column("attname", Type::Utf8),
                Arc::new(StringViewArray::from_iter_values(
                    attributes
                        .iter()
                        .map(|attribute| attribute.column.name.as_str()),
                )),
            ),
            (
                column("atttypid", Type::UInt32),
                Arc::new(UInt32Array::from(
                    attributes
                        .iter()
                        .map(|attribute| type_oid(&attribute.column.col_type))
                        .collect::<Vec<_>>(),
                )),
            ),
            int32_column("attstattarget", row_count, -1),
            int16_column("attlen", row_count, -1),
            (
                column("attnum", Type::Int16),
                Arc::new(Int16Array::from(
                    attributes
                        .iter()
                        .map(|attribute| attribute.position)
                        .collect::<Vec<_>>(),
                )),
            ),
            int32_column("attndims", row_count, 0),
            int32_column("attcacheoff", row_count, -1),
            (
                column("atttypmod", Type::Int64),
                Arc::new(Int64Array::from(
                    attributes
                        .iter()
                        .map(|attribute| type_modifier(&attribute.column.col_type))
                        .collect::<Vec<_>>(),
                )),
            ),
            bool_column("attbyval", row_count, false),
            string_column("attstorage", row_count, "p"),
            string_column("attalign", row_count, "i"),
            bool_column("attnotnull", row_count, false),
            bool_column("atthasdef", row_count, false),
            bool_column("atthasmissing", row_count, false),
            string_column("attidentity", row_count, ""),
            string_column("attgenerated", row_count, ""),
            bool_column("attisdropped", row_count, false),
            bool_column("attislocal", row_count, true),
            int32_column("attinhcount", row_count, 0),
            oid_column("attcollation", row_count, 0),
            string_column("attcompression", row_count, ""),
        ],
    )
}

fn build_pg_type(
    reference: &TableReference,
    transaction: &dyn DatastoreTransaction,
    oid_registry: &OidRegistry,
) -> VirtualTableBinding {
    let mut descriptors = BTreeMap::new();
    for table in transaction.tables() {
        for table_column in table.columns {
            let descriptor = describe_type(&table_column.col_type);
            descriptors.entry(descriptor.oid).or_insert(descriptor);
        }
    }
    let descriptors: Vec<_> = descriptors.into_values().collect();
    let row_count = descriptors.len();
    let schema_oid = oid_registry.schema_oid(&reference.datastore, SCHEMA_NAME);

    build_table(
        reference,
        vec![
            (
                column("oid", Type::UInt32),
                Arc::new(UInt32Array::from(
                    descriptors
                        .iter()
                        .map(|descriptor| descriptor.oid)
                        .collect::<Vec<_>>(),
                )),
            ),
            (
                column("typname", Type::Utf8),
                Arc::new(StringViewArray::from_iter_values(
                    descriptors
                        .iter()
                        .map(|descriptor| descriptor.name.as_str()),
                )),
            ),
            oid_column("typnamespace", row_count, schema_oid),
            oid_column("typowner", row_count, PIVOT_OWNER_OID),
            (
                column("typlen", Type::Int16),
                Arc::new(Int16Array::from(
                    descriptors
                        .iter()
                        .map(|descriptor| descriptor.length)
                        .collect::<Vec<_>>(),
                )),
            ),
            bool_column("typbyval", row_count, false),
            string_column("typtype", row_count, "b"),
            (
                column("typcategory", Type::Utf8),
                Arc::new(StringViewArray::from_iter_values(
                    descriptors.iter().map(|descriptor| descriptor.category),
                )),
            ),
            bool_column("typispreferred", row_count, false),
            bool_column("typisdefined", row_count, true),
            oid_column("typrelid", row_count, 0),
            oid_column("typelem", row_count, 0),
            oid_column("typarray", row_count, 0),
            oid_column("typcollation", row_count, 0),
            // Domain-type bookkeeping the JDBC driver's column metadata query
            // reads: none of pivot's types are domains.
            bool_column("typnotnull", row_count, false),
            oid_column("typbasetype", row_count, 0),
            int32_column("typtypmod", row_count, -1),
        ],
    )
}

fn build_pg_roles(
    reference: &TableReference,
    metastore: &dyn Metastore,
    oid_registry: &OidRegistry,
) -> VirtualTableBinding {
    let mut names = metastore.user_names();
    names.push(DEFAULT_USER_NAME.to_string());
    names.sort();
    names.dedup();
    let row_count = names.len();
    let oids: Vec<_> = names
        .iter()
        .map(|name| oid_registry.role_oid(name))
        .collect();

    build_table(
        reference,
        vec![
            (
                column("rolname", Type::Utf8),
                Arc::new(StringViewArray::from_iter_values(
                    names.iter().map(String::as_str),
                )),
            ),
            bool_column("rolsuper", row_count, false),
            bool_column("rolinherit", row_count, true),
            bool_column("rolcreaterole", row_count, false),
            bool_column("rolcreatedb", row_count, false),
            bool_column("rolcanlogin", row_count, true),
            bool_column("rolreplication", row_count, false),
            int32_column("rolconnlimit", row_count, -1),
            string_column("rolpassword", row_count, "********"),
            (
                column("rolvaliduntil", Type::Timestamp),
                new_null_array(&physical_arrow_type(&Type::Timestamp), row_count),
            ),
            bool_column("rolbypassrls", row_count, false),
            (
                column("rolconfig", Type::List(Box::new(Type::Utf8))),
                new_null_array(
                    &physical_arrow_type(&Type::List(Box::new(Type::Utf8))),
                    row_count,
                ),
            ),
            (
                column("oid", Type::UInt32),
                Arc::new(UInt32Array::from(oids)),
            ),
        ],
    )
}

fn build_empty(reference: &TableReference, columns: &[Column]) -> VirtualTableBinding {
    let data = columns
        .iter()
        .cloned()
        .map(|column| {
            let array = new_empty_array(&physical_arrow_type(&column.col_type));
            (column, array)
        })
        .collect();
    build_table(reference, data)
}

fn build_table(
    reference: &TableReference,
    columns: Vec<(Column, ArrayRef)>,
) -> VirtualTableBinding {
    let fields: Vec<_> = columns
        .iter()
        .map(|(column, _)| Field::new(&column.name, physical_arrow_type(&column.col_type), true))
        .collect();
    let arrays: Vec<_> = columns.iter().map(|(_, array)| array.clone()).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .expect("virtual catalog columns have equal lengths and declared physical types");
    VirtualTableBinding {
        reference: reference.clone(),
        revision: table_revision(&reference.datastore, &reference.table)
            .expect("only registered virtual relations are materialized"),
        columns: columns.into_iter().map(|(column, _)| column).collect(),
        batch: Arc::new(batch),
    }
}

fn column(name: &str, col_type: Type) -> Column {
    Column {
        name: name.to_string(),
        col_type,
    }
}

fn int16_column(name: &str, rows: usize, value: i16) -> (Column, ArrayRef) {
    (
        column(name, Type::Int16),
        Arc::new(Int16Array::from(vec![value; rows])),
    )
}

fn int32_column(name: &str, rows: usize, value: i32) -> (Column, ArrayRef) {
    (
        column(name, Type::Int32),
        Arc::new(Int32Array::from(vec![value; rows])),
    )
}

fn oid_column(name: &str, rows: usize, value: u32) -> (Column, ArrayRef) {
    (
        column(name, Type::UInt32),
        Arc::new(UInt32Array::from(vec![value; rows])),
    )
}

fn bool_column(name: &str, rows: usize, value: bool) -> (Column, ArrayRef) {
    (
        column(name, Type::Boolean),
        Arc::new(BooleanArray::from(vec![value; rows])),
    )
}

fn string_column(name: &str, rows: usize, value: &str) -> (Column, ArrayRef) {
    (
        column(name, Type::Utf8),
        Arc::new(StringViewArray::from(vec![value; rows])),
    )
}
