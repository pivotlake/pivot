//! Virtual `pg_catalog` system tables for client catalog introspection.
//!
//! A JDBC driver (e.g. the Kafka Connect sink) describes a table before it can
//! build an `INSERT` by running `DatabaseMetaData.getColumns`/`getTables`, which
//! are `SELECT`s joining `pg_catalog.pg_class`, `pg_namespace`, `pg_attribute`,
//! `pg_type`, `pg_attrdef` and `pg_description`. Pivot has no such stored tables,
//! so each is materialized on demand: when a query references one of these names
//! and no user table shadows it, the transaction hands back a
//! [`VirtualMetadataTable`] whose scan builds one `RecordBatch` describing the
//! snapshot's user tables and columns — the same `values_input` shape
//! [`metadata()`](super::metadata_function) uses, but wearing the Postgres
//! catalog's schema.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    CatalogTransaction, Column, DynamicScanPredicate, Error as CatalogError,
    Result as CatalogResult, Table,
};
use planner::types::Type;

use super::ParquetTransaction;

/// The schema every user table is reported under. `public` is a Postgres
/// client's default schema, so an unqualified table name matches.
const PUBLIC_OID: i64 = 2200;
/// First synthesized table oid; each user table gets `TABLE_OID_BASE + i` in
/// sorted-name order — stable across the separate `pg_class`/`pg_attribute`
/// scans so their join keys (`pg_class.oid` = `pg_attribute.attrelid`) line up.
const TABLE_OID_BASE: i64 = 16384;

/// Which system table a [`VirtualMetadataTable`] presents.
#[derive(Debug, Clone, Copy)]
pub(super) enum PgCatalogTable {
    Namespace,
    Class,
    Attribute,
    Type,
    Attrdef,
    Description,
}

/// Recognize a `pg_catalog` table name, or `None` for anything else.
pub(super) fn resolve(name: &str) -> Option<PgCatalogTable> {
    Some(match name {
        "pg_namespace" => PgCatalogTable::Namespace,
        "pg_class" => PgCatalogTable::Class,
        "pg_attribute" => PgCatalogTable::Attribute,
        "pg_type" => PgCatalogTable::Type,
        "pg_attrdef" => PgCatalogTable::Attrdef,
        "pg_description" => PgCatalogTable::Description,
        _ => return None,
    })
}

impl PgCatalogTable {
    /// The fixed column list, a superset of what the introspection queries read.
    fn schema(self) -> &'static [(&'static str, Type)] {
        match self {
            PgCatalogTable::Namespace => &[("oid", Type::Int64), ("nspname", Type::Utf8)],
            PgCatalogTable::Class => &[
                ("oid", Type::Int64),
                ("relname", Type::Utf8),
                ("relnamespace", Type::Int64),
                ("relkind", Type::Utf8),
            ],
            PgCatalogTable::Attribute => &[
                ("attrelid", Type::Int64),
                ("attname", Type::Utf8),
                ("attnum", Type::Int64),
                ("atttypid", Type::Int64),
                ("atttypmod", Type::Int64),
                ("attlen", Type::Int64),
                ("attnotnull", Type::Boolean),
                ("attisdropped", Type::Boolean),
                ("attidentity", Type::Utf8),
                ("attgenerated", Type::Utf8),
            ],
            PgCatalogTable::Type => &[
                ("oid", Type::Int64),
                ("typname", Type::Utf8),
                ("typtype", Type::Utf8),
                ("typnotnull", Type::Boolean),
                ("typtypmod", Type::Int64),
                ("typbasetype", Type::Int64),
            ],
            PgCatalogTable::Attrdef => &[
                ("adrelid", Type::Int64),
                ("adnum", Type::Int64),
                ("adbin", Type::Utf8),
            ],
            PgCatalogTable::Description => &[
                ("objoid", Type::Int64),
                ("objsubid", Type::Int64),
                ("classoid", Type::Int64),
                ("description", Type::Utf8),
            ],
        }
    }
}

/// A `pg_catalog` table materialized from the current snapshot.
#[derive(Debug, Clone)]
pub(super) struct VirtualMetadataTable {
    table: PgCatalogTable,
}

impl VirtualMetadataTable {
    pub(super) fn new(table: PgCatalogTable) -> Self {
        Self { table }
    }
}

impl Table for VirtualMetadataTable {
    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        _dynamic_filters: Vec<DynamicScanPredicate>,
        _emit_row_group_metadata: bool,
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        let transaction = transaction
            .as_any()
            .downcast_ref::<ParquetTransaction>()
            .ok_or_else(|| {
                CatalogError::Other("pg_catalog tables require a parquet-backed catalog".into())
            })?;
        let tables = transaction.snapshot_tables();
        let batch = self.build_batch(&tables);
        let projected = batch
            .project(projection.indices())
            .map_err(|e| CatalogError::Other(Box::new(e)))?;
        Ok(dispatch::values_input(dispatcher, [projected]).record_batches())
    }

    fn columns(&self) -> Vec<Column> {
        self.table
            .schema()
            .iter()
            .map(|(name, col_type)| Column {
                name: name.to_string(),
                col_type: col_type.clone(),
            })
            .collect()
    }

    fn clone_box(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }
}

impl VirtualMetadataTable {
    /// Build the full (unprojected) batch of rows for this table from the
    /// snapshot's `tables` (each `(name, columns)` in sorted-name order).
    fn build_batch(&self, tables: &[(String, Vec<Column>)]) -> RecordBatch {
        let columns: Vec<ArrayRef> = match self.table {
            PgCatalogTable::Namespace => {
                vec![int64(vec![PUBLIC_OID]), utf8(vec!["public".to_string()])]
            }
            PgCatalogTable::Class => {
                let n = tables.len();
                vec![
                    int64((0..n as i64).map(|i| TABLE_OID_BASE + i).collect()),
                    utf8(tables.iter().map(|(name, _)| name.clone()).collect()),
                    int64(vec![PUBLIC_OID; n]),
                    utf8(vec!["r".to_string(); n]),
                ]
            }
            PgCatalogTable::Attribute => {
                let mut attrelid = Vec::new();
                let mut attname = Vec::new();
                let mut attnum = Vec::new();
                let mut atttypid = Vec::new();
                let mut attlen = Vec::new();
                for (table_idx, (_, cols)) in tables.iter().enumerate() {
                    let oid = TABLE_OID_BASE + table_idx as i64;
                    for (col_idx, col) in cols.iter().enumerate() {
                        attrelid.push(oid);
                        attname.push(col.name.clone());
                        attnum.push(col_idx as i64 + 1);
                        atttypid.push(pg_type_oid(&col.col_type));
                        attlen.push(pg_type_len(&col.col_type));
                    }
                }
                let rows = attrelid.len();
                vec![
                    int64(attrelid),
                    utf8(attname),
                    int64(attnum),
                    int64(atttypid),
                    int64(vec![-1; rows]), // atttypmod: no modifier
                    int64(attlen),
                    boolean(vec![false; rows]),      // attnotnull
                    boolean(vec![false; rows]),      // attisdropped
                    utf8(vec![String::new(); rows]), // attidentity
                    utf8(vec![String::new(); rows]), // attgenerated
                ]
            }
            PgCatalogTable::Type => {
                let n = PG_TYPES.len();
                vec![
                    int64(PG_TYPES.iter().map(|(oid, _)| *oid).collect()),
                    utf8(PG_TYPES.iter().map(|(_, name)| name.to_string()).collect()),
                    utf8(vec!["b".to_string(); n]), // typtype: base type
                    boolean(vec![false; n]),        // typnotnull
                    int64(vec![-1; n]),             // typtypmod
                    int64(vec![0; n]),              // typbasetype
                ]
            }
            PgCatalogTable::Attrdef => vec![int64(vec![]), int64(vec![]), utf8(vec![])],
            PgCatalogTable::Description => {
                vec![int64(vec![]), int64(vec![]), int64(vec![]), utf8(vec![])]
            }
        };
        let fields: Vec<Field> = self
            .table
            .schema()
            .iter()
            .map(|(name, col_type)| Field::new(*name, arrow_type(col_type), true))
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .expect("pg_catalog columns are equal-length and match the declared schema")
    }
}

/// The Postgres type oid a pivot column reports as `pg_attribute.atttypid` — the
/// driver derives the JDBC type, name and size from it. Every value listed here
/// also appears in [`PG_TYPES`] so a `pg_type` join finds it.
fn pg_type_oid(col_type: &Type) -> i64 {
    match col_type {
        Type::Boolean => 16,
        Type::Int8 | Type::Int16 => 21,       // int2
        Type::Int32 => 23,                    // int4
        Type::Int64 => 20,                    // int8
        Type::Float32 => 700,                 // float4
        Type::Float64 | Type::Decimal => 701, // float8
        Type::Date => 1082,
        _ => 25, // text
    }
}

/// `pg_attribute.attlen`: a fixed-width type's byte width, or `-1` for varlena.
fn pg_type_len(col_type: &Type) -> i64 {
    match col_type {
        Type::Boolean => 1,
        Type::Int8 | Type::Int16 => 2,
        Type::Int32 | Type::Float32 => 4,
        Type::Int64 | Type::Float64 | Type::Decimal | Type::Date => 8,
        _ => -1,
    }
}

/// The base types `pg_type` reports, `(oid, typname)`. Kept in sync with
/// [`pg_type_oid`] so every column's `atttypid` resolves.
const PG_TYPES: &[(i64, &str)] = &[
    (16, "bool"),
    (20, "int8"),
    (21, "int2"),
    (23, "int4"),
    (25, "text"),
    (700, "float4"),
    (701, "float8"),
    (1082, "date"),
    (1043, "varchar"),
];

/// The Arrow type a declared pivot [`Type`] materializes as in these batches.
fn arrow_type(col_type: &Type) -> DataType {
    match col_type {
        Type::Boolean => DataType::Boolean,
        Type::Utf8 => DataType::Utf8View,
        _ => DataType::Int64,
    }
}

fn int64(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn boolean(values: Vec<bool>) -> ArrayRef {
    Arc::new(BooleanArray::from(values))
}

fn utf8(values: Vec<String>) -> ArrayRef {
    Arc::new(StringViewArray::from(
        values.iter().map(String::as_str).collect::<Vec<_>>(),
    ))
}
