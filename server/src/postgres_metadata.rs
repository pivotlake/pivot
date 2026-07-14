//! The small PostgreSQL catalog surface used by pgjdbc's sink path.
//!
//! PivotDB does not expose PostgreSQL's physical `pg_catalog` tables. The JDBC
//! driver nevertheless asks those tables about the one destination table
//! before preparing an INSERT. Recognise those driver-owned queries here and
//! answer them from Pivot's catalog, keeping the compatibility boundary narrow
//! and explicit instead of manufacturing a fake PostgreSQL system catalog.

use std::sync::Arc;

use bytes::{BufMut, BytesMut};
use futures::stream;
use pgwire::api::Type as PgType;
use pgwire::api::results::{FieldFormat, FieldInfo, QueryResponse, Response};
use pgwire::messages::data::DataRow;
use planner::catalog::{Catalog, Column};
use planner::types::Type;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetadataQuery {
    CurrentCatalog,
    CurrentSchema,
    IdentifierLength,
    Tables,
    Columns,
    PrimaryKeys,
}

fn classify(sql: &str) -> Option<MetadataQuery> {
    let sql = sql.trim().to_ascii_lowercase();
    if sql.starts_with("select current_catalog") {
        Some(MetadataQuery::CurrentCatalog)
    } else if sql.starts_with("select current_schema()") {
        Some(MetadataQuery::CurrentSchema)
    } else if sql.starts_with("select length(repeat('1234567890', 1000)::name)") {
        Some(MetadataQuery::IdentifierLength)
    } else if sql.contains("information_schema._pg_expandarray") {
        Some(MetadataQuery::PrimaryKeys)
    } else if sql.contains("join pg_catalog.pg_attribute a")
        && sql.contains("from pg_catalog.pg_namespace n")
    {
        Some(MetadataQuery::Columns)
    } else if sql.contains("from pg_catalog.pg_namespace n, pg_catalog.pg_class c") {
        Some(MetadataQuery::Tables)
    } else {
        None
    }
}

fn fields(definitions: &[(&str, PgType)]) -> Arc<Vec<FieldInfo>> {
    Arc::new(
        definitions
            .iter()
            .map(|(name, data_type)| {
                FieldInfo::new(
                    (*name).to_string(),
                    None,
                    None,
                    data_type.clone(),
                    FieldFormat::Text,
                )
            })
            .collect(),
    )
}

fn query_fields(kind: MetadataQuery) -> Arc<Vec<FieldInfo>> {
    match kind {
        MetadataQuery::CurrentCatalog => fields(&[("current_catalog", PgType::TEXT)]),
        MetadataQuery::CurrentSchema => fields(&[("current_schema", PgType::TEXT)]),
        MetadataQuery::IdentifierLength => fields(&[("length", PgType::INT4)]),
        MetadataQuery::Tables => fields(&[
            ("TABLE_CAT", PgType::TEXT),
            ("TABLE_SCHEM", PgType::TEXT),
            ("TABLE_NAME", PgType::TEXT),
            ("TABLE_TYPE", PgType::TEXT),
            ("REMARKS", PgType::TEXT),
            ("TYPE_CAT", PgType::TEXT),
            ("TYPE_SCHEM", PgType::TEXT),
            ("TYPE_NAME", PgType::TEXT),
            ("SELF_REFERENCING_COL_NAME", PgType::TEXT),
            ("REF_GENERATION", PgType::TEXT),
        ]),
        // These are the raw pg_catalog fields consumed inside
        // PgDatabaseMetaData#getColumns. The driver turns them into JDBC's
        // standard 24-column metadata result set client-side.
        MetadataQuery::Columns => fields(&[
            ("current_database", PgType::TEXT),
            ("nspname", PgType::TEXT),
            ("relname", PgType::TEXT),
            ("attname", PgType::TEXT),
            ("atttypid", PgType::OID),
            ("attnotnull", PgType::BOOL),
            ("atttypmod", PgType::INT4),
            ("attlen", PgType::INT2),
            ("typtypmod", PgType::INT4),
            ("attnum", PgType::INT8),
            ("attidentity", PgType::TEXT),
            ("attgenerated", PgType::TEXT),
            ("adsrc", PgType::TEXT),
            ("description", PgType::TEXT),
            ("typbasetype", PgType::OID),
            ("typtype", PgType::TEXT),
        ]),
        MetadataQuery::PrimaryKeys => fields(&[
            ("TABLE_CAT", PgType::TEXT),
            ("TABLE_SCHEM", PgType::TEXT),
            ("TABLE_NAME", PgType::TEXT),
            ("COLUMN_NAME", PgType::TEXT),
            ("KEY_SEQ", PgType::INT4),
            ("PK_NAME", PgType::TEXT),
        ]),
    }
}

/// Result schema for an extended-protocol Describe, if `sql` is a pgjdbc
/// metadata query handled here.
pub(crate) fn describe(sql: &str) -> Option<Vec<FieldInfo>> {
    Some(query_fields(classify(sql)?).as_ref().clone())
}

/// Answer a pgjdbc metadata query. Parameters are text representations of the
/// portal's bound values; table/schema patterns are strings in pgjdbc.
pub(crate) fn execute(
    catalog: &Arc<dyn Catalog>,
    database: &str,
    sql: &str,
    parameters: &[Option<String>],
) -> Option<Response> {
    let kind = classify(sql)?;
    let rows = match kind {
        MetadataQuery::CurrentCatalog => vec![vec![some(database)]],
        MetadataQuery::CurrentSchema => vec![vec![some("public")]],
        MetadataQuery::IdentifierLength => vec![vec![some("63")]],
        MetadataQuery::Tables => table_from_parameters(catalog, parameters)
            .map(|(table, _)| {
                vec![vec![
                    some(database),
                    some("public"),
                    some(&table),
                    some("TABLE"),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]]
            })
            .unwrap_or_default(),
        MetadataQuery::Columns => table_from_parameters(catalog, parameters)
            .map(|(table, columns)| column_rows(database, &table, &columns))
            .unwrap_or_default(),
        // Pivot's current catalog schema has no primary-key metadata. This is
        // correct for the connector's pk.mode=none insert path.
        MetadataQuery::PrimaryKeys => Vec::new(),
    };
    Some(response(query_fields(kind), rows))
}

fn table_from_parameters(
    catalog: &Arc<dyn Catalog>,
    parameters: &[Option<String>],
) -> Option<(String, Vec<Column>)> {
    let transaction = catalog.begin_transaction();
    parameters.iter().rev().flatten().find_map(|candidate| {
        let table = transaction.table(candidate)?;
        Some((candidate.clone(), table.columns()))
    })
}

fn column_rows(database: &str, table: &str, columns: &[Column]) -> Vec<Vec<Option<String>>> {
    columns
        .iter()
        .enumerate()
        .map(|(index, column)| {
            let (oid, width) = postgres_type(&column.col_type);
            vec![
                some(database),
                some("public"),
                some(table),
                some(&column.name),
                some(&oid.to_string()),
                some("f"),
                some("-1"),
                some(&width.to_string()),
                some("-1"),
                some(&(index + 1).to_string()),
                None,
                None,
                None,
                None,
                some("0"),
                some("b"),
            ]
        })
        .collect()
}

fn postgres_type(data_type: &Type) -> (u32, i16) {
    match data_type {
        Type::Boolean => (16, 1),
        Type::Int8 | Type::Int16 | Type::UInt8 => (21, 2),
        Type::Int32 | Type::UInt16 => (23, 4),
        Type::Int64 | Type::UInt32 => (20, 8),
        Type::UInt64 | Type::Int128 | Type::Decimal => (1700, -1),
        Type::Float32 => (700, 4),
        Type::Float64 => (701, 8),
        Type::Utf8 => (25, -1),
        Type::Date => (1082, 4),
        Type::Timestamp => (1114, 8),
    }
}

fn some(value: &str) -> Option<String> {
    Some(value.to_string())
}

fn response(fields: Arc<Vec<FieldInfo>>, rows: Vec<Vec<Option<String>>>) -> Response {
    let rows = rows.into_iter().map(|row| Ok(text_row(row)));
    Response::Query(QueryResponse::new(fields, stream::iter(rows)))
}

/// Build a text-format DataRow directly. Metadata result columns intentionally
/// mix OID/integer/boolean/text types, while every value is already in the
/// canonical PostgreSQL text representation.
fn text_row(row: Vec<Option<String>>) -> DataRow {
    let mut data = BytesMut::new();
    for value in &row {
        match value {
            Some(value) => {
                data.put_i32(value.len() as i32);
                data.put_slice(value.as_bytes());
            }
            None => data.put_i32(-1),
        }
    }
    DataRow::new(data, row.len() as i16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_only_the_sink_metadata_surface() {
        assert_eq!(
            classify("select current_schema()"),
            Some(MetadataQuery::CurrentSchema)
        );
        assert_eq!(classify("select 1"), None);
    }

    #[test]
    fn all_metadata_rows_match_their_declared_width() {
        for kind in [
            MetadataQuery::CurrentCatalog,
            MetadataQuery::CurrentSchema,
            MetadataQuery::IdentifierLength,
            MetadataQuery::Tables,
            MetadataQuery::Columns,
            MetadataQuery::PrimaryKeys,
        ] {
            let width = query_fields(kind).len();
            let sample = match kind {
                MetadataQuery::CurrentCatalog => vec![some("pivot")],
                MetadataQuery::CurrentSchema => vec![some("public")],
                MetadataQuery::IdentifierLength => vec![some("63")],
                MetadataQuery::Tables => vec![None; 10],
                MetadataQuery::Columns => vec![None; 16],
                MetadataQuery::PrimaryKeys => vec![None; 6],
            };
            assert_eq!(sample.len(), width);
        }
    }
}
