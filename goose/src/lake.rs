//! Lake tables: catalog tables whose schema and data-file list come from a
//! goose object-store snapshot (`_goose_log/`) instead of a raw local directory.
//!
//! The catalog-root URL passed to `CREATE TABLE ... WITH (url = '…')` identifies
//! a single table. Resolving it loads the latest snapshot, reads that table's
//! recorded columns and data files, and builds a [`ParquetTable`] over them.
//!
//! This module holds the pure pieces — the SQL-type ↔ Pivot-type mapping (the
//! snapshot's type contract) and data-file location resolution; the catalog
//! wiring that uses them lives in `lib.rs`.
//!
//! [`ParquetTable`]: crate::parquet::ParquetTable

use planner::types::Type;
use std::path::PathBuf;

/// Map a SQL type spelling (as recorded in a snapshot column's `type`) to a
/// Pivot [`Type`]. Case-insensitive; precision/scale suffixes (`DECIMAL(18,2)`)
/// and modifiers are ignored. Returns `None` for an unrecognized type.
///
/// Spellings follow DuckDB's, including its byte-width aliases — note `INT8` is
/// `BIGINT` (8 bytes) and `INT16` is `HUGEINT` (16 bytes), *not* Rust's `i8`/`i16`.
pub fn sql_type_to_pivot(sql: &str) -> Option<Type> {
    let base = sql
        .split(['(', ' '])
        .next()
        .unwrap_or(sql)
        .trim()
        .to_ascii_uppercase();
    Some(match base.as_str() {
        "BOOLEAN" | "BOOL" | "LOGICAL" => Type::Boolean,
        "TINYINT" | "INT1" => Type::Int8,
        "SMALLINT" | "INT2" => Type::Int16,
        "INTEGER" | "INT" | "INT4" | "SIGNED" => Type::Int32,
        "BIGINT" | "INT8" | "LONG" => Type::Int64,
        "HUGEINT" | "INT16" => Type::Int128,
        "DOUBLE" | "FLOAT8" | "REAL" | "FLOAT" | "FLOAT4" => Type::Float64,
        "DECIMAL" | "NUMERIC" => Type::Decimal,
        "VARCHAR" | "STRING" | "TEXT" | "CHAR" | "BPCHAR" => Type::Utf8,
        "DATE" => Type::Date,
        "TIMESTAMP" | "DATETIME" => Type::Timestamp,
        _ => return None,
    })
}

/// The canonical SQL spelling for a Pivot [`Type`], used when *writing* a
/// snapshot column (the inverse of [`sql_type_to_pivot`]).
pub fn pivot_type_to_sql(ty: &Type) -> &'static str {
    match ty {
        Type::Boolean => "BOOLEAN",
        Type::Int8 => "TINYINT",
        Type::Int16 => "SMALLINT",
        Type::Int32 => "INTEGER",
        Type::Int64 => "BIGINT",
        Type::Int128 => "HUGEINT",
        Type::Float64 => "DOUBLE",
        Type::Decimal => "DECIMAL",
        Type::Utf8 => "VARCHAR",
        Type::Date => "DATE",
        Type::Timestamp => "TIMESTAMP",
    }
}

/// Whether a snapshot data-file location lives on remote object storage.
pub fn is_remote_location(loc: &str) -> bool {
    loc.starts_with("s3://") || loc.starts_with("s3a://") || loc.starts_with("gs://")
}

/// The local filesystem path for a data-file location, or `None` if it is
/// remote (`s3://`/`gs://`). A relative location is resolved against `root`
/// (the catalog root); an absolute one (`/…`, `file://…`) is used directly.
pub fn local_path(root: &str, loc: &str) -> Option<PathBuf> {
    if is_remote_location(loc) {
        return None;
    }
    let loc = loc.strip_prefix("file://").unwrap_or(loc);
    if loc.starts_with('/') {
        Some(PathBuf::from(loc))
    } else {
        let root = root.strip_prefix("file://").unwrap_or(root);
        Some(PathBuf::from(root).join(loc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sql_type_mapping_round_trips_and_is_case_insensitive() {
        for ty in [
            Type::Boolean,
            Type::Int8,
            Type::Int16,
            Type::Int32,
            Type::Int64,
            Type::Int128,
            Type::Float64,
            Type::Decimal,
            Type::Utf8,
            Type::Date,
            Type::Timestamp,
        ] {
            let sql = pivot_type_to_sql(&ty);
            assert_eq!(sql_type_to_pivot(sql), Some(ty.clone()), "round trip {sql}");
            assert_eq!(sql_type_to_pivot(&sql.to_ascii_lowercase()), Some(ty));
        }
        assert_eq!(sql_type_to_pivot("DECIMAL(18,2)"), Some(Type::Decimal));
        assert_eq!(sql_type_to_pivot("nonsense"), None);
        // DuckDB byte-width aliases, easy to get backwards.
        assert_eq!(sql_type_to_pivot("INT8"), Some(Type::Int64));
        assert_eq!(sql_type_to_pivot("INT16"), Some(Type::Int128));
    }

    #[test]
    fn location_resolution() {
        assert!(is_remote_location("s3://b/k.parquet"));
        assert!(is_remote_location("gs://b/k.parquet"));
        assert!(!is_remote_location("/data/k.parquet"));

        assert_eq!(local_path("s3://b/x", "data/f.parquet"), None);
        assert_eq!(
            local_path("/lake", "_goose_data/f.parquet"),
            Some(PathBuf::from("/lake/_goose_data/f.parquet"))
        );
        assert_eq!(
            local_path("/lake", "/abs/f.parquet"),
            Some(PathBuf::from("/abs/f.parquet"))
        );
        assert_eq!(
            local_path("file:///lake", "f.parquet"),
            Some(PathBuf::from("/lake/f.parquet"))
        );
    }
}
