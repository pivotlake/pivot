//! The SQL-type ↔ Pivot-type mapping used when a table's declared schema is
//! written to (and read back from) the table manifest. The spellings follow
//! DuckDB's, so they round-trip a column's type through the manifest as text.

use planner::types::Type;

/// Map a SQL type spelling (as recorded in a manifest column's `type`) to a
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
/// manifest column (the inverse of [`sql_type_to_pivot`]).
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
}
