//! PostgreSQL catalog type identities used by virtual `pg_catalog` relations.

use crate::types::Type;

pub const BOOLEAN_OID: u32 = 16;
pub const INT64_OID: u32 = 20;
pub const INT16_OID: u32 = 21;
pub const INT32_OID: u32 = 23;
pub const TEXT_OID: u32 = 25;
pub const FLOAT32_OID: u32 = 700;
pub const FLOAT64_OID: u32 = 701;
pub const DATE_OID: u32 = 1082;
pub const TIMESTAMP_OID: u32 = 1114;
pub const NUMERIC_OID: u32 = 1700;
pub const JSONB_OID: u32 = 3802;

/// The synthetic role that owns every object exposed through Pivot's virtual
/// PostgreSQL catalogs. Datastore metadata does not persist object owners;
/// this conventional bootstrap OID exists only for client compatibility.
pub const PIVOT_OWNER_OID: u32 = 10;
pub const PIVOT_OWNER_NAME: &str = "pivot";

const BOOLEAN_ARRAY_OID: u32 = 1000;
const INT16_ARRAY_OID: u32 = 1005;
const INT32_ARRAY_OID: u32 = 1007;
const TEXT_ARRAY_OID: u32 = 1009;
const INT64_ARRAY_OID: u32 = 1016;
const FLOAT32_ARRAY_OID: u32 = 1021;
const FLOAT64_ARRAY_OID: u32 = 1022;
const TIMESTAMP_ARRAY_OID: u32 = 1115;
const DATE_ARRAY_OID: u32 = 1182;
const NUMERIC_ARRAY_OID: u32 = 1231;
const JSONB_ARRAY_OID: u32 = 3807;

/// Virtual object IDs occupy a disjoint part of PostgreSQL's unsigned 32-bit
/// OID range. Relation IDs reserve one bit to carry whether the relation is on
/// Pivot's fixed search path, so `pg_table_is_visible` can answer from the OID.
pub const VIRTUAL_OID_PREFIX: u32 = 1_u32 << 31;
pub const VISIBLE_RELATION_OID_FLAG: u32 = 1_u32 << 30;

const INT8_OID: u32 = 91_001;
const UINT8_OID: u32 = 91_002;
const UINT16_OID: u32 = 91_003;
const UINT32_OID: u32 = 91_004;
const UINT64_OID: u32 = 91_005;
const INT128_OID: u32 = 91_006;
const INT8_ARRAY_OID: u32 = 92_001;
const UINT8_ARRAY_OID: u32 = 92_002;
const UINT16_ARRAY_OID: u32 = 92_003;
const UINT32_ARRAY_OID: u32 = 92_004;
const UINT64_ARRAY_OID: u32 = 92_005;
const INT128_ARRAY_OID: u32 = 92_006;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PgTypeDescriptor {
    pub oid: u32,
    pub name: String,
    pub length: i16,
    pub category: &'static str,
}

pub fn type_oid(data_type: &Type) -> u32 {
    match data_type {
        Type::Boolean => BOOLEAN_OID,
        Type::Int8 => INT8_OID,
        Type::Int16 => INT16_OID,
        Type::Int32 => INT32_OID,
        Type::Int64 => INT64_OID,
        Type::UInt8 => UINT8_OID,
        Type::UInt16 => UINT16_OID,
        Type::UInt32 => UINT32_OID,
        Type::UInt64 => UINT64_OID,
        Type::Int128 => INT128_OID,
        Type::Float32 => FLOAT32_OID,
        Type::Float64 => FLOAT64_OID,
        Type::Decimal { .. } => NUMERIC_OID,
        Type::Utf8 => TEXT_OID,
        Type::Date => DATE_OID,
        Type::Timestamp => TIMESTAMP_OID,
        Type::List(child) => array_type_oid(child),
        Type::Variant => JSONB_OID,
    }
}

fn array_type_oid(element: &Type) -> u32 {
    match element {
        Type::Boolean => BOOLEAN_ARRAY_OID,
        Type::Int8 => INT8_ARRAY_OID,
        Type::Int16 => INT16_ARRAY_OID,
        Type::Int32 => INT32_ARRAY_OID,
        Type::Int64 => INT64_ARRAY_OID,
        Type::UInt8 => UINT8_ARRAY_OID,
        Type::UInt16 => UINT16_ARRAY_OID,
        Type::UInt32 => UINT32_ARRAY_OID,
        Type::UInt64 => UINT64_ARRAY_OID,
        Type::Int128 => INT128_ARRAY_OID,
        Type::Float32 => FLOAT32_ARRAY_OID,
        Type::Float64 => FLOAT64_ARRAY_OID,
        Type::Decimal { .. } => NUMERIC_ARRAY_OID,
        Type::Utf8 => TEXT_ARRAY_OID,
        Type::Date => DATE_ARRAY_OID,
        Type::Timestamp => TIMESTAMP_ARRAY_OID,
        // PostgreSQL's array type does not encode dimensionality.
        Type::List(child) => array_type_oid(child),
        Type::Variant => JSONB_ARRAY_OID,
    }
}

/// The catalog modifier passed back to [`format_type`]. A negative value means
/// the type has no modifier. Decimal precision and scale use a private encoding
/// because Pivot evaluates `format_type` itself.
pub fn type_modifier(data_type: &Type) -> i64 {
    match data_type {
        Type::Decimal { precision, scale } => i64::from(*precision) * 1_000 + i64::from(*scale),
        _ => -1,
    }
}

pub fn describe_type(data_type: &Type) -> PgTypeDescriptor {
    let (name, length, category) = match data_type {
        Type::Boolean => ("bool", 1, "B"),
        Type::Int8 => ("tinyint", 1, "N"),
        Type::Int16 => ("int2", 2, "N"),
        Type::Int32 => ("int4", 4, "N"),
        Type::Int64 => ("int8", 8, "N"),
        Type::UInt8 => ("utinyint", 1, "N"),
        Type::UInt16 => ("usmallint", 2, "N"),
        Type::UInt32 => ("uinteger", 4, "N"),
        Type::UInt64 => ("ubigint", 8, "N"),
        Type::Int128 => ("hugeint", 16, "N"),
        Type::Float32 => ("float4", 4, "N"),
        Type::Float64 => ("float8", 8, "N"),
        Type::Decimal { .. } => ("numeric", -1, "N"),
        Type::Utf8 => ("text", -1, "S"),
        Type::Date => ("date", 4, "D"),
        Type::Timestamp => ("timestamp", 8, "D"),
        Type::List(child) => {
            let element = describe_type(child);
            return PgTypeDescriptor {
                oid: type_oid(data_type),
                name: format!("_{}", element.name),
                length: -1,
                category: "A",
            };
        }
        Type::Variant => ("jsonb", -1, "U"),
    };
    PgTypeDescriptor {
        oid: type_oid(data_type),
        name: name.to_string(),
        length,
        category,
    }
}

pub fn format_type(oid: u32, modifier: i64) -> String {
    match oid {
        BOOLEAN_OID => "boolean".to_string(),
        INT8_OID => "tinyint".to_string(),
        INT16_OID => "smallint".to_string(),
        INT32_OID => "integer".to_string(),
        INT64_OID => "bigint".to_string(),
        UINT8_OID => "utinyint".to_string(),
        UINT16_OID => "usmallint".to_string(),
        UINT32_OID => "uinteger".to_string(),
        UINT64_OID => "ubigint".to_string(),
        INT128_OID => "hugeint".to_string(),
        FLOAT32_OID => "real".to_string(),
        FLOAT64_OID => "double precision".to_string(),
        NUMERIC_OID if modifier >= 0 => {
            format!("numeric({},{})", modifier / 1_000, modifier % 1_000)
        }
        NUMERIC_OID => "numeric".to_string(),
        TEXT_OID => "text".to_string(),
        DATE_OID => "date".to_string(),
        TIMESTAMP_OID => "timestamp without time zone".to_string(),
        JSONB_OID => "jsonb".to_string(),
        BOOLEAN_ARRAY_OID => "boolean[]".to_string(),
        INT8_ARRAY_OID => "tinyint[]".to_string(),
        INT16_ARRAY_OID => "smallint[]".to_string(),
        INT32_ARRAY_OID => "integer[]".to_string(),
        INT64_ARRAY_OID => "bigint[]".to_string(),
        UINT8_ARRAY_OID => "utinyint[]".to_string(),
        UINT16_ARRAY_OID => "usmallint[]".to_string(),
        UINT32_ARRAY_OID => "uinteger[]".to_string(),
        UINT64_ARRAY_OID => "ubigint[]".to_string(),
        INT128_ARRAY_OID => "hugeint[]".to_string(),
        FLOAT32_ARRAY_OID => "real[]".to_string(),
        FLOAT64_ARRAY_OID => "double precision[]".to_string(),
        NUMERIC_ARRAY_OID => "numeric[]".to_string(),
        TEXT_ARRAY_OID => "text[]".to_string(),
        DATE_ARRAY_OID => "date[]".to_string(),
        TIMESTAMP_ARRAY_OID => "timestamp without time zone[]".to_string(),
        JSONB_ARRAY_OID => "jsonb[]".to_string(),
        _ => "???".to_string(),
    }
}

pub fn mark_relation_oid_visible(oid: u32) -> u32 {
    oid | VISIBLE_RELATION_OID_FLAG
}

pub fn relation_oid_is_visible(oid: u32) -> bool {
    oid & VISIBLE_RELATION_OID_FLAG != 0
}
