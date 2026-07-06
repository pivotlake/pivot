use super::general::{Encoding, PageType, TimeUnit};
use super::parquet_thrift::*;
use crate::{general_err, thrift_struct};
use std::io::Write;

// LogicalType is a thrift union where most variants are empty structs. We model
// the ones whose arrow type pivot decodes natively: String, Integer, Date,
// Timestamp, and Variant; everything else is `Other`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogicalType {
    String,
    Integer {
        bit_width: i8,
        is_signed: bool,
    },
    /// `DATE`: an `INT32` of days since the epoch.
    Date,
    /// `TIMESTAMP`: an `INT64` of `unit` since the epoch.
    Timestamp {
        unit: TimeUnit,
        is_adjusted_to_utc: bool,
    },
    /// Marks a group as a Parquet `variant` (its `metadata`/`value` children are
    /// binary). The writer emits it; the reader uses it to read those leaves as
    /// binary rather than the default string.
    Variant,
    Other,
}

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for LogicalType {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        let field_ident = prot.read_field_begin(0)?;
        if field_ident.field_type == FieldType::Stop {
            return Err(general_err!("Empty LogicalType union"));
        }
        let result = match field_ident.id {
            // STRING: empty struct
            1 => {
                prot.skip_empty_struct()?;
                LogicalType::String
            }
            // INTEGER: struct { 1: i8 bitWidth, 2: bool isSigned }
            10 => {
                let mut bit_width: i8 = 0;
                let mut is_signed: bool = false;
                let mut last_field_id = 0i16;
                loop {
                    let fi = prot.read_field_begin(last_field_id)?;
                    if fi.field_type == FieldType::Stop {
                        break;
                    }
                    match fi.id {
                        1 => bit_width = prot.read_i8()?,
                        2 => is_signed = fi.bool_val.unwrap_or(false),
                        _ => prot.skip(fi.field_type)?,
                    }
                    last_field_id = fi.id;
                }
                LogicalType::Integer {
                    bit_width,
                    is_signed,
                }
            }
            // DATE: empty struct
            6 => {
                prot.skip_empty_struct()?;
                LogicalType::Date
            }
            // TIMESTAMP: struct { 1: bool isAdjustedToUTC, 2: TimeUnit unit }
            8 => {
                let mut is_adjusted_to_utc = false;
                let mut unit = TimeUnit::MILLIS;
                let mut last_field_id = 0i16;
                loop {
                    let fi = prot.read_field_begin(last_field_id)?;
                    if fi.field_type == FieldType::Stop {
                        break;
                    }
                    match fi.id {
                        1 => is_adjusted_to_utc = fi.bool_val.unwrap_or(false),
                        2 => unit = TimeUnit::read_thrift(prot)?,
                        _ => prot.skip(fi.field_type)?,
                    }
                    last_field_id = fi.id;
                }
                LogicalType::Timestamp {
                    unit,
                    is_adjusted_to_utc,
                }
            }
            // VARIANT: a (possibly empty) VariantType struct we don't read into.
            16 => {
                prot.skip(field_ident.field_type)?;
                LogicalType::Variant
            }
            // Any other variant — skip it
            _ => {
                prot.skip(field_ident.field_type)?;
                LogicalType::Other
            }
        };
        // Read the stop field for the union struct
        let fi = prot.read_field_begin(field_ident.id)?;
        if fi.field_type != FieldType::Stop {
            prot.skip(fi.field_type)?;
            // Drain remaining fields
            let mut last = fi.id;
            loop {
                let fi = prot.read_field_begin(last)?;
                if fi.field_type == FieldType::Stop {
                    break;
                }
                prot.skip(fi.field_type)?;
                last = fi.id;
            }
        }
        Ok(result)
    }
}

impl WriteThrift for LogicalType {
    const ELEMENT_TYPE: ElementType = ElementType::Struct;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        match self {
            // VARIANT union member: field 16, an (empty) VariantType struct.
            LogicalType::Variant => {
                writer.write_empty_struct(16, 0)?;
            }
            other => unimplemented!("LogicalType serialization for {other:?} not implemented"),
        }
        writer.write_struct_end()
    }
}

impl WriteThriftField for LogicalType {
    fn write_thrift_field<W: Write>(
        &self,
        writer: &mut ThriftCompactOutputProtocol<W>,
        field_id: i16,
        last_field_id: i16,
    ) -> Result<i16> {
        writer.write_field_begin(FieldType::Struct, field_id, last_field_id)?;
        self.write_thrift(writer)?;
        Ok(field_id)
    }
}

thrift_struct!(
    pub struct SchemaElement {
        1: optional i32 physical_type;
        3: optional i32 repetition_type;
        4: required string name;
        5: optional i32 num_children;
        6: optional i32 converted_type;
        10: optional LogicalType logical_type;
    }
);

thrift_struct!(
    pub struct Statistics {
        1: optional binary max;
        2: optional binary min;
        3: optional i64 null_count;
        4: optional i64 distinct_count;
        5: optional binary max_value;
        6: optional binary min_value;
    }
);

thrift_struct!(
    /// Per-(page-type, encoding) page counts for a column chunk. Lets a reader
    /// tell, without scanning the data, whether every data page is dictionary
    /// encoded — the precondition for soundly pruning a row group by its
    /// dictionary contents.
    pub struct PageEncodingStats {
        1: required PageType page_type;
        2: required Encoding encoding;
        3: required i32 count;
    }
);

thrift_struct!(
    pub struct ColumnMetaData {
        1: required i32 physical_type;
        2: required list<i32> encodings;
        3: required list<string> path_in_schema;
        4: required i32 codec;
        5: required i64 num_values;
        6: required i64 total_uncompressed_size;
        7: required i64 total_compressed_size;
        9: required i64 data_page_offset;
        11: optional i64 dictionary_page_offset;
        12: optional Statistics statistics;
        13: optional list<PageEncodingStats> encoding_stats;
    }
);

thrift_struct!(
    pub struct ColumnChunk {
        2: required i64 file_offset;
        3: optional ColumnMetaData meta_data;
    }
);

thrift_struct!(
    pub struct RowGroup {
        1: required list<ColumnChunk> columns;
        2: required i64 total_byte_size;
        3: required i64 num_rows;
    }
);

thrift_struct!(
    pub struct FileMetaData {
        1: required i32 version;
        2: required list<SchemaElement> schema;
        3: required i64 num_rows;
        4: required list<RowGroup> row_groups;
        6: optional string created_by;
    }
);
