use super::general::{Encoding, PageType};
use super::parquet_thrift::*;
use crate::{general_err, thrift_struct};
use std::io::Write;

// LogicalType is a thrift union where most variants are empty structs.
// We only care about String (id=1) and Integer (id=10).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum LogicalType {
    String,
    Integer { bit_width: i8, is_signed: bool },
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

// Stub WriteThrift/WriteThriftField impls needed to satisfy thrift_struct! bounds
impl WriteThrift for LogicalType {
    const ELEMENT_TYPE: ElementType = ElementType::Struct;

    fn write_thrift<W: Write>(&self, _writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        unimplemented!("LogicalType serialization not needed")
    }
}

impl WriteThriftField for LogicalType {
    fn write_thrift_field<W: Write>(
        &self,
        _writer: &mut ThriftCompactOutputProtocol<W>,
        _field_id: i16,
        _last_field_id: i16,
    ) -> Result<i16> {
        unimplemented!("LogicalType serialization not needed")
    }
}

thrift_struct!(
    pub(crate) struct SchemaElement {
        1: optional i32 physical_type;
        3: optional i32 repetition_type;
        4: required string name;
        5: optional i32 num_children;
        6: optional i32 converted_type;
        10: optional LogicalType logical_type;
    }
);

thrift_struct!(
    pub(crate) struct Statistics {
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
    pub(crate) struct PageEncodingStats {
        1: required PageType page_type;
        2: required Encoding encoding;
        3: required i32 count;
    }
);

thrift_struct!(
    pub(crate) struct ColumnMetaData {
        7: required i64 total_compressed_size;
        9: required i64 data_page_offset;
        11: optional i64 dictionary_page_offset;
        12: optional Statistics statistics;
        13: optional list<PageEncodingStats> encoding_stats;
    }
);

thrift_struct!(
    pub(crate) struct ColumnChunk {
        3: optional ColumnMetaData meta_data;
    }
);

thrift_struct!(
    pub(crate) struct RowGroup {
        1: required list<ColumnChunk> columns;
        3: required i64 num_rows;
    }
);

thrift_struct!(
    pub(crate) struct FileMetaData {
        2: required list<SchemaElement> schema;
        4: required list<RowGroup> row_groups;
    }
);
