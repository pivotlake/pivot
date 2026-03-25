//! Thrift page header types for Parquet files.

use super::general::{Encoding, PageType};
use super::parquet_thrift::*;
use crate::{general_err, thrift_struct};
use std::io::Write;

thrift_struct!(
    pub struct IndexPageHeader {}
);

thrift_struct!(
pub struct DictionaryPageHeader {
  /// Number of values in the dictionary
  1: required i32 num_values;

  /// Encoding using this dictionary page
  2: required Encoding encoding

  /// If true, the entries in the dictionary are sorted in ascending order
  3: optional bool is_sorted;
}
);

thrift_struct!(
/// Statistics for the page header.
///
/// This is a duplicate of the Statistics struct. Because the page reader uses
/// the Read API, we cannot read the min/max values as slices.
pub struct PageStatistics {
   1: optional binary max;
   2: optional binary min;
   3: optional i64 null_count;
   4: optional i64 distinct_count;
   5: optional binary max_value;
   6: optional binary min_value;
   7: optional bool is_max_value_exact;
   8: optional bool is_min_value_exact;
}
);

thrift_struct!(
pub struct DataPageHeader {
  1: required i32 num_values
  2: required Encoding encoding
  3: required Encoding definition_level_encoding;
  4: required Encoding repetition_level_encoding;
  5: optional PageStatistics statistics;
}
);

impl DataPageHeader {
    // reader that skips decoding page statistics
    fn read_thrift_without_stats<'a, R>(prot: &mut R) -> Result<Self>
    where
        R: ThriftCompactInputProtocol<'a>,
    {
        let mut num_values: Option<i32> = None;
        let mut encoding: Option<Encoding> = None;
        let mut definition_level_encoding: Option<Encoding> = None;
        let mut repetition_level_encoding: Option<Encoding> = None;
        let statistics: Option<PageStatistics> = None;
        let mut last_field_id = 0i16;
        loop {
            let field_ident = prot.read_field_begin(last_field_id)?;
            if field_ident.field_type == FieldType::Stop {
                break;
            }
            match field_ident.id {
                1 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    num_values = Some(val);
                }
                2 => {
                    let val = Encoding::read_thrift(&mut *prot)?;
                    encoding = Some(val);
                }
                3 => {
                    let val = Encoding::read_thrift(&mut *prot)?;
                    definition_level_encoding = Some(val);
                }
                4 => {
                    let val = Encoding::read_thrift(&mut *prot)?;
                    repetition_level_encoding = Some(val);
                }
                _ => {
                    prot.skip(field_ident.field_type)?;
                }
            };
            last_field_id = field_ident.id;
        }
        let Some(num_values) = num_values else {
            return Err(general_err!("Required field num_values is missing"));
        };
        let Some(encoding) = encoding else {
            return Err(general_err!("Required field encoding is missing"));
        };
        let Some(definition_level_encoding) = definition_level_encoding else {
            return Err(general_err!(
                "Required field definition_level_encoding is missing"
            ));
        };
        let Some(repetition_level_encoding) = repetition_level_encoding else {
            return Err(general_err!(
                "Required field repetition_level_encoding is missing"
            ));
        };
        Ok(Self {
            num_values,
            encoding,
            definition_level_encoding,
            repetition_level_encoding,
            statistics,
        })
    }
}

thrift_struct!(
pub struct DataPageHeaderV2 {
  1: required i32 num_values
  2: required i32 num_nulls
  3: required i32 num_rows
  4: required Encoding encoding
  5: required i32 definition_levels_byte_length;
  6: required i32 repetition_levels_byte_length;
  7: optional bool is_compressed = true;
  8: optional PageStatistics statistics;
}
);

impl DataPageHeaderV2 {
    // reader that skips decoding page statistics
    fn read_thrift_without_stats<'a, R>(prot: &mut R) -> Result<Self>
    where
        R: ThriftCompactInputProtocol<'a>,
    {
        let mut num_values: Option<i32> = None;
        let mut num_nulls: Option<i32> = None;
        let mut num_rows: Option<i32> = None;
        let mut encoding: Option<Encoding> = None;
        let mut definition_levels_byte_length: Option<i32> = None;
        let mut repetition_levels_byte_length: Option<i32> = None;
        let mut is_compressed: Option<bool> = None;
        let statistics: Option<PageStatistics> = None;
        let mut last_field_id = 0i16;
        loop {
            let field_ident = prot.read_field_begin(last_field_id)?;
            if field_ident.field_type == FieldType::Stop {
                break;
            }
            match field_ident.id {
                1 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    num_values = Some(val);
                }
                2 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    num_nulls = Some(val);
                }
                3 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    num_rows = Some(val);
                }
                4 => {
                    let val = Encoding::read_thrift(&mut *prot)?;
                    encoding = Some(val);
                }
                5 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    definition_levels_byte_length = Some(val);
                }
                6 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    repetition_levels_byte_length = Some(val);
                }
                7 => {
                    let val = field_ident.bool_val.unwrap();
                    is_compressed = Some(val);
                }
                _ => {
                    prot.skip(field_ident.field_type)?;
                }
            };
            last_field_id = field_ident.id;
        }
        let Some(num_values) = num_values else {
            return Err(general_err!("Required field num_values is missing"));
        };
        let Some(num_nulls) = num_nulls else {
            return Err(general_err!("Required field num_nulls is missing"));
        };
        let Some(num_rows) = num_rows else {
            return Err(general_err!("Required field num_rows is missing"));
        };
        let Some(encoding) = encoding else {
            return Err(general_err!("Required field encoding is missing"));
        };
        let Some(definition_levels_byte_length) = definition_levels_byte_length else {
            return Err(general_err!(
                "Required field definition_levels_byte_length is missing"
            ));
        };
        let Some(repetition_levels_byte_length) = repetition_levels_byte_length else {
            return Err(general_err!(
                "Required field repetition_levels_byte_length is missing"
            ));
        };
        Ok(Self {
            num_values,
            num_nulls,
            num_rows,
            encoding,
            definition_levels_byte_length,
            repetition_levels_byte_length,
            is_compressed,
            statistics,
        })
    }
}

thrift_struct!(
pub struct PageHeader {
  /// the type of the page: indicates which of the *_header fields is set
  1: required PageType r#type

  /// Uncompressed page size in bytes (not including this header)
  2: required i32 uncompressed_page_size

  /// Compressed (and potentially encrypted) page size in bytes, not including this header
  3: required i32 compressed_page_size

  /// The 32-bit CRC checksum for the page, to be be calculated as follows:
  4: optional i32 crc

  // Headers for page specific data.  One only will be set.
  5: optional DataPageHeader data_page_header;
  6: optional IndexPageHeader index_page_header;
  7: optional DictionaryPageHeader dictionary_page_header;
  8: optional DataPageHeaderV2 data_page_header_v2;
}
);

impl PageHeader {
    // reader that skips reading page statistics. obtained by running
    // `cargo expand -p parquet --all-features --lib file::metadata::thrift`
    // and modifying the impl of `read_thrift`
    pub(crate) fn read_thrift_without_stats<'a, R>(prot: &mut R) -> Result<Self>
    where
        R: ThriftCompactInputProtocol<'a>,
    {
        let mut type_: Option<PageType> = None;
        let mut uncompressed_page_size: Option<i32> = None;
        let mut compressed_page_size: Option<i32> = None;
        let mut crc: Option<i32> = None;
        let mut data_page_header: Option<DataPageHeader> = None;
        let mut index_page_header: Option<IndexPageHeader> = None;
        let mut dictionary_page_header: Option<DictionaryPageHeader> = None;
        let mut data_page_header_v2: Option<DataPageHeaderV2> = None;
        let mut last_field_id = 0i16;
        loop {
            let field_ident = prot.read_field_begin(last_field_id)?;
            if field_ident.field_type == FieldType::Stop {
                break;
            }
            match field_ident.id {
                1 => {
                    let val = PageType::read_thrift(&mut *prot)?;
                    type_ = Some(val);
                }
                2 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    uncompressed_page_size = Some(val);
                }
                3 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    compressed_page_size = Some(val);
                }
                4 => {
                    let val = i32::read_thrift(&mut *prot)?;
                    crc = Some(val);
                }
                5 => {
                    let val = DataPageHeader::read_thrift_without_stats(&mut *prot)?;
                    data_page_header = Some(val);
                }
                6 => {
                    let val = IndexPageHeader::read_thrift(&mut *prot)?;
                    index_page_header = Some(val);
                }
                7 => {
                    let val = DictionaryPageHeader::read_thrift(&mut *prot)?;
                    dictionary_page_header = Some(val);
                }
                8 => {
                    let val = DataPageHeaderV2::read_thrift_without_stats(&mut *prot)?;
                    data_page_header_v2 = Some(val);
                }
                _ => {
                    prot.skip(field_ident.field_type)?;
                }
            };
            last_field_id = field_ident.id;
        }
        let Some(type_) = type_ else {
            return Err(general_err!("Required field type_ is missing"));
        };
        let Some(uncompressed_page_size) = uncompressed_page_size else {
            return Err(general_err!(
                "Required field uncompressed_page_size is missing"
            ));
        };
        let Some(compressed_page_size) = compressed_page_size else {
            return Err(general_err!(
                "Required field compressed_page_size is missing"
            ));
        };
        Ok(Self {
            r#type: type_,
            uncompressed_page_size,
            compressed_page_size,
            crc,
            data_page_header,
            index_page_header,
            dictionary_page_header,
            data_page_header_v2,
        })
    }

    pub fn data_page_num_values(&self) -> i32 {
        self.data_page_header
            .as_ref()
            .expect("called data_page_num_values on a non-data page")
            .num_values
    }

    pub fn data_page_encoding(&self) -> Encoding {
        self.data_page_header
            .as_ref()
            .expect("called data_page_encoding on a non-data page")
            .encoding
    }

    pub fn dict_page_num_values(&self) -> i32 {
        self.dictionary_page_header
            .as_ref()
            .expect("called dict_page_num_values on a non-dictionary page")
            .num_values
    }

    pub fn for_data_page(num_values: i32, encoding: Encoding) -> Self {
        Self {
            r#type: PageType::DATA_PAGE,
            uncompressed_page_size: 0,
            compressed_page_size: 0,
            crc: None,
            data_page_header: Some(DataPageHeader {
                num_values,
                encoding,
                definition_level_encoding: Encoding::RLE,
                repetition_level_encoding: Encoding::RLE,
                statistics: None,
            }),
            index_page_header: None,
            dictionary_page_header: None,
            data_page_header_v2: None,
        }
    }

    pub fn for_dict_page(num_values: i32) -> Self {
        Self {
            r#type: PageType::DICTIONARY_PAGE,
            uncompressed_page_size: 0,
            compressed_page_size: 0,
            crc: None,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: Some(DictionaryPageHeader {
                num_values,
                encoding: Encoding::PLAIN,
                is_sorted: None,
            }),
            data_page_header_v2: None,
        }
    }
}
