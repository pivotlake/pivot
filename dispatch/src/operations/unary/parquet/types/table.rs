//! Table above parquet files
//!
//! A [`ParquetTable`] is the engine's handle to one or more Parquet files that
//! together form a logical table. Construction reads and parses each file's
//! Thrift footer, converts the Parquet schema to Arrow, and collects
//! `RowGroupMetadata` entries with globally unique row-group indices.

use crate::io::open_direct_read;
use crate::memory::FILE_CACHE;
use crate::operations::unary::parquet::types::metadata::{ColumnChunkMeta, RowGroupMetadata};
use crate::operations::unary::parquet::types::thrift::footer::FileMetaData;
use crate::operations::unary::parquet::types::thrift::parquet_thrift::{
    ReadThrift, ThriftSliceInputProtocol,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::{fs, io};
use thiserror::Error;

static EMPTY_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| SchemaRef::new(Schema::empty()));

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    IO(#[from] std::io::Error),
    #[error("Column not found {0}")]
    ColumnNotFound(String),
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// A logical table backed by one or more Parquet files.
///
/// Holds a flat, globally-indexed list of `RowGroupMetadata` entries spanning
/// every file. Cheaply shareable via `Arc` because the metadata is read-only
/// after construction.
pub struct ParquetTable {
    pub(crate) row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl Debug for ParquetTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("table")
    }
}

impl ParquetTable {
    /// Wraps pre-built row group metadata into a table.
    pub fn new(row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        Self { row_groups }
    }

    /// Creates a table from every Parquet file in `path`.
    ///
    /// Reads and parses the Thrift footer of each file, opens the file with
    /// direct IO, and registers it in the file cache. Row groups are assigned
    /// globally unique indices in the order files are discovered.
    pub fn from_directory(path: &Path) -> Result<Self> {
        let mut start_offset = 0;
        let row_groups: Vec<_> = fs::read_dir(path)?
            .flatten()
            .filter_map(|d| {
                if d.path().is_file() {
                    Some(
                        parse_row_group_metadatas(start_offset, d.path()).inspect(|v| {
                            start_offset += v.len();
                        }),
                    )
                } else {
                    None
                }
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .map(Arc::new)
            .collect();
        Ok(Self::new(row_groups))
    }

    /// Returns the Arrow schema (taken from the first row group).
    pub fn schema(&self) -> &SchemaRef {
        if self.row_groups.is_empty() {
            &EMPTY_SCHEMA
        } else {
            &self.row_groups[0].schema
        }
    }
}

const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

/// Read the parquet footer bytes from a file. Returns the raw thrift-encoded metadata.
fn read_parquet_footer(file: &mut File) -> std::io::Result<Vec<u8>> {
    // Read the last 8 bytes: 4-byte footer length + 4-byte magic
    file.seek(SeekFrom::End(-8))?;
    let mut tail = [0u8; 8];
    file.read_exact(&mut tail)?;

    if tail[4..] != PARQUET_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a parquet file",
        ));
    }

    let footer_len = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;

    // Read the footer metadata
    file.seek(SeekFrom::End(-(8 + footer_len as i64)))?;
    let mut buf = vec![0u8; footer_len];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

/// Parses a single Parquet file's footer and returns one [`RowGroupMetadata`]
/// per row group, with global indices starting at `global_row_group_offset`.
fn parse_row_group_metadatas(
    global_row_group_offset: usize,
    path: impl AsRef<Path>,
) -> Result<Vec<RowGroupMetadata>> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let buf = read_parquet_footer(&mut file)?;
    let file_meta = parse_footer_thrift(&buf)?;
    let file = open_direct_read(path)?;
    FILE_CACHE.open_file_entry(file.as_raw_fd());

    let file = Arc::new(file);

    let (schema, def_levels) = schema_elements_to_arrow(&file_meta.schema)?;
    let schema = Arc::new(schema);

    let row_groups = file_meta
        .row_groups
        .iter()
        .enumerate()
        .map(|(i, rg)| {
            let columns = rg
                .columns
                .iter()
                .enumerate()
                .map(|(j, cc)| {
                    let meta = cc.meta_data.as_ref().expect("missing column metadata");
                    ColumnChunkMeta {
                        dictionary_page_offset: meta.dictionary_page_offset,
                        data_page_offset: meta.data_page_offset,
                        total_compressed_size: meta.total_compressed_size,
                        max_def_level: def_levels[j],
                    }
                })
                .collect();
            RowGroupMetadata {
                file: file.clone(),
                schema: schema.clone(),
                columns,
                num_rows: rg.num_rows,
                file_row_group_idx: i,
                global_row_group_idx: global_row_group_offset + i,
            }
        })
        .collect();

    Ok(row_groups)
}

/// Deserialises raw Thrift bytes into a [`FileMetaData`].
fn parse_footer_thrift(buf: &[u8]) -> Result<FileMetaData> {
    let mut prot = ThriftSliceInputProtocol::new(buf);
    FileMetaData::read_thrift(&mut prot)
        .map_err(|e| Error::IO(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))
}

/// Convert flat SchemaElement list to Arrow Schema + max definition levels per leaf column.
fn schema_elements_to_arrow(
    elements: &[crate::operations::parquet::types::thrift::footer::SchemaElement],
) -> Result<(Schema, Vec<i16>)> {
    if elements.is_empty() {
        return Err(Error::IO(io::Error::new(
            io::ErrorKind::InvalidData,
            "empty schema",
        )));
    }

    // Element 0 is the root group
    let num_children = elements[0].num_children.unwrap_or(0) as usize;
    let mut fields = Vec::with_capacity(num_children);
    let mut def_levels = Vec::with_capacity(num_children);

    for elem in elements.iter().skip(1).take(num_children) {
        // repetition_type: 0=REQUIRED, 1=OPTIONAL, 2=REPEATED
        let nullable = elem.repetition_type == Some(1);
        let def_level: i16 = if nullable { 1 } else { 0 };

        let data_type = convert_physical_to_arrow(
            elem.physical_type,
            elem.converted_type,
            elem.logical_type.as_ref(),
        )?;

        fields.push(Field::new(&elem.name, data_type, nullable));
        def_levels.push(def_level);
    }

    Ok((Schema::new(fields), def_levels))
}

/// Maps a Parquet physical type (+ optional logical/converted type annotations)
/// to an Arrow [`DataType`].
fn convert_physical_to_arrow(
    physical_type: Option<i32>,
    converted_type: Option<i32>,
    logical_type: Option<&crate::operations::parquet::types::thrift::footer::LogicalType>,
) -> Result<DataType> {
    use crate::operations::unary::parquet::types::thrift::footer::LogicalType;

    let pt = physical_type.ok_or_else(|| {
        Error::IO(io::Error::new(
            io::ErrorKind::InvalidData,
            "leaf schema element missing physical type",
        ))
    })?;

    match pt {
        // BOOLEAN = 0
        0 => Ok(DataType::Boolean),
        // INT32 = 1
        1 => match logical_type {
            Some(LogicalType::Integer {
                bit_width: 8,
                is_signed: true,
            }) => Ok(DataType::Int8),
            Some(LogicalType::Integer {
                bit_width: 8,
                is_signed: false,
            }) => Ok(DataType::UInt8),
            Some(LogicalType::Integer {
                bit_width: 16,
                is_signed: true,
            }) => Ok(DataType::Int16),
            Some(LogicalType::Integer {
                bit_width: 16,
                is_signed: false,
            }) => Ok(DataType::UInt16),
            Some(LogicalType::Integer {
                bit_width: 32,
                is_signed: true,
            }) => Ok(DataType::Int32),
            Some(LogicalType::Integer {
                bit_width: 32,
                is_signed: false,
            }) => Ok(DataType::UInt32),
            _ => match converted_type {
                Some(15) => Ok(DataType::Int8),   // INT_8
                Some(11) => Ok(DataType::UInt8),  // UINT_8
                Some(16) => Ok(DataType::Int16),  // INT_16
                Some(12) => Ok(DataType::UInt16), // UINT_16
                Some(17) => Ok(DataType::Int32),  // INT_32
                Some(13) => Ok(DataType::UInt32), // UINT_32
                _ => Ok(DataType::Int32),
            },
        },
        // INT64 = 2
        2 => Ok(DataType::Int64),
        // FLOAT = 4
        4 => Ok(DataType::Float32),
        // DOUBLE = 5
        5 => Ok(DataType::Float64),
        // BYTE_ARRAY = 6
        6 => Ok(DataType::Utf8View),
        _ => Err(Error::IO(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported physical type {}", pt),
        ))),
    }
}
