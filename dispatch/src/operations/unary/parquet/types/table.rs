//! Table above parquet files
//!
//! A [`ParquetTable`] is the engine's handle to one or more Parquet files that
//! together form a logical table. Construction reads and parses each file's
//! Thrift footer, converts the Parquet schema to Arrow, and collects
//! `RowGroupMetadata` entries with globally unique row-group indices.

use crate::io::open_direct_read;
use crate::memory::FILE_CACHE;
use crate::operations::unary::parquet::types::metadata::{
    ColumnChunkMeta, ColumnStatistics, RowGroupMetadata,
};
use crate::operations::unary::parquet::types::thrift::footer::{FileMetaData, Statistics};
use crate::operations::unary::parquet::types::thrift::parquet_thrift::{
    ReadThrift, ThriftSliceInputProtocol,
};
use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, Scalar, StringViewArray, UInt8Array, UInt16Array, UInt32Array,
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
#[derive(Clone)]
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
        .into_iter()
        .enumerate()
        .map(|(i, rg)| {
            let num_rows = rg.num_rows;
            let columns = rg
                .columns
                .into_iter()
                .enumerate()
                .map(|(j, cc)| {
                    let meta = cc.meta_data.expect("missing column metadata");
                    let statistics = meta
                        .statistics
                        .and_then(|s| decode_statistics(s, schema.field(j).data_type()));
                    ColumnChunkMeta {
                        dictionary_page_offset: meta.dictionary_page_offset,
                        data_page_offset: meta.data_page_offset,
                        total_compressed_size: meta.total_compressed_size,
                        max_def_level: def_levels[j],
                        statistics,
                    }
                })
                .collect();
            RowGroupMetadata {
                file: file.clone(),
                schema: schema.clone(),
                columns,
                num_rows,
                file_row_group_idx: i,
                global_row_group_idx: global_row_group_offset + i,
            }
        })
        .collect();

    Ok(row_groups)
}

/// Decode a Parquet `Statistics` blob into our [`ColumnStatistics`], using
/// `data_type` to choose the right physical-bytes -> Arrow scalar conversion.
///
/// Prefer the modern `min_value` / `max_value` fields and fall back to the
/// legacy `min` / `max` only when those aren't populated. Unsupported types
/// (or stats whose bytes don't match the expected width) yield `None` for
/// that side rather than failing the parse.
fn decode_statistics(stats: Statistics, data_type: &DataType) -> Option<ColumnStatistics> {
    let max_bytes = stats.max_value.or(stats.max);
    let min_bytes = stats.min_value.or(stats.min);
    if max_bytes.is_none()
        && min_bytes.is_none()
        && stats.null_count.is_none()
        && stats.distinct_count.is_none()
    {
        return None;
    }
    Some(ColumnStatistics {
        min: min_bytes.and_then(|bytes| decode_scalar(&bytes, data_type)),
        max: max_bytes.and_then(|bytes| decode_scalar(&bytes, data_type)),
        null_count: stats.null_count,
        distinct_count: stats.distinct_count,
    })
}

fn decode_scalar(bytes: &[u8], data_type: &DataType) -> Option<Scalar<ArrayRef>> {
    /// Read `N` bytes as a little-endian fixed-width primitive. Returns `None`
    /// if the byte slice doesn't have exactly `N` bytes.
    fn read_le<const N: usize>(bytes: &[u8]) -> Option<[u8; N]> {
        bytes.try_into().ok()
    }

    fn erase_type<T: arrow_array::Array + 'static>(s: Scalar<T>) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(s.into_inner()))
    }

    match data_type {
        DataType::Boolean => bytes
            .first()
            .map(|b| erase_type(BooleanArray::new_scalar(*b != 0))),
        DataType::Int8 => read_le::<4>(bytes)
            .map(i32::from_le_bytes)
            .and_then(|v| i8::try_from(v).ok())
            .map(|v| erase_type(Int8Array::new_scalar(v))),
        DataType::UInt8 => read_le::<4>(bytes)
            .map(u32::from_le_bytes)
            .and_then(|v| u8::try_from(v).ok())
            .map(|v| erase_type(UInt8Array::new_scalar(v))),
        DataType::Int16 => read_le::<4>(bytes)
            .map(i32::from_le_bytes)
            .and_then(|v| i16::try_from(v).ok())
            .map(|v| erase_type(Int16Array::new_scalar(v))),
        DataType::UInt16 => read_le::<4>(bytes)
            .map(u32::from_le_bytes)
            .and_then(|v| u16::try_from(v).ok())
            .map(|v| erase_type(UInt16Array::new_scalar(v))),
        DataType::Int32 => read_le::<4>(bytes)
            .map(i32::from_le_bytes)
            .map(|v| erase_type(Int32Array::new_scalar(v))),
        DataType::UInt32 => read_le::<4>(bytes)
            .map(u32::from_le_bytes)
            .map(|v| erase_type(UInt32Array::new_scalar(v))),
        DataType::Int64 => read_le::<8>(bytes)
            .map(i64::from_le_bytes)
            .map(|v| erase_type(Int64Array::new_scalar(v))),
        DataType::Float32 => read_le::<4>(bytes)
            .map(f32::from_le_bytes)
            .map(|v| erase_type(Float32Array::new_scalar(v))),
        DataType::Float64 => read_le::<8>(bytes)
            .map(f64::from_le_bytes)
            .map(|v| erase_type(Float64Array::new_scalar(v))),
        DataType::Utf8View => std::str::from_utf8(bytes)
            .ok()
            .map(|s| erase_type(StringViewArray::new_scalar(s))),
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BooleanArray, Datum, Float64Array, Int32Array, Int64Array, RecordBatch, StringViewArray,
    };
    use arrow_schema::Field;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use std::sync::Arc;
    use tempfile::TempDir;

    fn write_parquet(batch: &RecordBatch, stats: EnabledStatistics) -> (TempDir, ParquetTable) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data.parquet");
        let props = WriterProperties::builder()
            .set_statistics_enabled(stats)
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props))
                .unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        let table = ParquetTable::from_directory(dir.path()).unwrap();
        (dir, table)
    }

    fn col_stats(table: &ParquetTable, col: usize) -> ColumnStatistics {
        table.row_groups[0].columns[col]
            .statistics
            .clone()
            .expect("stats should be present")
    }

    /// End-to-end: write a parquet file with several column types, then read it
    /// back via `ParquetTable::from_directory` and verify the decoded min/max
    /// stats match the values we wrote.
    #[test]
    fn decodes_min_max_from_real_parquet_file() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i32", DataType::Int32, false),
            Field::new("i64", DataType::Int64, false),
            Field::new("f64", DataType::Float64, false),
            Field::new("name", DataType::Utf8View, false),
            Field::new("flag", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![3, -7, 42, 0])),
                Arc::new(Int64Array::from(vec![100i64, -200, 300, 50])),
                Arc::new(Float64Array::from(vec![1.5, -1.25, 3.75, 0.0])),
                Arc::new(StringViewArray::from(vec![
                    "banana", "apple", "zebra", "kiwi",
                ])),
                Arc::new(BooleanArray::from(vec![true, false, true, false])),
            ],
        )
        .unwrap();

        let (_dir, table) = write_parquet(&batch, EnabledStatistics::Chunk);

        let i32_stats = col_stats(&table, 0);
        let i32_min = i32_stats.min.as_ref().unwrap().get().0;
        let i32_max = i32_stats.max.as_ref().unwrap().get().0;
        assert_eq!(
            i32_min
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            -7
        );
        assert_eq!(
            i32_max
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            42
        );

        let i64_stats = col_stats(&table, 1);
        let i64_min = i64_stats.min.as_ref().unwrap().get().0;
        let i64_max = i64_stats.max.as_ref().unwrap().get().0;
        assert_eq!(
            i64_min
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            -200
        );
        assert_eq!(
            i64_max
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            300
        );

        let f64_stats = col_stats(&table, 2);
        let f64_min = f64_stats.min.as_ref().unwrap().get().0;
        let f64_max = f64_stats.max.as_ref().unwrap().get().0;
        assert_eq!(
            f64_min
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0),
            -1.25
        );
        assert_eq!(
            f64_max
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0),
            3.75
        );

        let name_stats = col_stats(&table, 3);
        let name_min = name_stats.min.as_ref().unwrap().get().0;
        let name_max = name_stats.max.as_ref().unwrap().get().0;
        assert_eq!(
            name_min
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(0),
            "apple"
        );
        assert_eq!(
            name_max
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(0),
            "zebra"
        );

        let bool_stats = col_stats(&table, 4);
        let bool_min = bool_stats.min.as_ref().unwrap().get().0;
        let bool_max = bool_stats.max.as_ref().unwrap().get().0;
        assert!(
            !bool_min
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0)
        );
        assert!(
            bool_max
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0)
        );
    }

    #[test]
    fn no_statistics_when_writer_disables_them() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

        let (_dir, table) = write_parquet(&batch, EnabledStatistics::None);

        assert!(table.row_groups[0].columns[0].statistics.is_none());
    }
}
