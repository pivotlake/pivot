//! Table above parquet files
//!
//! A [`ParquetTable`] is the engine's handle to one or more Parquet files that
//! together form a logical table. Construction reads and parses each file's
//! Thrift footer, converts the Parquet schema to Arrow, and collects
//! `RowGroupMetadata` entries with globally unique row-group indices.

use crate::parquet::types::metadata::{ColumnChunkMeta, ColumnStatistics, RowGroupMetadata};
use crate::parquet::types::thrift::footer::{FileMetaData, PageEncodingStats, Statistics};
use crate::parquet::types::thrift::general::{Encoding, PageType};
use crate::parquet::types::thrift::parquet_thrift::{ReadThrift, ThriftSliceInputProtocol};
use crate::store::DataFile;
use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, Scalar, StringViewArray, TimestampSecondArray, UInt8Array, UInt16Array,
    UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use dispatch::DataFlowDispatcher;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, LazyLock};
use std::{fs, io};
use thiserror::Error;
use url::Url;

static EMPTY_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| SchemaRef::new(Schema::empty()));

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    IO(#[from] std::io::Error),
    #[error("Column not found {0}")]
    ColumnNotFound(String),
    #[error("fetching row-group metadata: {0}")]
    Materialize(String),
    #[error("invalid parquet footer: {0}")]
    InvalidFooter(String),
    /// A Parquet/arrow type with no mapping in either direction (an arrow type
    /// the writer can't emit, or a Parquet leaf the reader can't decode).
    #[error("unsupported type: {0}")]
    UnsupportedType(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A logical table backed by one or more Parquet files.
///
/// Holds a flat, globally-indexed list of `RowGroupMetadata` entries spanning
/// every file. Cheaply shareable via `Arc` because the metadata is read-only
/// after construction.
#[derive(Clone)]
pub struct ParquetTable {
    pub(crate) row_groups: Vec<Arc<RowGroupMetadata>>,
    /// The table's Arrow schema, captured at construction so it survives even
    /// when every row group is pruned away (e.g. predicate-stats pruning removes
    /// all of them). Deriving it from `row_groups[0]` instead would yield an
    /// empty schema for a fully-pruned table, panicking any scan that projects a
    /// column.
    schema: SchemaRef,
}

impl Debug for ParquetTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("table")
    }
}

impl ParquetTable {
    /// Wraps pre-built row group metadata into a table. The schema is taken from
    /// the first row group (or empty when there are none) and then retained, so
    /// later pruning of all row groups does not lose it.
    pub fn new(row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        let schema = row_groups
            .first()
            .map(|rg| rg.schema.clone())
            .unwrap_or_else(|| EMPTY_SCHEMA.clone());
        Self { row_groups, schema }
    }

    /// Read-only view of this table's row groups.
    pub fn row_groups(&self) -> &[Arc<RowGroupMetadata>] {
        &self.row_groups
    }

    /// Mutable access to this table's row groups. Used by callers that need
    /// to prune the row-group set in place (e.g. catalog-side filter pushdown).
    pub fn row_groups_mut(&mut self) -> &mut Vec<Arc<RowGroupMetadata>> {
        &mut self.row_groups
    }

    /// Build a table from every Parquet file in a local directory: list the
    /// `*.parquet` files, then delegate to [`from_files`](Self::from_files).
    ///
    /// Drives the metadata-fetch dataflow, so it must run on the **coordinator**
    /// (the thread holding `dispatcher`), not inside a `run_on_worker` closure —
    /// a nested dataflow would deadlock the worker pool.
    pub fn from_directory(dispatcher: &DataFlowDispatcher, path: &Path) -> Result<Self> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            // Only `*.parquet` — skip any sidecar files.
            if path.extension().is_some_and(|ext| ext == "parquet") && path.is_file() {
                paths.push(path);
            }
        }
        Self::from_files(dispatcher, &paths)
    }

    /// Build a table from an explicit, ordered list of local files. Same
    /// coordinator requirement as [`from_directory`](Self::from_directory).
    pub fn from_files<P: AsRef<Path>>(
        dispatcher: &DataFlowDispatcher,
        paths: &[P],
    ) -> Result<Self> {
        let files = paths
            .iter()
            .map(|p| {
                let path = p.as_ref();
                let size = fs::metadata(path)?.len();
                Ok(DataFile::local(path.to_path_buf(), size))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Self::from_locations(dispatcher, files)
    }

    /// Build a table from remote files: concrete fetchable URLs paired with
    /// their total size (from the store listing), which locates each footer.
    /// Same coordinator requirement as [`from_directory`](Self::from_directory).
    pub fn from_remote_files(
        dispatcher: &DataFlowDispatcher,
        files: &[(Url, u64)],
    ) -> Result<Self> {
        let files = files
            .iter()
            .map(|(url, size)| DataFile::remote(url.clone(), *size))
            .collect();
        Self::from_locations(dispatcher, files)
    }

    /// Read every file's footer in parallel (the metadata-fetch dataflow) and
    /// assemble the row groups. The locations may freely mix local and remote
    /// files. Same coordinator requirement as
    /// [`from_directory`](Self::from_directory).
    pub fn from_locations(dispatcher: &DataFlowDispatcher, files: Vec<DataFile>) -> Result<Self> {
        let table_files = crate::parquet::metadata::load_table_files(dispatcher, &files)
            .map_err(|e| Error::Materialize(e.to_string()))?;
        let row_groups = table_files
            .iter()
            .flat_map(|f| f.row_groups().iter().cloned())
            .collect();
        Ok(Self::new(row_groups))
    }

    /// Returns the table's Arrow schema. Preserved across pruning, so a table
    /// whose row groups were all eliminated still reports its real schema.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }
}

/// How much of a file's tail to fetch when probing for the footer. Almost every
/// footer fits, so it's a single read; a larger footer falls back to an exact
/// second read.
pub(crate) const FOOTER_PROBE_BYTES: usize = 64 * 1024;

/// Parse raw thrift footer bytes into this file's row groups (file-local
/// indices), tying each to `location` for the column-chunk reads that follow.
pub(crate) fn row_groups_from_footer(
    footer: &[u8],
    location: dispatch::io::FileLocation,
) -> Result<Vec<RowGroupMetadata>> {
    let file_meta = parse_footer_thrift(footer)?;
    row_groups_from_metadata(file_meta, location)
}

/// Build the per-row-group metadata from a parsed footer and the (local or
/// remote) open file. Shared by the local and remote readers and by the writer's
/// upload path, which already holds the footer metadata it wrote and so skips the
/// parse. Row groups carry only their *file-local* index; the global index is the
/// row group's eventual position in the table's flat list.
pub(crate) fn row_groups_from_metadata(
    file_meta: FileMetaData,
    location: dispatch::io::FileLocation,
) -> Result<Vec<RowGroupMetadata>> {
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
                    let data_pages_all_dictionary = meta.dictionary_page_offset.is_some()
                        && data_pages_all_dictionary(meta.encoding_stats.as_deref());
                    ColumnChunkMeta {
                        dictionary_page_offset: meta.dictionary_page_offset,
                        data_page_offset: meta.data_page_offset,
                        total_compressed_size: meta.total_compressed_size,
                        max_def_level: def_levels[j],
                        statistics,
                        data_pages_all_dictionary,
                    }
                })
                .collect();
            RowGroupMetadata {
                location: location.clone(),
                schema: schema.clone(),
                columns,
                num_rows,
                file_row_group_idx: i,
                live_decompressed_pages: Arc::new(AtomicUsize::new(0)),
            }
        })
        .collect();

    Ok(row_groups)
}

/// Returns `true` when `encoding_stats` proves every *data* page in the chunk
/// is dictionary-encoded (`PLAIN_DICTIONARY` / `RLE_DICTIONARY`). Requires at
/// least one data page and no data page using any other encoding. Returns
/// `false` when stats are absent — a missing signal is not proof, so we must
/// not prune.
fn data_pages_all_dictionary(encoding_stats: Option<&[PageEncodingStats]>) -> bool {
    let Some(stats) = encoding_stats else {
        return false;
    };
    let mut saw_data_page = false;
    for s in stats {
        if s.page_type != PageType::DATA_PAGE && s.page_type != PageType::DATA_PAGE_V2 {
            continue;
        }
        if s.count == 0 {
            continue;
        }
        saw_data_page = true;
        if s.encoding != Encoding::PLAIN_DICTIONARY && s.encoding != Encoding::RLE_DICTIONARY {
            return false;
        }
    }
    saw_data_page
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
        // Temporal stats share their physical int's encoding (Date32 the i32
        // days, Timestamp(Second) the i64 count).
        DataType::Date32 => read_le::<4>(bytes)
            .map(i32::from_le_bytes)
            .map(|v| erase_type(Date32Array::new_scalar(v))),
        DataType::Timestamp(TimeUnit::Second, None) => read_le::<8>(bytes)
            .map(i64::from_le_bytes)
            .map(|v| erase_type(TimestampSecondArray::new_scalar(v))),
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
    elements: &[crate::parquet::types::thrift::footer::SchemaElement],
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

        let data_type = super::arrow_map::parquet_to_arrow(
            elem.physical_type,
            elem.converted_type,
            elem.logical_type.as_ref(),
        )?;

        fields.push(Field::new(&elem.name, data_type, nullable));
        def_levels.push(def_level);
    }

    Ok((Schema::new(fields), def_levels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BooleanArray, Datum, Float64Array, Int32Array, Int64Array, RecordBatch, StringViewArray,
    };
    use arrow_schema::Field;
    use dispatch::{DataFlowDispatcher, Dispatch};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use std::fs::File;
    use std::sync::{Arc, OnceLock};
    use tempfile::TempDir;

    /// A shared one-worker dispatch pool for this module's tests, so
    /// `ParquetTable::from_directory` can drive its metadata-fetch dataflow.
    fn test_dispatcher() -> &'static DataFlowDispatcher {
        static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
        DISPATCH
            .get_or_init(|| Dispatch::spin_up(1, 32, None))
            .dispatcher()
    }

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
        let table = ParquetTable::from_directory(test_dispatcher(), dir.path()).unwrap();
        (dir, table)
    }

    fn col_stats(table: &ParquetTable, col: usize) -> ColumnStatistics {
        table.row_groups[0].columns[col]
            .statistics
            .clone()
            .expect("stats should be present")
    }

    fn enc_stat(page_type: PageType, encoding: Encoding, count: i32) -> PageEncodingStats {
        PageEncodingStats {
            page_type,
            encoding,
            count,
        }
    }

    /// Pruning away every row group keeps the table's schema, so a scan that
    /// projects a column still builds. Regression: a fully-pruned table reported
    /// an empty schema, panicking the decoder build on the projected column.
    #[test]
    fn schema_survives_pruning_every_row_group() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![2])),
            ],
        )
        .unwrap();
        let (_dir, mut table) = write_parquet(&batch, EnabledStatistics::Chunk);
        let schema = table.schema().clone();

        table.row_groups_mut().clear();

        assert_eq!(table.schema(), &schema);
        assert_eq!(table.schema().fields().len(), 2);
    }

    /// All data pages dictionary-encoded → prunable.
    #[test]
    fn all_dict_when_data_pages_are_dictionary() {
        let stats = [
            enc_stat(PageType::DICTIONARY_PAGE, Encoding::PLAIN_DICTIONARY, 1),
            enc_stat(PageType::DATA_PAGE, Encoding::PLAIN_DICTIONARY, 3),
        ];
        assert!(data_pages_all_dictionary(Some(&stats)));
    }

    /// A single non-dictionary (PLAIN) data page disables pruning, even
    /// alongside dictionary-encoded data pages.
    #[test]
    fn not_all_dict_when_a_data_page_is_plain() {
        let stats = [
            enc_stat(PageType::DICTIONARY_PAGE, Encoding::PLAIN, 1),
            enc_stat(PageType::DATA_PAGE, Encoding::PLAIN_DICTIONARY, 2),
            enc_stat(PageType::DATA_PAGE, Encoding::PLAIN, 1),
        ];
        assert!(!data_pages_all_dictionary(Some(&stats)));
    }

    /// Missing encoding stats are not proof of anything → not prunable.
    #[test]
    fn not_all_dict_when_stats_absent() {
        assert!(!data_pages_all_dictionary(None));
    }

    /// A dictionary page with no data pages is not prunable (nothing to prove).
    #[test]
    fn not_all_dict_when_no_data_pages() {
        let stats = [enc_stat(
            PageType::DICTIONARY_PAGE,
            Encoding::PLAIN_DICTIONARY,
            1,
        )];
        assert!(!data_pages_all_dictionary(Some(&stats)));
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
