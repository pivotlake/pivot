//! BoundTable above parquet files
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
    ArrayRef, BooleanArray, Date32Array, Decimal64Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, Scalar, StringViewArray,
    TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use dispatch::DataFlowDispatcher;
use planner::catalog::Column;
use planner::types::Type;
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
    /// A file's stored column type contradicts the table's declared schema.
    #[error("column '{column}' is declared {declared} but the file stores {file_type}")]
    DeclaredTypeMismatch {
        column: String,
        declared: Type,
        file_type: DataType,
    },
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
    ///
    /// `declared_columns` is the table's declared schema, reconciled with
    /// each file's own schema (see [`apply_declared_types`]); pass `&[]` when
    /// nothing was declared.
    pub fn from_directory(
        dispatcher: &DataFlowDispatcher,
        path: &Path,
        declared_columns: &[Column],
    ) -> Result<Self> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            // Only `*.parquet` — skip any sidecar files.
            if path.extension().is_some_and(|ext| ext == "parquet") && path.is_file() {
                paths.push(path);
            }
        }
        Self::from_files(dispatcher, &paths, declared_columns)
    }

    /// Build a table from an explicit, ordered list of local files. Same
    /// coordinator requirement as [`from_directory`](Self::from_directory).
    pub fn from_files<P: AsRef<Path>>(
        dispatcher: &DataFlowDispatcher,
        paths: &[P],
        declared_columns: &[Column],
    ) -> Result<Self> {
        let files = paths
            .iter()
            .map(|p| {
                let path = p.as_ref();
                let size = fs::metadata(path)?.len();
                Ok(DataFile::local(path.to_path_buf(), size))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Self::from_locations(dispatcher, files, declared_columns)
    }

    /// Build a table from remote files: concrete fetchable URLs paired with
    /// their total size (from the store listing), which locates each footer.
    /// Same coordinator requirement as [`from_directory`](Self::from_directory).
    pub fn from_remote_files(
        dispatcher: &DataFlowDispatcher,
        files: &[(Url, u64)],
        declared_columns: &[Column],
    ) -> Result<Self> {
        let files = files
            .iter()
            .map(|(url, size)| DataFile::remote(url.clone(), *size))
            .collect();
        Self::from_locations(dispatcher, files, declared_columns)
    }

    /// Read every file's footer in parallel (the metadata-fetch dataflow) and
    /// assemble the row groups. The locations may freely mix local and remote
    /// files. Same coordinator requirement as
    /// [`from_directory`](Self::from_directory).
    pub fn from_locations(
        dispatcher: &DataFlowDispatcher,
        files: Vec<DataFile>,
        declared_columns: &[Column],
    ) -> Result<Self> {
        let table_files =
            crate::parquet::metadata::load_table_files(dispatcher, &files, declared_columns.into())
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
/// indices), tying each to `open_file` for the column-chunk reads that follow.
/// `declared_columns` is the table's declared schema (empty when the table has
/// none), reconciled with the file's own schema by [`apply_declared_types`].
pub(crate) fn row_groups_from_footer(
    footer: &[u8],
    open_file: dispatch::io::OpenFile,
    declared_columns: &[Column],
) -> Result<Vec<RowGroupMetadata>> {
    let file_meta = parse_footer_thrift(footer)?;
    row_groups_from_metadata(file_meta, open_file, declared_columns)
}

/// Build the per-row-group metadata from a parsed footer and the (local or
/// remote) open file. Shared by the local and remote readers and by the writer's
/// upload path, which already holds the footer metadata it wrote and so skips the
/// parse. Row groups carry only their *file-local* index; the global index is the
/// row group's eventual position in the table's flat list.
pub(crate) fn row_groups_from_metadata(
    file_meta: FileMetaData,
    open_file: dispatch::io::OpenFile,
    declared_columns: &[Column],
) -> Result<Vec<RowGroupMetadata>> {
    let (schema, leaf_infos) = schema_elements_to_arrow(&file_meta.schema)?;
    // Use reconciled types for statistics, decoders, and output fields.
    let schema = Arc::new(apply_declared_types(schema, declared_columns)?);
    // Statistics belong to leaf chunks, not top-level columns.
    let leaves = super::leaves::leaf_fields(schema.fields());

    let row_groups = file_meta
        .row_groups
        .into_iter()
        .enumerate()
        .map(|(i, rg)| {
            // Chunks are per leaf; a row group with a different count is a
            // malformed footer and would index out of bounds below.
            if rg.columns.len() != leaves.len() {
                return Err(Error::InvalidFooter(format!(
                    "row group {i} has {} column chunks but the schema has {} leaves",
                    rg.columns.len(),
                    leaves.len()
                )));
            }
            let num_rows = rg.num_rows;
            let columns = rg
                .columns
                .into_iter()
                .enumerate()
                .map(|(j, cc)| {
                    let meta = cc.meta_data.expect("missing column metadata");
                    let physical_type = meta.physical_type;
                    let statistics = meta
                        .statistics
                        .and_then(|s| decode_statistics(s, leaves[j].data_type(), physical_type));
                    let data_pages_all_dictionary = meta.dictionary_page_offset.is_some()
                        && data_pages_all_dictionary(meta.encoding_stats.as_deref());
                    ColumnChunkMeta {
                        dictionary_page_offset: meta.dictionary_page_offset,
                        data_page_offset: meta.data_page_offset,
                        total_compressed_size: meta.total_compressed_size,
                        max_def_level: leaf_infos[j].def_level,
                        physical_type,
                        fixed_len_byte_width: leaf_infos[j].type_length,
                        statistics,
                        data_pages_all_dictionary,
                    }
                })
                .collect();
            Ok(RowGroupMetadata {
                open_file: open_file.clone(),
                schema: schema.clone(),
                columns,
                num_rows,
                file_row_group_idx: i,
                live_decompressed_pages: Arc::new(AtomicUsize::new(0)),
            })
        })
        .collect::<Result<_>>()?;

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
/// `physical_type` disambiguates a decimal's byte encoding, which follows the
/// column's physical storage rather than its arrow type.
///
/// Prefer the modern `min_value` / `max_value` fields and fall back to the
/// legacy `min` / `max` only when those aren't populated. Unsupported types
/// (or stats whose bytes don't match the expected width) yield `None` for
/// that side rather than failing the parse.
fn decode_statistics(
    stats: Statistics,
    data_type: &DataType,
    physical_type: i32,
) -> Option<ColumnStatistics> {
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
        min: min_bytes.and_then(|bytes| decode_scalar(&bytes, data_type, physical_type)),
        max: max_bytes.and_then(|bytes| decode_scalar(&bytes, data_type, physical_type)),
        null_count: stats.null_count,
        distinct_count: stats.distinct_count,
    })
}

fn decode_scalar(
    bytes: &[u8],
    data_type: &DataType,
    physical_type: i32,
) -> Option<Scalar<ArrayRef>> {
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
        DataType::UInt64 => read_le::<8>(bytes)
            .map(u64::from_le_bytes)
            .map(|v| erase_type(UInt64Array::new_scalar(v))),
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
        // days, Timestamp(Microsecond) the i64 count).
        DataType::Date32 => read_le::<4>(bytes)
            .map(i32::from_le_bytes)
            .map(|v| erase_type(Date32Array::new_scalar(v))),
        DataType::Timestamp(TimeUnit::Microsecond, None) => read_le::<8>(bytes)
            .map(i64::from_le_bytes)
            .map(|v| erase_type(TimestampMicrosecondArray::new_scalar(v))),
        // A decimal's unscaled integer follows the column's physical storage;
        // the scalar's carrier follows the column's arrow type.
        DataType::Decimal64(precision, scale) => {
            let value = decimal_stat_value(bytes, physical_type)?;
            let array = Decimal64Array::new_scalar(i64::try_from(value).ok()?)
                .into_inner()
                .with_precision_and_scale(*precision, *scale)
                .ok()?;
            Some(Scalar::new(Arc::new(array) as ArrayRef))
        }
        DataType::Decimal128(precision, scale) => {
            let value = decimal_stat_value(bytes, physical_type)?;
            let array = Decimal128Array::new_scalar(value)
                .into_inner()
                .with_precision_and_scale(*precision, *scale)
                .ok()?;
            Some(Scalar::new(Arc::new(array) as ArrayRef))
        }
        _ => None,
    }
}

/// A decimal statistic's unscaled integer, decoded per the column's physical
/// storage: INT32/INT64 stats are little-endian, FIXED_LEN_BYTE_ARRAY stats
/// are big-endian two's complement in the declared length.
fn decimal_stat_value(bytes: &[u8], physical_type: i32) -> Option<i128> {
    use crate::parquet::types::thrift::general::Type as PhysicalType;
    if physical_type == PhysicalType::INT32 as i32 {
        bytes
            .try_into()
            .ok()
            .map(i32::from_le_bytes)
            .map(i128::from)
    } else if physical_type == PhysicalType::INT64 as i32 {
        bytes
            .try_into()
            .ok()
            .map(i64::from_le_bytes)
            .map(i128::from)
    } else if physical_type == PhysicalType::FIXED_LEN_BYTE_ARRAY as i32 {
        i128_from_be_bytes(bytes)
    } else {
        None
    }
}

/// Sign-extend a big-endian two's-complement integer of up to 16 bytes into
/// an `i128`. Returns `None` for byte lengths a `Decimal128` cannot hold.
fn i128_from_be_bytes(bytes: &[u8]) -> Option<i128> {
    if bytes.is_empty() || bytes.len() > 16 {
        return None;
    }
    let fill = if bytes[0] & 0x80 != 0 { 0xff } else { 0 };
    let mut buf = [fill; 16];
    buf[16 - bytes.len()..].copy_from_slice(bytes);
    Some(i128::from_be_bytes(buf))
}

/// Deserialises raw Thrift bytes into a [`FileMetaData`].
fn parse_footer_thrift(buf: &[u8]) -> Result<FileMetaData> {
    let mut prot = ThriftSliceInputProtocol::new(buf);
    FileMetaData::read_thrift(&mut prot)
        .map_err(|e| Error::IO(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))
}

/// Per-leaf schema facts collected while walking the footer's schema
/// elements, in depth-first (column-chunk) order.
#[derive(Debug)]
struct LeafSchemaInfo {
    /// Maximum definition level (one per optional ancestor plus the leaf).
    def_level: i16,
    /// The element's `type_length` (the byte width of a FIXED_LEN_BYTE_ARRAY
    /// value).
    type_length: Option<i32>,
}

/// Converts footer schema elements into an Arrow schema and per-leaf schema
/// facts.
fn schema_elements_to_arrow(
    elements: &[crate::parquet::types::thrift::footer::SchemaElement],
) -> Result<(Schema, Vec<LeafSchemaInfo>)> {
    if elements.is_empty() {
        return Err(Error::IO(io::Error::new(
            io::ErrorKind::InvalidData,
            "empty schema",
        )));
    }
    let mut leaf_infos = Vec::new();
    let mut cursor = 1; // element 0 is the root group
    let mut fields = Vec::new();
    for _ in 0..elements[0].num_children.unwrap_or(0) {
        let (field, next) = parse_schema_element(elements, cursor, 0, 0, &mut leaf_infos)?;
        fields.push(field);
        cursor = next;
    }
    Ok((Schema::new(fields), leaf_infos))
}

/// Deepest schema nesting accepted. Real schemas stay far below this (a
/// shredded variant adds two levels per path segment); the cap only exists so
/// a malformed or hostile footer can't overflow the stack through
/// [`parse_schema_element`]'s per-level recursion, which would abort the
/// process rather than fail the load.
const MAX_SCHEMA_DEPTH: usize = 128;

/// Reconcile a file's parsed schema with the table's declared column types,
/// matched by column name: a file may carry more columns than the table
/// declares (an ingest mapping can write a superset), and those pass through
/// untouched.
///
/// A declared VARCHAR may retype unannotated binary data as text. A declared
/// VARIANT accepts any struct shape because shredding adds fields per file.
///
/// Other declared types must match the file's storage type. Undeclared file
/// fields are unchanged.
fn apply_declared_types(schema: Schema, declared_columns: &[Column]) -> Result<Schema> {
    // Keyed case-insensitively: DuckDB resolves identifiers that way, so a
    // file written by another tool may spell a declared column differently
    // and must still reconcile with it.
    let declared_by_name: std::collections::HashMap<String, &Type> = declared_columns
        .iter()
        .map(|c| (c.name.to_lowercase(), &c.col_type))
        .collect();
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let Some(declared) = declared_by_name.get(&field.name().to_lowercase()) else {
                return Ok(field.clone());
            };
            let file_type = field.data_type();
            if **declared == Type::Utf8 && *file_type == DataType::BinaryView {
                return Ok(Arc::new(
                    field.as_ref().clone().with_data_type(DataType::Utf8View),
                ));
            }
            // A pivot timestamp counts microseconds (see `Type::Timestamp`).
            // A leaf that carries the TIMESTAMP annotation, which is what the
            // writer stamps, already parsed to `Timestamp(Microsecond)` and
            // matches below; any other unit was rejected when the footer was
            // read. A bare INT64 leaf says nothing about its own unit, so it is
            // re-labelled as the timestamp the table declares, which is how a
            // file written by another engine, or by an older pivot, reads.
            if **declared == Type::Timestamp && *file_type == DataType::Int64 {
                return Ok(Arc::new(
                    field
                        .as_ref()
                        .clone()
                        .with_data_type(DataType::Timestamp(TimeUnit::Microsecond, None)),
                ));
            }
            let matches_declared = match declared {
                Type::Variant => matches!(file_type, DataType::Struct(_)),
                _ => *file_type == planner::types::physical_arrow_type(declared),
            };
            if matches_declared {
                Ok(field.clone())
            } else {
                Err(Error::DeclaredTypeMismatch {
                    column: field.name().clone(),
                    declared: (*declared).clone(),
                    file_type: file_type.clone(),
                })
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Schema::new(fields))
}

/// Parse the subtree rooted at `elements[idx]`. `parent_def` is the definition
/// level contributed by ancestors (each optional ancestor adds one). Returns
/// the Arrow field and the index just past this subtree, appending one entry
/// per leaf to `leaf_infos` in depth-first (column-chunk) order.
fn parse_schema_element(
    elements: &[crate::parquet::types::thrift::footer::SchemaElement],
    idx: usize,
    parent_def: i16,
    depth: usize,
    leaf_infos: &mut Vec<LeafSchemaInfo>,
) -> Result<(Field, usize)> {
    use crate::parquet::types::thrift::footer::LogicalType;
    // The cursor is driven by each group's *claimed* child count, so a
    // malformed footer can point past the element list; fail the load
    // instead of panicking mid-fetch.
    let Some(elem) = elements.get(idx) else {
        return Err(Error::InvalidFooter(
            "schema group claims more children than the footer holds".into(),
        ));
    };
    if depth >= MAX_SCHEMA_DEPTH {
        return Err(Error::InvalidFooter(format!(
            "schema nesting exceeds {MAX_SCHEMA_DEPTH} levels"
        )));
    }
    // repetition_type: 0=REQUIRED, 1=OPTIONAL, 2=REPEATED.
    // REPEATED elements (the LIST/MAP encodings) need repetition-level
    // decoding, which the page decoder doesn't do; accepting them here would
    // silently misdecode their pages.
    if elem.repetition_type == Some(2) {
        return Err(Error::UnsupportedType(format!(
            "repeated field '{}' (LIST/MAP columns are not supported)",
            elem.name
        )));
    }
    let nullable = elem.repetition_type == Some(1);
    let def_level = parent_def + nullable as i16;

    match elem.num_children {
        Some(n) if n > 0 => {
            // Children can't outnumber the elements after this one; a bigger
            // claim is malformed (and would size the Vec from hostile input).
            if n as usize > elements.len() - idx - 1 {
                return Err(Error::InvalidFooter(
                    "schema group claims more children than the footer holds".into(),
                ));
            }
            let is_variant = elem.logical_type == Some(LogicalType::Variant);
            let mut children = Vec::with_capacity(n as usize);
            let mut cursor = idx + 1;
            for _ in 0..n {
                let (child, next) =
                    parse_schema_element(elements, cursor, def_level, depth + 1, leaf_infos)?;
                children.push(child);
                cursor = next;
            }
            let field = Field::new(&elem.name, DataType::Struct(children.into()), nullable);
            // Re-mark the group so a reconstructed variant re-tags VARIANT if
            // the table is re-written (e.g. by compaction).
            let field = if is_variant {
                field.with_metadata(variant_extension_metadata())
            } else {
                field
            };
            Ok((field, cursor))
        }
        _ => {
            leaf_infos.push(LeafSchemaInfo {
                def_level,
                type_length: elem.type_length,
            });
            let data_type = super::arrow_map::parquet_to_arrow(elem)?;
            Ok((Field::new(&elem.name, data_type, nullable), idx + 1))
        }
    }
}

/// The canonical Arrow extension type name for a Parquet variant. The reader
/// stamps it onto a VARIANT group's field and the writer looks for it to decide
/// which struct columns to write back as VARIANT, so both name it from here.
pub const VARIANT_EXTENSION_NAME: &str = "arrow.parquet.variant";

/// The Arrow field metadata that marks a struct as a Parquet variant.
pub fn variant_extension_metadata() -> std::collections::HashMap<String, String> {
    use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
    [
        (
            EXTENSION_TYPE_NAME_KEY.to_owned(),
            VARIANT_EXTENSION_NAME.to_owned(),
        ),
        (EXTENSION_TYPE_METADATA_KEY.to_owned(), String::new()),
    ]
    .into()
}

/// Whether `field` is a variant column — a struct carrying the Arrow variant
/// extension tag, stamped either by [`parse_schema_element`] when reading a
/// VARIANT group back or by a producer building one to write.
pub fn is_variant_field(field: &Field) -> bool {
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    matches!(field.data_type(), DataType::Struct(_))
        && field
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .map(String::as_str)
            == Some(VARIANT_EXTENSION_NAME)
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
        let table = ParquetTable::from_directory(test_dispatcher(), dir.path(), &[]).unwrap();
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

    fn column(name: &str, col_type: Type) -> Column {
        Column {
            name: name.to_string(),
            col_type,
        }
    }

    /// A column the table declares VARCHAR whose file leaf is unannotated
    /// binary is retyped to text; every other pairing is left alone.
    #[test]
    fn declared_varchar_retypes_an_unannotated_binary_column() {
        let schema = Schema::new(vec![
            Field::new("url", DataType::BinaryView, false),
            Field::new("id", DataType::Int64, false),
        ]);

        let declared = [column("url", Type::Utf8), column("id", Type::Int64)];
        let reconciled = apply_declared_types(schema, &declared).unwrap();

        assert_eq!(*reconciled.field(0).data_type(), DataType::Utf8View);
        assert_eq!(*reconciled.field(1).data_type(), DataType::Int64);
    }

    /// Matching is by column name: file columns the table doesn't declare
    /// pass through untouched (an ingest mapping can write a superset, in any
    /// order), and an empty declaration changes nothing.
    #[test]
    fn undeclared_file_columns_pass_through_untouched() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::BinaryView, false),
            Field::new("b", DataType::BinaryView, false),
        ]);

        let reconciled = apply_declared_types(schema.clone(), &[column("b", Type::Utf8)]).unwrap();
        let untouched = apply_declared_types(schema, &[]).unwrap();

        assert_eq!(*reconciled.field(0).data_type(), DataType::BinaryView);
        assert_eq!(*reconciled.field(1).data_type(), DataType::Utf8View);
        assert_eq!(*untouched.field(0).data_type(), DataType::BinaryView);
    }

    /// A declared VARIANT column accepts any struct shape (shredding widens
    /// the stored struct per file).
    #[test]
    fn declared_variant_accepts_any_struct_shape() {
        let doc = Field::new(
            "d",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("age", DataType::Int64, true),
                ]
                .into(),
            ),
            true,
        );
        let schema = Schema::new(vec![doc.clone()]);

        let reconciled = apply_declared_types(schema, &[column("d", Type::Variant)]).unwrap();

        assert_eq!(reconciled.field(0), &doc);
    }

    /// A file whose stored type contradicts the declaration fails the load,
    /// naming the offending column and both types.
    #[test]
    fn declared_type_mismatch_fails_naming_the_column() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("price", DataType::Utf8View, false),
        ]);

        let declared = [column("id", Type::Int64), column("price", Type::Float64)];
        let err = apply_declared_types(schema, &declared).unwrap_err();

        let message = err.to_string();
        assert!(
            message.contains("'price'")
                && message.contains("Float64")
                && message.contains("Utf8View"),
            "unhelpful mismatch error: {message}"
        );
    }

    /// Both timestamp storage flavors (pivot's plain INT64 count of
    /// microseconds and an annotated `Timestamp(Microsecond)`) reconcile to the
    /// one arrow type the executor reads, so a table mixing such files emits
    /// uniformly-typed batches.
    #[test]
    fn declared_timestamp_reconciles_both_storage_flavors() {
        let micros = DataType::Timestamp(TimeUnit::Microsecond, None);
        let schema = Schema::new(vec![
            Field::new("ts_plain", DataType::Int64, false),
            Field::new("ts_annotated", micros.clone(), false),
        ]);
        let declared = [
            column("ts_plain", Type::Timestamp),
            column("ts_annotated", Type::Timestamp),
        ];

        let reconciled = apply_declared_types(schema, &declared).unwrap();

        assert_eq!(*reconciled.field(0).data_type(), micros);
        assert_eq!(*reconciled.field(1).data_type(), micros);
    }

    /// DuckDB resolves identifiers case-insensitively, so a declared column
    /// must reconcile with a file column spelled in another case.
    #[test]
    fn declared_columns_match_file_columns_case_insensitively() {
        let schema = Schema::new(vec![Field::new("URL", DataType::BinaryView, false)]);

        let reconciled = apply_declared_types(schema, &[column("url", Type::Utf8)]).unwrap();

        assert_eq!(*reconciled.field(0).data_type(), DataType::Utf8View);
    }

    fn schema_element(
        name: &str,
        repetition_type: Option<i32>,
        num_children: Option<i32>,
        physical_type: Option<i32>,
    ) -> crate::parquet::types::thrift::footer::SchemaElement {
        crate::parquet::types::thrift::footer::SchemaElement {
            physical_type,
            type_length: None,
            repetition_type,
            name: name.to_string(),
            num_children,
            converted_type: None,
            scale: None,
            precision: None,
            logical_type: None,
        }
    }

    /// A REPEATED element (the LIST/MAP encoding) fails the load cleanly:
    /// the page decoder has no repetition-level support, so accepting it
    /// would silently misdecode.
    #[test]
    fn repeated_schema_elements_fail_the_load() {
        let elements = vec![
            schema_element("root", None, Some(1), None),
            schema_element("tags", Some(1), Some(1), None),
            schema_element("list", Some(2), Some(1), None),
            schema_element("element", Some(1), None, Some(2)),
        ];

        let err = schema_elements_to_arrow(&elements).unwrap_err();

        assert!(err.to_string().contains("not supported"), "{err}");
    }

    /// A group claiming more children than the footer holds is malformed and
    /// fails the load instead of indexing out of bounds.
    #[test]
    fn overclaimed_child_counts_fail_the_load() {
        let elements = vec![
            schema_element("root", None, Some(3), None),
            schema_element("a", Some(1), None, Some(2)),
        ];

        let err = schema_elements_to_arrow(&elements).unwrap_err();

        assert!(err.to_string().contains("more children"), "{err}");
    }

    /// Nesting past the cap fails the load instead of overflowing the stack
    /// (which would abort the process, not unwind).
    #[test]
    fn absurdly_deep_nesting_fails_the_load() {
        let mut elements = vec![schema_element("root", None, Some(1), None)];
        for i in 0..=MAX_SCHEMA_DEPTH {
            elements.push(schema_element(&format!("g{i}"), Some(1), Some(1), None));
        }
        elements.push(schema_element("leaf", Some(1), None, Some(2)));

        let err = schema_elements_to_arrow(&elements).unwrap_err();

        assert!(err.to_string().contains("nesting"), "{err}");
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
