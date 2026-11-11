//! The live data files of one partition of a table version, held once: every
//! file as one record, ordered by where the file starts on the table's sort
//! key, then by path. The path only makes the order total: files that start
//! at the same place on the sort key, which is every file of a table without
//! a sort key, come out in an order that is fixed for the version and means
//! nothing.
//!
//! A table is indexed a partition at a time. A partition is the files
//! written under one partition spec with one tuple, so an unpartitioned
//! table is a single partition. A partition's index is built from the
//! manifests that list a file of it, when something first asks for the
//! partition, and is kept by the process-wide
//! [`FileMetadataCache`](crate::files::file_metadata_cache::FileMetadataCache)
//! for as long as its budget allows.
//!
//! A manifest is decoded and each of its live entries in a wanted partition
//! moves into a record: the entry's data file as the spec crate decoded it,
//! with its partition tuple and the bounds and null counts of every column
//! the manifest recorded something for, so a record grows with the table's
//! width. Planning selects files from the records, and a commit rebuilds
//! the entry of a file it removes from one. What a decode learns of the
//! whole manifest, the partitions it holds, is kept as its
//! [`ManifestSummary`].
//!
//! Maintenance walks the same records: the stretch of the sort key a file
//! covers is kept as bytes that order as the values do, so ranges are
//! compared without decoding a bound.
//!
//! The index is persistent: the next version's is this one's with the files
//! of the manifests that left the list removed and those of the manifests
//! that joined it inserted, sharing every untouched record and tree node, so
//! a version costs what changed, and a query keeps the one it started on.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::ArrayRef;
use arrow_row::{RowConverter, SortField};
use iceberg::expr::BoundPredicate;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, Literal, Manifest, ManifestEntry,
    ManifestFile, ManifestStatus, PrimitiveLiteral, Schema, SchemaRef, Struct, TableMetadata,
};
use imbl::{HashMap as SharedMap, OrdMap};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::FileStats;
use parquet_engine::file_entry::FileEntry;
use planner::catalog::Column;
use planner::expression::Expression;
use planner::types::{Type, physical_arrow_type};

use crate::files::data_files::{DataFileEntry, literal_to_scalar, read_partition_values};
use crate::files::pruning::{self, FieldRange, PartitionPruner, StoredColumn};
use crate::metadata::layout::TableLocation;
use crate::{Error, Result};

/// One partition of a table: the files written under one partition spec
/// with one tuple.
#[derive(Debug)]
pub struct Partition {
    /// The spec's id, then the tuple, each value in a form whose bytes order
    /// as the values do: equal for exactly the files of this partition.
    key: Arc<[u8]>,
    spec_id: i32,
    tuple: Struct,
}

impl Partition {
    /// The partition as bytes equal for exactly the files of it.
    pub fn key(&self) -> &Arc<[u8]> {
        &self.key
    }

    /// The partition spec the tuple is laid out by.
    pub fn spec_id(&self) -> i32 {
        self.spec_id
    }

    pub fn tuple(&self) -> &Struct {
        &self.tuple
    }
}

/// One shared copy of each partition a table's manifests were seen to hold,
/// by key.
#[derive(Default)]
pub(crate) struct PartitionInterner {
    partitions: Mutex<HashMap<Arc<[u8]>, Arc<Partition>>>,
}

impl PartitionInterner {
    /// The shared copy of the partition of `tuple` under the spec `spec_id`.
    pub(crate) fn intern(&self, spec_id: i32, tuple: &Struct) -> Arc<Partition> {
        let key = encode_partition(spec_id, tuple);
        let mut partitions = self.partitions.lock().unwrap();
        if let Some(known) = partitions.get(&key) {
            return Arc::clone(known);
        }
        let partition = Arc::new(Partition {
            key: Arc::clone(&key),
            spec_id,
            tuple: tuple.clone(),
        });
        partitions.insert(key, Arc::clone(&partition));
        partition
    }

    /// The partition with the key `key`, when a manifest was seen to hold it.
    pub(crate) fn find(&self, key: &[u8]) -> Option<Arc<Partition>> {
        self.partitions.lock().unwrap().get(key).cloned()
    }
}

/// What decoding a data manifest learns of the whole of it, kept for as
/// long as some version lists the manifest.
pub struct ManifestSummary {
    /// Each partition the manifest holds a live file of, in key order, with
    /// the highest data sequence number among those files.
    partitions: Vec<(Arc<Partition>, i64)>,
}

impl ManifestSummary {
    /// Each partition the manifest holds a live file of, in key order, with
    /// the highest data sequence number among those files.
    pub fn partitions(&self) -> &[(Arc<Partition>, i64)] {
        &self.partitions
    }

    /// Whether the manifest holds a live file of the partition `key`.
    pub fn holds_partition(&self, key: &[u8]) -> bool {
        self.partitions
            .binary_search_by(|(partition, _)| partition.key.as_ref().cmp(key))
            .is_ok()
    }
}

/// A decoded data manifest: what was learned of the whole of it, and the
/// live files of the partitions the decode was asked for.
pub(crate) struct DecodedManifest {
    pub summary: ManifestSummary,
    /// The live files of each wanted partition.
    pub files: FilesByPartition,
}

/// Files by the key of the partition they sit in.
pub(crate) type FilesByPartition = HashMap<Arc<[u8]>, Vec<Arc<IndexedFile>>>;

/// What reading a manifest entry into a record needs of its table.
pub(crate) struct DecodeContext<'t> {
    pub table: &'t str,
    pub encoder: &'t SortKeyEncoder,
    pub metadata: &'t TableMetadata,
    pub location: &'t TableLocation,
    pub partitions: &'t PartitionInterner,
}

/// Where a file sorts among its table's files: by its partition, then by the
/// lower bound of the table's sort key in it, then by its path, which only
/// makes the order total.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FileKey {
    /// The partition spec's id, then the tuple, each value in a form whose
    /// bytes order as the values do.
    partition: Arc<[u8]>,
    /// The sort key's lower bound across the file as one row whose bytes
    /// order as the tuples do; a column without a bound orders first. Empty
    /// for a table without a sort key.
    sort_minimum: Arc<[u8]>,
    path: Arc<str>,
}

impl FileKey {
    /// The file's partition, as bytes equal for exactly the files of one
    /// partition under one spec.
    pub fn partition(&self) -> &[u8] {
        &self.partition
    }
}

/// Encodes a bound of a table's sort key in a file as bytes that order as
/// the bounds do.
pub(crate) struct SortKeyEncoder {
    /// The sort columns, each by field id and Pivot type, with the converter
    /// over their Arrow types; `None` for a table without a sort key, or one
    /// sorted by a column whose type has no ordered byte form, whose files
    /// then order by commit alone.
    columns: Option<(Vec<(i32, Type)>, RowConverter)>,
}

impl SortKeyEncoder {
    /// The encoder for a table sorted by `sort_by`, columns among `columns`
    /// whose field ids `metadata`'s current schema gives.
    pub(crate) fn new(
        metadata: &TableMetadata,
        columns: &[Column],
        sort_by: &[String],
    ) -> Result<Self> {
        let schema = metadata.current_schema();
        let sort_columns: Vec<(i32, Type)> = sort_by
            .iter()
            .filter_map(|name| {
                let field_id = schema.field_id_by_name(name)?;
                let column = columns.iter().find(|column| &column.name == name)?;
                Some((field_id, column.col_type.clone()))
            })
            .collect();
        let fields: Vec<SortField> = sort_columns
            .iter()
            .map(|(_, pivot_type)| SortField::new(physical_arrow_type(pivot_type)))
            .collect();
        if fields.is_empty() || !RowConverter::supports_fields(&fields) {
            return Ok(Self { columns: None });
        }
        let converter = RowConverter::new(fields).map_err(Error::arrow)?;
        Ok(Self {
            columns: Some((sort_columns, converter)),
        })
    }

    /// The field the sort key leads with, when the key has an ordered byte
    /// form.
    fn find_leading_field(&self) -> Option<i32> {
        let (columns, _) = self.columns.as_ref()?;
        columns.first().map(|(field_id, _)| *field_id)
    }

    /// The key bytes of a file whose manifest entry bounds the field
    /// `field_id`, on one side, by `bound(field_id)`.
    fn encode<'a>(&self, bound: impl Fn(i32) -> Option<&'a Datum>) -> Result<Arc<[u8]>> {
        let Some((columns, converter)) = &self.columns else {
            return Ok(Arc::from([]));
        };
        let arrays: Vec<ArrayRef> = columns
            .iter()
            .map(|(field_id, pivot_type)| {
                let arrow_type = physical_arrow_type(pivot_type);
                bound(*field_id)
                    .and_then(|bound| literal_to_scalar(bound.literal(), pivot_type))
                    .map(|scalar| scalar.into_inner())
                    .and_then(|array| {
                        if array.data_type() == &arrow_type {
                            Some(array)
                        } else {
                            arrow_cast::cast(array.as_ref(), &arrow_type).ok()
                        }
                    })
                    .unwrap_or_else(|| arrow_array::new_null_array(&arrow_type, 1))
            })
            .collect();
        let rows = converter.convert_columns(&arrays).map_err(Error::arrow)?;
        Ok(Arc::from(rows.row(0).as_ref()))
    }
}

/// One live data file of a table version.
pub struct IndexedFile {
    /// The file's path under the table's data root, and its size.
    pub file: FileRef,
    /// The file as its manifest entry describes it: its row count, its
    /// partition tuple, the bounds and null counts of its columns by field
    /// id, and the sort order its rows are in.
    pub data: DataFile,
    /// The partition the file sits in: the spec its manifest was written
    /// under, and its tuple.
    pub partition: Arc<Partition>,
    /// The schema the file's manifest was written under: the types its
    /// bounds are recorded in, and the fields the file has.
    pub schema_id: i32,
    /// The id of the file's first row, its manifest's running assignment
    /// where the entry leaves it implicit; `None` in a table whose rows
    /// have no ids, and for a file this process swapped in that no manifest
    /// of the version records yet.
    pub first_row_id: Option<i64>,
    key: FileKey,
    /// The sort key's upper bound across the file, in the byte form of the
    /// key's lower bound; `None` when the table has no sort key with an
    /// ordered byte form, or the entry does not bound the column the key
    /// leads with on both sides.
    sort_maximum: Option<Arc<[u8]>>,
    /// An estimate of the memory the record holds, itself included.
    heap_bytes: u64,
}

impl IndexedFile {
    /// The file `data` describes, of `partition`, recorded by a manifest
    /// written under the schema `schema_id`.
    fn new(
        data: DataFile,
        schema_id: i32,
        partition: Arc<Partition>,
        first_row_id: Option<i64>,
        encoder: &SortKeyEncoder,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<Self> {
        let path = location.file_path_for_uri(data.file_path())?;
        let stored_schema = metadata.schema_by_id(schema_id);
        let read_bound =
            |field_id: i32, lower: bool| find_bound(&data, stored_schema, field_id, lower);
        let sort_minimum = encoder.encode(|field_id| read_bound(field_id, true))?;
        let leading_field_is_bounded = encoder.find_leading_field().is_some_and(|field_id| {
            read_bound(field_id, true).is_some() && read_bound(field_id, false).is_some()
        });
        let sort_maximum = if leading_field_is_bounded {
            Some(encoder.encode(|field_id| read_bound(field_id, false))?)
        } else {
            None
        };
        let key = FileKey {
            partition: Arc::clone(&partition.key),
            sort_minimum,
            path: Arc::from(path.as_str()),
        };
        let mut file = Self {
            file: FileRef {
                path,
                size: data.file_size_in_bytes(),
            },
            data,
            partition,
            schema_id,
            first_row_id,
            key,
            sort_maximum,
            heap_bytes: 0,
        };
        file.heap_bytes = file.estimate_heap_bytes();
        Ok(file)
    }

    /// Decode the data manifest `file`, taken out of `manifest`: summarize
    /// the whole of it, and keep the live files of the partitions `wants`
    /// accepts, with everything the manifest leaves implicit filled in.
    /// First row ids are assigned the way the spec inherits them: an entry
    /// without one gets the manifest's, advanced by the row counts of the
    /// entries assigned before it. A delete file among a data manifest's
    /// entries, or a file that is not Parquet, is refused.
    pub(crate) fn decode_manifest(
        file: &ManifestFile,
        manifest: Manifest,
        context: &DecodeContext<'_>,
        wants: &mut dyn FnMut(&Partition) -> Result<bool>,
    ) -> Result<DecodedManifest> {
        let schema_id = manifest.metadata().schema_id();
        let partition_spec_id = manifest.metadata().partition_spec().spec_id();
        let mut next_row_id = file.first_row_id.map(|first| first as i64);
        let mut partitions: HashMap<Arc<[u8]>, (Arc<Partition>, i64)> = HashMap::new();
        let mut files = FilesByPartition::new();
        // The partition of the entry before and whether it is wanted: a
        // manifest's entries mostly come in runs of one partition.
        let mut latest: Option<(Arc<Partition>, bool)> = None;
        let (entries, _) = manifest.into_parts();
        for entry in entries {
            let data = entry.data_file();
            let first_row_id = match data.first_row_id() {
                Some(assigned) => Some(assigned),
                None if entry.is_alive() => {
                    let assigned = next_row_id;
                    if let Some(next) = next_row_id.as_mut() {
                        *next += data.record_count() as i64;
                    }
                    assigned
                }
                None => None,
            };
            if !entry.is_alive() {
                continue;
            }
            if data.content_type() != DataContentType::Data {
                return Err(Error::DeleteFile {
                    file: data.file_path().to_string(),
                });
            }
            if data.file_format() != DataFileFormat::Parquet {
                return Err(Error::NonParquetFile {
                    table: context.table.to_string(),
                    file: data.file_path().to_string(),
                    format: data.file_format().to_string(),
                });
            }
            let (partition, is_wanted) = match &latest {
                Some((partition, is_wanted))
                    if partition.spec_id == partition_spec_id
                        && &partition.tuple == data.partition() =>
                {
                    (Arc::clone(partition), *is_wanted)
                }
                _ => {
                    let partition = context
                        .partitions
                        .intern(partition_spec_id, data.partition());
                    let is_wanted = wants(&partition)?;
                    latest = Some((Arc::clone(&partition), is_wanted));
                    (partition, is_wanted)
                }
            };
            let sequence_number = read_sequence_number(&entry, file, context.metadata)?;
            partitions
                .entry(Arc::clone(&partition.key))
                .and_modify(|(_, newest)| *newest = (*newest).max(sequence_number))
                .or_insert_with(|| (Arc::clone(&partition), sequence_number));
            if !is_wanted {
                continue;
            }
            // The manifest was parsed for this decode alone, so its entries
            // are its own to give away.
            let entry = Arc::unwrap_or_clone(entry);
            files
                .entry(Arc::clone(&partition.key))
                .or_default()
                .push(Arc::new(Self::new(
                    entry.data_file,
                    schema_id,
                    partition,
                    first_row_id,
                    context.encoder,
                    context.metadata,
                    context.location,
                )?));
        }
        let mut partitions: Vec<(Arc<Partition>, i64)> = partitions.into_values().collect();
        partitions.sort_unstable_by(|(left, _), (right, _)| left.key.cmp(&right.key));
        Ok(DecodedManifest {
            summary: ManifestSummary { partitions },
            files,
        })
    }

    /// `entry`, a file this process committed that no manifest of the version
    /// records yet: it orders among the committed files of its partition as
    /// they do, by where it starts on the sort key and its path.
    pub(crate) fn uncommitted(
        entry: &DataFileEntry,
        encoder: &SortKeyEncoder,
        metadata: &TableMetadata,
        location: &TableLocation,
        partitions: &PartitionInterner,
    ) -> Result<Self> {
        let data = entry.create_data_file(metadata, location)?;
        let partition = partitions.intern(
            metadata.default_partition_spec().spec_id(),
            data.partition(),
        );
        Self::new(
            data,
            metadata.current_schema().schema_id(),
            partition,
            None,
            encoder,
            metadata,
            location,
        )
    }

    /// An estimate of the memory the record holds, itself included: its
    /// paths, its key bytes, and every map and value of its data file, each
    /// at the size an allocation of it takes, with what the index's own
    /// maps spend on one file.
    fn estimate_heap_bytes(&self) -> u64 {
        /// What the ordered map, the map by path and a manifest's file list
        /// spend on one file between them.
        const INDEX_ENTRY_BYTES: usize = 3 * size_of::<FileKey>() + 96;

        let data = &self.data;
        let count_maps = [
            data.column_sizes(),
            data.value_counts(),
            data.null_value_counts(),
            data.nan_value_counts(),
        ];
        let count_bytes: usize = count_maps
            .iter()
            .map(|counts| estimate_map_bytes::<u64>(counts.capacity()))
            .sum();
        let bound_bytes: usize = [data.lower_bounds(), data.upper_bounds()]
            .iter()
            .map(|bounds| {
                estimate_map_bytes::<Datum>(bounds.capacity())
                    + bounds
                        .values()
                        .map(|bound| estimate_literal_heap_bytes(bound.literal()))
                        .sum::<usize>()
            })
            .sum();
        let tuple = data.partition().fields();
        let tuple_bytes: usize = round_to_allocation(size_of_val(tuple))
            + tuple
                .iter()
                .map(|value| match value {
                    Some(Literal::Primitive(value)) => estimate_literal_heap_bytes(value),
                    _ => 0,
                })
                .sum::<usize>();
        // A shared slice is allocated with its two reference counts.
        let shared = |length: usize| round_to_allocation(length + size_of::<[usize; 2]>());
        let key_bytes = shared(self.key.sort_minimum.len())
            + self
                .sort_maximum
                .as_ref()
                .map_or(0, |bound| shared(bound.len()))
            + shared(self.key.path.len());
        let total = shared(size_of::<Self>())
            + INDEX_ENTRY_BYTES
            + round_to_allocation(self.file.path.as_str().len())
            + round_to_allocation(data.file_path().len())
            + key_bytes
            + count_bytes
            + bound_bytes
            + tuple_bytes
            + data
                .key_metadata()
                .map_or(0, |key| round_to_allocation(key.len()))
            + data
                .split_offsets()
                .map_or(0, |offsets| round_to_allocation(size_of_val(offsets)));
        total as u64
    }

    /// An estimate of the memory the record holds, itself included.
    pub fn heap_bytes(&self) -> u64 {
        self.heap_bytes
    }

    pub fn key(&self) -> &FileKey {
        &self.key
    }

    /// The stretch of the table's sort key the file covers, its lowest and
    /// highest keys as bytes that order as the keys do. `None` for a table
    /// without a sort key, and for a file whose entry does not bound the
    /// column the key leads with.
    pub fn sort_range(&self) -> Option<(&[u8], &[u8])> {
        let sort_maximum = self.sort_maximum.as_deref()?;
        Some((&self.key.sort_minimum, sort_maximum))
    }

    /// The rows the file holds, deleted ones included.
    pub fn record_count(&self) -> u64 {
        self.data.record_count()
    }

    /// The id of the table sort order the file's rows are in, as its
    /// manifest entry records it; `None` for an entry that says nothing.
    pub fn sort_order_id(&self) -> Option<i32> {
        self.data.sort_order_id()
    }

    /// The statistics the entry recorded for `columns`, as the datastore keys
    /// them: by column name, as typed scalars of the column's Pivot type.
    pub fn file_stats<'c>(
        &self,
        metadata: &TableMetadata,
        columns: impl Iterator<Item = &'c Column>,
    ) -> FileStats {
        let schema = metadata.current_schema();
        let stored_schema = metadata.schema_by_id(self.schema_id);
        let mut stats = FileStats {
            num_records: i64::try_from(self.data.record_count()).ok(),
            min_values: Default::default(),
            max_values: Default::default(),
            null_counts: Default::default(),
        };
        for column in columns {
            let Some(field_id) = schema.field_id_by_name(&column.name) else {
                continue;
            };
            let read = |lower: bool| {
                find_bound(&self.data, stored_schema, field_id, lower)
                    .and_then(|bound| literal_to_scalar(bound.literal(), &column.col_type))
                    .map(|scalar| scalar.into_inner())
            };
            if let (Some(min), Some(max)) = (read(true), read(false)) {
                stats.min_values.insert(column.name.clone(), min);
                stats.max_values.insert(column.name.clone(), max);
            }
            let null_count = self.data.null_value_counts().get(&field_id);
            if let Some(nulls) = null_count.and_then(|nulls| i64::try_from(*nulls).ok()) {
                stats.null_counts.insert(column.name.clone(), nulls);
            }
        }
        stats
    }

    /// The file as a commit records and removes it: its tuple and the
    /// statistics of every declared column, as the datastore holds them.
    pub fn to_entry(&self, metadata: &TableMetadata, columns: &[Column]) -> Result<DataFileEntry> {
        Ok(DataFileEntry {
            file: self.file.clone(),
            partition: read_partition_values(
                metadata,
                columns,
                self.data.partition(),
                &self.file.path,
            )?,
            stats: Some(Arc::new(self.file_stats(metadata, columns.iter()))),
            sort_order_id: self.data.sort_order_id(),
            first_row_id: self.first_row_id,
        })
    }

    /// The file as the datastore's planners read it: its tuple, its row
    /// count, its bounds on `bounded_columns`, the columns files are laid
    /// out by, and `sorted_by`, the columns the caller read its rows to be
    /// in order of.
    pub fn to_datastore_entry(
        &self,
        metadata: &TableMetadata,
        columns: &[Column],
        bounded_columns: &[String],
        sorted_by: Arc<[String]>,
    ) -> Result<FileEntry> {
        let bounded = columns
            .iter()
            .filter(|column| bounded_columns.contains(&column.name));
        Ok(FileEntry {
            file: self.file.clone(),
            partition: read_partition_values(
                metadata,
                columns,
                self.data.partition(),
                &self.file.path,
            )?,
            stats: Some(Arc::new(self.file_stats(metadata, bounded))),
            sorted_by,
        })
    }
}

/// The data sequence number of the live `entry` of the manifest `manifest`:
/// its own, or the manifest's for an ADDED entry, which leaves it implicit.
fn read_sequence_number(
    entry: &ManifestEntry,
    manifest: &ManifestFile,
    metadata: &TableMetadata,
) -> Result<i64> {
    match entry.sequence_number() {
        Some(number) => Ok(number),
        None if entry.status() == ManifestStatus::Added => Ok(manifest.sequence_number),
        // A version 1 table has no sequence numbers: every entry reads as
        // the spec's inherited 0.
        None if metadata.format_version() == iceberg::spec::FormatVersion::V1 => Ok(0),
        None => Err(Error::InvalidDataFile {
            file: entry.data_file().file_path().to_string(),
            message: "an existing entry carries no sequence number".to_string(),
        }),
    }
}

/// An estimate of what a hash map with room for `capacity` entries of an
/// `i32` key and a `Value` allocates: its buckets, a power of two filled to
/// seven eighths, each with one byte of control, and a group of control
/// bytes after the last.
fn estimate_map_bytes<Value>(capacity: usize) -> usize {
    /// The control bytes a map keeps past its last bucket.
    const TRAILING_CONTROL_BYTES: usize = 16;

    if capacity == 0 {
        return 0;
    }
    let buckets = (capacity * 8).div_ceil(7).next_power_of_two();
    round_to_allocation(buckets * (size_of::<(i32, Value)>() + 1) + TRAILING_CONTROL_BYTES)
}

/// What `literal` holds outside its own value: the bytes of a string or a
/// binary.
fn estimate_literal_heap_bytes(literal: &PrimitiveLiteral) -> usize {
    match literal {
        PrimitiveLiteral::String(value) => round_to_allocation(value.len()),
        PrimitiveLiteral::Binary(value) => round_to_allocation(value.len()),
        _ => 0,
    }
}

/// What an allocation of `bytes` takes: an allocator serves a request from
/// a size class, of which there are four to each doubling of the size, and
/// none below sixteen bytes but eight.
fn round_to_allocation(bytes: usize) -> usize {
    match bytes {
        0 => 0,
        1..=8 => 8,
        9..=16 => 16,
        _ => {
            let doubling = (bytes - 1).ilog2();
            bytes.next_multiple_of(1 << doubling.saturating_sub(2).max(4))
        }
    }
}

/// The bound `data` records for the field `field_id`, below when `lower`
/// and above otherwise, `stored_schema` being the schema its manifest was
/// written under: `None` for a field that schema does not have as a
/// primitive, and for a schema the table no longer lists.
fn find_bound<'a>(
    data: &'a DataFile,
    stored_schema: Option<&SchemaRef>,
    field_id: i32,
    lower: bool,
) -> Option<&'a Datum> {
    let field = stored_schema?.field_by_id(field_id)?;
    field.field_type.as_primitive_type()?;
    let bounds = if lower {
        data.lower_bounds()
    } else {
        data.upper_bounds()
    };
    bounds.get(&field_id)
}

/// A partition tuple under its spec's id, as bytes that are equal exactly for
/// equal tuples of one spec and order as the tuples do: a NULL first, an
/// integer by value, a string by its bytes.
fn encode_partition(partition_spec_id: i32, partition: &Struct) -> Arc<[u8]> {
    let mut key = Vec::new();
    key.extend_from_slice(&(partition_spec_id as u32 ^ (1 << 31)).to_be_bytes());
    for value in partition.fields() {
        let Some(Literal::Primitive(value)) = value else {
            key.push(0);
            continue;
        };
        key.push(1);
        encode_value(&mut key, value);
    }
    Arc::from(key)
}

/// Append `value` to `key` in a form whose bytes order as the values of one
/// type do, and that what follows cannot be read as part of.
fn encode_value(key: &mut Vec<u8>, value: &PrimitiveLiteral) {
    match value {
        PrimitiveLiteral::Boolean(value) => key.push(u8::from(*value)),
        PrimitiveLiteral::Int(value) => {
            key.extend_from_slice(&(*value as u32 ^ (1 << 31)).to_be_bytes());
        }
        PrimitiveLiteral::Long(value) => {
            key.extend_from_slice(&(*value as u64 ^ (1 << 63)).to_be_bytes());
        }
        PrimitiveLiteral::Float(value) => {
            key.extend_from_slice(&order_float_bits(u64::from(value.0.to_bits()) << 32));
        }
        PrimitiveLiteral::Double(value) => {
            key.extend_from_slice(&order_float_bits(value.0.to_bits()));
        }
        PrimitiveLiteral::Int128(value) => {
            key.extend_from_slice(&(*value as u128 ^ (1 << 127)).to_be_bytes());
        }
        PrimitiveLiteral::UInt128(value) => key.extend_from_slice(&value.to_be_bytes()),
        // A length after the bytes keeps one value's bytes apart from
        // the next field's, and a prefix ordered before what extends it.
        PrimitiveLiteral::String(value) => extend_with_bytes(key, value.as_bytes()),
        PrimitiveLiteral::Binary(value) => extend_with_bytes(key, value),
        PrimitiveLiteral::AboveMax => key.push(u8::MAX),
        PrimitiveLiteral::BelowMin => key.push(0),
    }
}

/// The bits of a float, sign in the top bit, as bytes that order as the
/// floats do: a negative value's bits are inverted, a positive one's sign bit
/// set.
fn order_float_bits(bits: u64) -> [u8; 8] {
    let ordered = if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    };
    ordered.to_be_bytes()
}

/// Append `bytes` so that what follows cannot be read as part of them: each
/// zero byte escaped, then a terminator below every escaped byte.
fn extend_with_bytes(key: &mut Vec<u8>, bytes: &[u8]) {
    for byte in bytes {
        match byte {
            0 => key.extend_from_slice(&[0, u8::MAX]),
            byte => key.push(*byte),
        }
    }
    key.extend_from_slice(&[0, 0]);
}

/// The live data files of one partition of one table version. Cloning
/// shares everything; an edit copies only what it touches.
#[derive(Clone, Default)]
pub struct FileIndex {
    files: OrdMap<FileKey, Arc<IndexedFile>>,
    by_path: SharedMap<Arc<str>, Arc<IndexedFile>>,
    /// The partition's files in each data manifest that lists one, by the
    /// manifest's path.
    manifests: SharedMap<Arc<str>, Arc<[Arc<IndexedFile>]>>,
    /// The sum of the files' [`IndexedFile::heap_bytes`].
    heap_bytes: u64,
}

impl FileIndex {
    /// The index of a partition at the version whose data manifests holding
    /// a file of it are `listed`, built on `previous`, the partition's
    /// index at another version: the files of the manifests not listed
    /// leave, and those of the manifests `previous` did not hold, which
    /// `take_files` hands over, join. A manifest is never rewritten in
    /// place, so one both versions list is not asked for.
    pub(crate) fn advance(
        previous: Option<&FileIndex>,
        listed: &[Arc<str>],
        mut take_files: impl FnMut(&Arc<str>) -> Vec<Arc<IndexedFile>>,
    ) -> FileIndex {
        let mut index = previous.cloned().unwrap_or_default();
        let gone: Vec<Arc<str>> = index
            .manifests
            .keys()
            .filter(|path| !listed.contains(path))
            .cloned()
            .collect();
        for path in gone {
            let files = index
                .manifests
                .remove(&path)
                .expect("the manifest was just listed among those held");
            for file in files.iter() {
                index.remove(&file.file.path);
            }
        }
        for manifest in listed {
            if index.manifests.contains_key(manifest) {
                continue;
            }
            let files = take_files(manifest);
            for file in &files {
                index.insert(Arc::clone(file));
            }
            index
                .manifests
                .insert(Arc::clone(manifest), Arc::from(files));
        }
        index
    }

    /// Whether the index was built from exactly the manifests `listed`.
    pub(crate) fn is_built_from(&self, listed: &[Arc<str>]) -> bool {
        self.manifests.len() == listed.len()
            && listed
                .iter()
                .all(|manifest| self.manifests.contains_key(manifest))
    }

    /// Whether the index holds the files of the manifest at `manifest_path`.
    pub(crate) fn holds_manifest(&self, manifest_path: &str) -> bool {
        self.manifests.contains_key(manifest_path)
    }

    /// Add `file`, replacing a file already at its path.
    pub fn insert(&mut self, file: Arc<IndexedFile>) {
        self.remove(&file.file.path);
        self.heap_bytes += file.heap_bytes;
        self.by_path
            .insert(Arc::clone(&file.key.path), Arc::clone(&file));
        self.files.insert(file.key.clone(), file);
    }

    /// Drop the file at `path`, if the index has one.
    pub fn remove(&mut self, path: &ObjectPath) -> Option<Arc<IndexedFile>> {
        let file = self.by_path.remove(path.as_str())?;
        self.files.remove(&file.key);
        self.heap_bytes -= file.heap_bytes;
        Some(file)
    }

    /// The file at `path`.
    pub fn find(&self, path: &ObjectPath) -> Option<&Arc<IndexedFile>> {
        self.by_path.get(path.as_str())
    }

    /// Every file, in key order.
    pub fn files(&self) -> impl Iterator<Item = &Arc<IndexedFile>> {
        self.files.values()
    }

    /// An estimate of the memory the index holds: the sum of its files'.
    pub fn heap_bytes(&self) -> u64 {
        self.heap_bytes
    }
}

/// What a scan's filters require of a table's partitions and files,
/// resolved once for all of them: a file is described by its column bounds,
/// and by the partition values of the spec it was written under when the
/// table lists that spec.
pub(crate) struct FileSelector {
    row_filters: Vec<BoundPredicate>,
    /// What the filters require of a tuple of each partition spec the table
    /// lists, in spec id order.
    partition_pruners: Vec<(i32, PartitionPruner)>,
    /// How the files of each schema the table lists record the columns the
    /// filters compare, in schema id order.
    columns_per_schema: Vec<(i32, Vec<StoredColumn>)>,
    columns_of_unlisted_schema: Vec<StoredColumn>,
}

impl FileSelector {
    /// What `filters` require of the files of the table `metadata`
    /// describes.
    pub(crate) fn new(metadata: &TableMetadata, filters: &[Expression]) -> iceberg::Result<Self> {
        let schema = metadata.current_schema();
        let row_filters = pruning::bind_filters(schema, filters);
        let mut partition_pruners = Vec::new();
        for partition_spec in metadata.partition_specs_iter() {
            let pruner = PartitionPruner::new(schema, partition_spec, &row_filters)?;
            partition_pruners.push((partition_spec.spec_id(), pruner));
        }
        partition_pruners.sort_unstable_by_key(|(spec_id, _)| *spec_id);
        let compared_columns = pruning::list_compared_fields(&row_filters);
        let resolve_columns = |file_schema: Option<&Schema>| -> Vec<StoredColumn> {
            compared_columns
                .iter()
                .map(|column| StoredColumn::resolve(column, file_schema))
                .collect()
        };
        let mut columns_per_schema: Vec<(i32, Vec<StoredColumn>)> = metadata
            .schemas_iter()
            .map(|file_schema| (file_schema.schema_id(), resolve_columns(Some(file_schema))))
            .collect();
        columns_per_schema.sort_unstable_by_key(|(schema_id, _)| *schema_id);
        let columns_of_unlisted_schema = resolve_columns(None);
        Ok(Self {
            row_filters,
            partition_pruners,
            columns_per_schema,
            columns_of_unlisted_schema,
        })
    }

    /// The filters bound to the table's current schema.
    pub(crate) fn row_filters(&self) -> &[BoundPredicate] {
        &self.row_filters
    }

    /// Whether a file of `partition` may hold a row satisfying every filter,
    /// by the partition's tuple alone. `values` is scratch space the caller
    /// keeps from one partition to the next.
    pub(crate) fn may_partition_match(
        &self,
        partition: &Partition,
        values: &mut Vec<FieldRange>,
    ) -> iceberg::Result<bool> {
        match find_by_id(&self.partition_pruners, partition.spec_id) {
            Some(pruner) => pruner.may_match(&partition.tuple, values),
            None => Ok(true),
        }
    }

    /// The files of `index`, a partition whose tuple the filters keep, that
    /// are not empty and whose column bounds may hold a row satisfying
    /// every filter, in key order.
    pub(crate) fn select<'s>(
        &'s self,
        index: &'s FileIndex,
    ) -> impl Iterator<Item = &'s Arc<IndexedFile>> + 's {
        let mut column_ranges = Vec::new();
        index.files().filter(move |file| {
            if file.data.record_count() == 0 {
                return false;
            }
            let columns = match find_by_id(&self.columns_per_schema, file.schema_id) {
                Some(columns) => columns,
                None => &self.columns_of_unlisted_schema,
            };
            column_ranges.clear();
            column_ranges.extend(columns.iter().map(|column| column.read_range(&file.data)));
            self.row_filters
                .iter()
                .all(|row_filter| pruning::may_match(row_filter, &column_ranges))
        })
    }
}

/// The entry under `id` of `entries`, which are in id order.
fn find_by_id<Entry>(entries: &[(i32, Entry)], id: i32) -> Option<&Entry> {
    let position = entries
        .binary_search_by_key(&id, |(entry_id, _)| *entry_id)
        .ok()?;
    Some(&entries[position].1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_tuples_order_as_their_values_do() {
        let tuple = |values: Vec<Option<Literal>>| encode_partition(0, &Struct::from_iter(values));
        let int = |value: i32| Some(Literal::int(value));
        let text = |value: &str| Some(Literal::string(value));

        let mut keys = vec![
            tuple(vec![int(7), text("b")]),
            tuple(vec![int(-3), text("z")]),
            tuple(vec![int(7), text("ab")]),
            tuple(vec![None, text("a")]),
            tuple(vec![int(7), text("a")]),
        ];
        keys.sort();

        assert_eq!(
            keys,
            [
                tuple(vec![None, text("a")]),
                tuple(vec![int(-3), text("z")]),
                tuple(vec![int(7), text("a")]),
                tuple(vec![int(7), text("ab")]),
                tuple(vec![int(7), text("b")]),
            ]
        );
    }
}
