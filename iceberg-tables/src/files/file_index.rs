//! The live data files of one partition of a table version, held once: every
//! file as one record, in the order a partition's files stand in
//! ([`table_file_index::file_index`]): by where the file starts on the
//! table's sort key, then by the sequence number of the commit that added
//! it, then by path.
//!
//! A table is indexed a partition at a time. A partition is the files
//! written under one partition spec with one tuple, so an unpartitioned
//! table is a single partition. A partition's index is built from the
//! manifests that list a file of it, when something first asks for the
//! partition, and is kept by the process-wide
//! [`FileMetadataCache`](table_file_index::FileMetadataCache) for as long as
//! its budget allows.
//!
//! A manifest is decoded and each of its live entries in a wanted partition
//! moves into a record: the entry's data file as the spec crate decoded it,
//! with its partition tuple and the bounds and null counts of every column
//! the manifest recorded something for, so a record grows with the table's
//! width. A scan selects its files from the records, and a commit rebuilds
//! the entry of a file it removes from one. Nothing else of a decoded
//! manifest is kept: which manifests a partition's files are looked for in
//! is read off the manifest list every time
//! (`ListedManifest::may_hold_partition`).
//!
//! Maintenance walks the facts every format keeps of a file
//! ([`FileFacts`]), which each record holds.
//!
//! The index is persistent: the next version's is this one's with the files
//! of the manifests that left the list removed and those of the manifests
//! that joined it inserted, sharing every untouched record and tree node, so
//! a version costs what changed, and a query keeps the one it started on.
//! It records the manifests it was built from, those with a file of the
//! partition and those found to have none, so it tells whether it is a
//! version's own.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::ArrayRef;
use iceberg::expr::BoundPredicate;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, Literal, Manifest, ManifestEntry,
    ManifestFile, ManifestStatus, PrimitiveLiteral, PrimitiveType, Schema, SchemaRef, Struct,
    TableMetadata,
};
use imbl::HashMap as SharedMap;
use object_storage::{FileRef, ObjectPath};
use parquet_engine::FileStats;
use parquet_engine::file_entry::FileEntry;
use parquet_engine::{measure_allocation, measure_shared_allocation};
use planner::catalog::Column;
use planner::expression::Expression;
use planner::types::{Type, physical_arrow_type};
use table_file_index::{CacheableIndex, FileFacts, IndexedRecord, PartitionFiles, VisitFacts};

use crate::files::data_files::{DataFileEntry, literal_to_scalar, read_partition_values};
use crate::files::partitions_from_filter::enumerate_tuples_from_filter;
use crate::files::pruning::{self, FieldRange, PartitionPruner, StoredColumn};
use crate::metadata::layout::TableLocation;
use crate::{Error, Result};

/// One partition of a table: the files written under one partition spec
/// with one tuple.
#[derive(Debug)]
pub struct Partition {
    spec_id: i32,
    tuple: Struct,
}

impl Partition {
    /// Encode the partition as its key: bytes equal for exactly the files
    /// of it, the spec's id, then the tuple, each value in a form whose
    /// bytes order as the values do. The key is built anew by every call,
    /// so whatever needs it more than once keeps what one call returned.
    pub fn encode_key(&self) -> Arc<[u8]> {
        encode_partition(self.spec_id, &self.tuple)
    }

    /// The partition spec the tuple is laid out by.
    pub fn spec_id(&self) -> i32 {
        self.spec_id
    }

    pub fn tuple(&self) -> &Struct {
        &self.tuple
    }

    /// The partition of `tuple` under the spec `spec_id`.
    pub(crate) fn new(spec_id: i32, tuple: Struct) -> Self {
        Self { spec_id, tuple }
    }

    /// An estimate of the memory the partition holds, itself included, as
    /// one shared allocation: its tuple.
    fn estimate_heap_bytes(&self) -> usize {
        let tuple = self.tuple.fields();
        measure_shared_allocation(size_of::<Self>())
            + measure_allocation(size_of_val(tuple))
            + tuple
                .iter()
                .map(|value| match value {
                    Some(Literal::Primitive(value)) => estimate_literal_heap_bytes(value),
                    _ => 0,
                })
                .sum::<usize>()
    }

    /// The partition `key` names in the table `metadata` describes: the
    /// spec whose id the key starts with, and the tuple the rest of the key
    /// holds, each value read back as the type the spec gives its field. A
    /// key that is none of the table's is refused.
    pub(crate) fn from_key(key: &[u8], metadata: &TableMetadata) -> Result<Self> {
        let unknown = || Error::UnknownPartition {
            table: metadata.location().to_string(),
        };
        let (spec_bytes, mut rest) = key.split_first_chunk::<4>().ok_or_else(unknown)?;
        let spec_id = (u32::from_be_bytes(*spec_bytes) ^ (1 << 31)) as i32;
        let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(unknown)?;
        let mut values = Vec::with_capacity(spec.fields().len());
        for field in spec.fields() {
            // A tuple shorter than its spec ends the key early.
            let Some((is_set, value_bytes)) = rest.split_first() else {
                break;
            };
            rest = value_bytes;
            if *is_set == 0 {
                values.push(None);
                continue;
            }
            let value_type = pruning::find_partition_value_type(metadata.current_schema(), field)
                .ok_or_else(unknown)?;
            let value = decode_value(&mut rest, &value_type).ok_or_else(unknown)?;
            values.push(Some(Literal::Primitive(value)));
        }
        if !rest.is_empty() {
            return Err(unknown());
        }
        Ok(Self {
            spec_id,
            tuple: Struct::from_iter(values),
        })
    }
}

/// A decoded data manifest: the partitions it holds a live file of, and the
/// live files of those the decode was asked for.
pub(crate) struct DecodedManifest {
    /// Each partition the manifest holds a live file of, under its key, in
    /// key order.
    pub partitions: Vec<(Arc<[u8]>, Arc<Partition>)>,
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
}

/// Encodes a bound of a table's sort key in a file, read from the file's
/// manifest entry, as bytes that order as the bounds do.
pub(crate) struct SortKeyEncoder {
    /// The sort columns, each by field id and Pivot type; none for a table
    /// without a sort key, or one sorted by a column whose type has no
    /// ordered byte form, whose files then order by commit alone.
    columns: Vec<(i32, Type)>,
    /// Over the Arrow types of `columns`.
    encoder: table_file_index::SortKeyEncoder,
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
        let types = sort_columns
            .iter()
            .map(|(_, pivot_type)| physical_arrow_type(pivot_type))
            .collect();
        let encoder = table_file_index::SortKeyEncoder::new(types).map_err(Error::arrow)?;
        let columns = if encoder.has_ordered_form() {
            sort_columns
        } else {
            Vec::new()
        };
        Ok(Self { columns, encoder })
    }

    /// The field the sort key leads with, when the key has an ordered byte
    /// form.
    fn find_leading_field(&self) -> Option<i32> {
        self.columns.first().map(|(field_id, _)| *field_id)
    }

    /// The key bytes of a file whose manifest entry bounds the field
    /// `field_id`, on one side, by `bound(field_id)`. A bound that does not
    /// read as its column's type orders as a missing one.
    fn encode<'a>(&self, bound: impl Fn(i32) -> Option<&'a Datum>) -> Result<Arc<[u8]>> {
        self.encoder
            .encode_bound(|position| {
                let (field_id, pivot_type) = &self.columns[position];
                let arrow_type = physical_arrow_type(pivot_type);
                let array: ArrayRef = bound(*field_id)
                    .and_then(|bound| literal_to_scalar(bound.literal(), pivot_type))?
                    .into_inner();
                if array.data_type() == &arrow_type {
                    Some(array)
                } else {
                    arrow_cast::cast(array.as_ref(), &arrow_type).ok()
                }
            })
            .map_err(Error::arrow)
    }
}

/// One live data file of a table version.
pub struct IndexedFile {
    /// What every format knows of the file: its path under the table's data
    /// root and its size, its row count, the stretch of the sort key it
    /// covers, and the sequence number of its commit.
    pub facts: FileFacts,
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
    /// have no ids.
    pub first_row_id: Option<i64>,
    /// An estimate of the memory the record holds, itself included.
    heap_bytes: u64,
}

impl IndexedFile {
    /// The file `data` describes, of `partition`, recorded by a manifest
    /// written under the schema `schema_id`, and added by the commit with
    /// the data sequence number `sequence_number`.
    #[allow(clippy::too_many_arguments)] // what a manifest entry and its table say of one file
    fn new(
        data: DataFile,
        schema_id: i32,
        partition: Arc<Partition>,
        first_row_id: Option<i64>,
        sequence_number: i64,
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
        let facts = FileFacts::new(
            FileRef {
                path,
                size: data.file_size_in_bytes(),
            },
            Some(data.record_count()),
            sort_minimum,
            sort_maximum,
            // A table that numbers no commit reads every file at 0.
            u64::try_from(sequence_number).unwrap_or(0),
        );
        let mut file = Self {
            facts,
            data,
            partition,
            schema_id,
            first_row_id,
            heap_bytes: 0,
        };
        file.heap_bytes = file.compute_heap_bytes();
        Ok(file)
    }

    /// Decode the data manifest `file`, taken out of `manifest`: name the
    /// partitions it holds a live file of, and keep the live files of those
    /// `wants` accepts, asked once for each with its key, with everything
    /// the manifest leaves implicit filled in. A partition's key is encoded
    /// once for each run of entries of the partition.
    /// First row ids are assigned the way the spec inherits them: an entry
    /// without one gets the manifest's, advanced by the row counts of the
    /// entries assigned before it. A delete file among a data manifest's
    /// entries, or a file that is not Parquet, is refused.
    pub(crate) fn decode_manifest(
        file: &ManifestFile,
        manifest: Manifest,
        context: &DecodeContext<'_>,
        wants: &mut dyn FnMut(&[u8], &Partition) -> Result<bool>,
    ) -> Result<DecodedManifest> {
        let schema_id = manifest.metadata().schema_id();
        let partition_spec_id = manifest.metadata().partition_spec().spec_id();
        let mut next_row_id = file.first_row_id.map(|first| first as i64);
        // The partitions met, each with whether it is wanted.
        let mut partitions: BTreeMap<Arc<[u8]>, (Arc<Partition>, bool)> = BTreeMap::new();
        let mut files = FilesByPartition::new();
        // The partition of the entry before, with its key and whether it
        // is wanted: a manifest's entries mostly come in runs of one
        // partition.
        let mut latest: Option<(Arc<[u8]>, Arc<Partition>, bool)> = None;
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
            let is_of_latest = latest.as_ref().is_some_and(|(_, partition, _)| {
                partition.spec_id == partition_spec_id && &partition.tuple == data.partition()
            });
            if !is_of_latest {
                let key = encode_partition(partition_spec_id, data.partition());
                let (partition, is_wanted) = match partitions.get(&key) {
                    Some(met) => met.clone(),
                    None => {
                        let partition =
                            Arc::new(Partition::new(partition_spec_id, data.partition().clone()));
                        let is_wanted = wants(&key, &partition)?;
                        partitions.insert(Arc::clone(&key), (Arc::clone(&partition), is_wanted));
                        (partition, is_wanted)
                    }
                };
                latest = Some((key, partition, is_wanted));
            }
            let (key, partition, is_wanted) = latest
                .as_ref()
                .expect("the partition of the entry in hand is the latest");
            if !is_wanted {
                continue;
            }
            let sequence_number = read_sequence_number(&entry, file, context.metadata)?;
            // The manifest was parsed for this decode alone, so its entries
            // are its own to give away.
            let entry = Arc::unwrap_or_clone(entry);
            files
                .entry(Arc::clone(key))
                .or_default()
                .push(Arc::new(Self::new(
                    entry.data_file,
                    schema_id,
                    Arc::clone(partition),
                    first_row_id,
                    sequence_number,
                    context.encoder,
                    context.metadata,
                    context.location,
                )?));
        }
        Ok(DecodedManifest {
            partitions: partitions
                .into_iter()
                .map(|(key, (partition, _))| (key, partition))
                .collect(),
            files,
        })
    }

    /// An estimate of the memory the record holds, itself included: its
    /// facts, its paths, and every map and value of its data file, each at
    /// the size an allocation of it takes, with what the index's own maps
    /// spend on one file.
    fn compute_heap_bytes(&self) -> u64 {
        /// What a manifest's file list spends on one file.
        const MANIFEST_ENTRY_BYTES: usize = size_of::<Arc<IndexedFile>>();

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
        let tuple_bytes: usize = measure_allocation(size_of_val(tuple))
            + tuple
                .iter()
                .map(|value| match value {
                    Some(Literal::Primitive(value)) => estimate_literal_heap_bytes(value),
                    _ => 0,
                })
                .sum::<usize>();
        let total = measure_shared_allocation(size_of::<Self>())
            + MANIFEST_ENTRY_BYTES
            + measure_allocation(data.file_path().len())
            + count_bytes
            + bound_bytes
            + tuple_bytes
            + data
                .key_metadata()
                .map_or(0, |key| measure_allocation(key.len()))
            + data
                .split_offsets()
                .map_or(0, |offsets| measure_allocation(size_of_val(offsets)));
        total as u64 + self.facts.estimate_heap_bytes()
    }

    /// An estimate of the memory the record holds, itself included.
    pub fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
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

    /// The file as a commit records and removes it: its tuple under the
    /// partition spec it was written under, that spec's id, and the
    /// statistics of every declared column, as the datastore holds them.
    pub fn to_entry(&self, metadata: &TableMetadata, columns: &[Column]) -> Result<DataFileEntry> {
        Ok(DataFileEntry {
            file: self.facts.file().clone(),
            partition: read_partition_values(
                metadata,
                columns,
                &self.partition,
                &self.facts.file().path,
            )?,
            partition_spec_id: Some(self.partition.spec_id()),
            stats: Some(Arc::new(self.file_stats(metadata, columns.iter()))),
            sort_order_id: self.data.sort_order_id(),
            first_row_id: self.first_row_id,
        })
    }

    /// The file as the datastore's planners read it: its tuple under the
    /// partition spec it was written under, its row count, its bounds on
    /// `bounded_columns`, the columns files are laid out by, and
    /// `sorted_by`, the columns the caller read its rows to be in order of.
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
            file: self.facts.file().clone(),
            partition: read_partition_values(
                metadata,
                columns,
                &self.partition,
                &self.facts.file().path,
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

/// The sequence number of the commit that added the file of the live
/// `entry` of the manifest `manifest` to the table: its own, or the
/// manifest's for an ADDED entry, which leaves it implicit. A rewrite that
/// carries a file's data sequence number over still adds it at its own.
fn read_file_sequence_number(
    entry: &ManifestEntry,
    manifest: &ManifestFile,
    metadata: &TableMetadata,
) -> Result<i64> {
    match entry.file_sequence_number {
        Some(number) => Ok(number),
        None if entry.status() == ManifestStatus::Added => Ok(manifest.sequence_number),
        None if metadata.format_version() == iceberg::spec::FormatVersion::V1 => Ok(0),
        None => Err(Error::InvalidDataFile {
            file: entry.data_file().file_path().to_string(),
            message: "an existing entry carries no file sequence number".to_string(),
        }),
    }
}

/// The partitions in which `manifest`, the parsed data manifest `file`
/// names, holds a live file that a commit after the one numbered
/// `sequence_number` added, by key.
pub(crate) fn list_partitions_added_after(
    file: &ManifestFile,
    manifest: &Manifest,
    metadata: &TableMetadata,
    sequence_number: i64,
) -> Result<BTreeMap<Arc<[u8]>, Arc<Partition>>> {
    let spec_id = manifest.metadata().partition_spec().spec_id();
    let mut partitions = BTreeMap::new();
    for entry in manifest.entries() {
        if !entry.is_alive() || read_file_sequence_number(entry, file, metadata)? <= sequence_number
        {
            continue;
        }
        let tuple = entry.data_file().partition();
        let key = encode_partition(spec_id, tuple);
        partitions
            .entry(key)
            .or_insert_with(|| Arc::new(Partition::new(spec_id, tuple.clone())));
    }
    Ok(partitions)
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
    measure_allocation(buckets * (size_of::<(i32, Value)>() + 1) + TRAILING_CONTROL_BYTES)
}

/// What `literal` holds outside its own value: the bytes of a string or a
/// binary.
fn estimate_literal_heap_bytes(literal: &PrimitiveLiteral) -> usize {
    match literal {
        PrimitiveLiteral::String(value) => measure_allocation(value.len()),
        PrimitiveLiteral::Binary(value) => measure_allocation(value.len()),
        _ => 0,
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

/// Take a value of `value_type` off the front of `key`, as
/// [`encode_value`] wrote it; `None` when the bytes are not one.
fn decode_value(key: &mut &[u8], value_type: &PrimitiveType) -> Option<PrimitiveLiteral> {
    fn take<const N: usize>(key: &mut &[u8]) -> Option<[u8; N]> {
        let (taken, rest) = key.split_first_chunk::<N>()?;
        *key = rest;
        Some(*taken)
    }
    Some(match value_type {
        PrimitiveType::Boolean => PrimitiveLiteral::Boolean(take::<1>(key)?[0] != 0),
        PrimitiveType::Int | PrimitiveType::Date => {
            PrimitiveLiteral::Int((u32::from_be_bytes(take(key)?) ^ (1 << 31)) as i32)
        }
        PrimitiveType::Long
        | PrimitiveType::Time
        | PrimitiveType::Timestamp
        | PrimitiveType::Timestamptz
        | PrimitiveType::TimestampNs
        | PrimitiveType::TimestamptzNs => {
            PrimitiveLiteral::Long((u64::from_be_bytes(take(key)?) ^ (1 << 63)) as i64)
        }
        PrimitiveType::Float => {
            let bits = read_float_bits(take(key)?);
            PrimitiveLiteral::Float(f32::from_bits((bits >> 32) as u32).into())
        }
        PrimitiveType::Double => {
            PrimitiveLiteral::Double(f64::from_bits(read_float_bits(take(key)?)).into())
        }
        PrimitiveType::Decimal { .. } => {
            PrimitiveLiteral::Int128((u128::from_be_bytes(take(key)?) ^ (1 << 127)) as i128)
        }
        PrimitiveType::Uuid => PrimitiveLiteral::UInt128(u128::from_be_bytes(take(key)?)),
        PrimitiveType::String => {
            PrimitiveLiteral::String(String::from_utf8(take_escaped_bytes(key)?).ok()?)
        }
        PrimitiveType::Fixed(_) | PrimitiveType::Binary => {
            PrimitiveLiteral::Binary(take_escaped_bytes(key)?)
        }
    })
}

/// The bits of the float [`order_float_bits`] wrote as `ordered`.
fn read_float_bits(ordered: [u8; 8]) -> u64 {
    let ordered = u64::from_be_bytes(ordered);
    if ordered >> 63 == 1 {
        ordered & !(1 << 63)
    } else {
        !ordered
    }
}

/// Take the bytes [`extend_with_bytes`] wrote off the front of `key`, up to
/// and with their terminator; `None` when the key ends before one.
fn take_escaped_bytes(key: &mut &[u8]) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    loop {
        let (byte, rest) = key.split_first()?;
        *key = rest;
        if *byte != 0 {
            bytes.push(*byte);
            continue;
        }
        let (escape, rest) = key.split_first()?;
        *key = rest;
        match escape {
            0 => return Some(bytes),
            &u8::MAX => bytes.push(0),
            _ => return None,
        }
    }
}

impl IndexedRecord for IndexedFile {
    fn facts(&self) -> &FileFacts {
        &self.facts
    }

    fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
    }
}

/// The live data files of one partition of one table version, with the data
/// manifests they were looked for in. Cloning shares everything; an edit
/// copies only what it touches.
#[derive(Clone, Default)]
pub struct PartitionIndex {
    files: table_file_index::FileIndex<IndexedFile>,
    /// The partition's files in each data manifest the index was built
    /// from, by the manifest's path: none for a manifest whose
    /// manifest-list entry does not rule the partition out and that holds
    /// no file of it.
    manifests: SharedMap<Arc<str>, Arc<[Arc<IndexedFile>]>>,
    /// An estimate of the memory the record of the manifests holds.
    manifest_bytes: u64,
}

/// What the record of one manifest an index was built from takes besides
/// the manifest's path and the partition its files share: its place in the
/// map, and its list of files.
const MANIFEST_RECORD_BYTES: usize = 96;

impl PartitionIndex {
    /// The index of a partition at the version whose data manifests that
    /// may hold a file of it are `listed`, built on `previous`, the
    /// partition's index at another version: the files of the manifests
    /// not listed leave, and the partition's files in the manifests
    /// `previous` was not built from, which `take_files` hands over, join.
    /// A manifest is never rewritten in place, so one both versions list is
    /// not asked for.
    pub(crate) fn advance(
        previous: Option<&PartitionIndex>,
        listed: &[Arc<str>],
        mut take_files: impl FnMut(&Arc<str>) -> Vec<Arc<IndexedFile>>,
    ) -> PartitionIndex {
        let mut index = previous.cloned().unwrap_or_default();
        let gone: Vec<Arc<str>> = index
            .manifests
            .keys()
            .filter(|path| !listed.contains(path))
            .cloned()
            .collect();
        for path in gone {
            index.drop_manifest(&path);
        }
        for manifest in listed {
            if !index.manifests.contains_key(manifest) {
                index.add_manifest(manifest, take_files(manifest));
            }
        }
        index
    }

    /// Record that the index is built from the manifest at `path` too,
    /// which holds `files` of the partition.
    fn add_manifest(&mut self, path: &Arc<str>, files: Vec<Arc<IndexedFile>>) {
        for file in &files {
            self.insert(Arc::clone(file));
        }
        self.manifest_bytes += measure_manifest_record(path, &files);
        self.manifests.insert(Arc::clone(path), Arc::from(files));
    }

    /// Take the files of the manifest at `path` out of the index, which is
    /// then built from one manifest fewer.
    fn drop_manifest(&mut self, path: &str) {
        let Some(files) = self.manifests.remove(path) else {
            return;
        };
        self.manifest_bytes -= measure_manifest_record(path, &files);
        for file in files.iter() {
            self.remove(&file.facts.file().path);
        }
    }

    /// Whether the index was built from exactly the manifests `listed`.
    pub(crate) fn is_built_from<'m>(
        &self,
        listed: impl ExactSizeIterator<Item = &'m Arc<str>>,
    ) -> bool {
        self.manifests.len() == listed.len()
            && listed
                .into_iter()
                .all(|manifest| self.manifests.contains_key(manifest))
    }

    /// Whether the index was built from the manifest at `manifest_path`.
    pub(crate) fn holds_manifest(&self, manifest_path: &str) -> bool {
        self.manifests.contains_key(manifest_path)
    }

    /// Add `file`, replacing a file already at its path.
    pub fn insert(&mut self, file: Arc<IndexedFile>) {
        self.files.insert(file);
    }

    /// Drop the file at `path`, if the index has one.
    pub fn remove(&mut self, path: &ObjectPath) -> Option<Arc<IndexedFile>> {
        self.files.remove(path)
    }

    /// The file at `path`.
    pub fn find(&self, path: &ObjectPath) -> Option<&Arc<IndexedFile>> {
        self.files.find(path)
    }

    /// Every file, in key order.
    pub fn files(&self) -> impl Iterator<Item = &Arc<IndexedFile>> {
        self.files.files()
    }

    /// The files as the index every format keeps a partition's files in.
    pub fn file_index(&self) -> &table_file_index::FileIndex<IndexedFile> {
        &self.files
    }

    /// An estimate of the memory the index holds: the sum of its files',
    /// and its record of the manifests it was built from.
    pub fn estimate_heap_bytes(&self) -> u64 {
        self.files.estimate_heap_bytes() + self.manifest_bytes
    }
}

/// What an index's record of the manifest at `path`, which holds `files`
/// of the index's partition, counts for: the record, the path, and the
/// partition as the manifest's decode named it, which its files share.
fn measure_manifest_record(path: &str, files: &[Arc<IndexedFile>]) -> u64 {
    let partition_bytes = files
        .first()
        .map_or(0, |file| file.partition.estimate_heap_bytes());
    (MANIFEST_RECORD_BYTES + measure_shared_allocation(path.len()) + partition_bytes) as u64
}

impl PartitionFiles for PartitionIndex {
    fn visit_files(&self, visit: &mut VisitFacts<'_>) -> std::ops::ControlFlow<()> {
        self.files.visit_files(visit)
    }

    fn find_file(&self, path: &ObjectPath) -> Option<&FileFacts> {
        self.files.find_file(path)
    }

    fn count_files(&self) -> usize {
        self.files.count_files()
    }
}

impl CacheableIndex for PartitionIndex {
    fn estimate_heap_bytes(&self) -> u64 {
        PartitionIndex::estimate_heap_bytes(self)
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
        let row_filters = pruning::bind_filters(metadata.current_schema(), filters);
        Self::from_row_filters(metadata, row_filters)
    }

    /// What `row_filters`, each over the table's columns, require of the
    /// files of the table `metadata` describes: a row passes when all are
    /// true.
    pub(crate) fn from_row_filters(
        metadata: &TableMetadata,
        row_filters: Vec<BoundPredicate>,
    ) -> iceberg::Result<Self> {
        let schema = metadata.current_schema();
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

    /// The partitions of the specs `spec_ids` a row satisfying every filter
    /// may sit in, computed from the filters alone when they leave every
    /// field of each of those specs only a few values: every combination
    /// of the values, whether or not the table has a file in it. `None`
    /// when a spec is not one `metadata` lists, the filters leave a field
    /// of one unrestricted, or the partitions number more than `limit`.
    pub(crate) fn enumerate_partitions_from_filter(
        &self,
        metadata: &TableMetadata,
        spec_ids: &[i32],
        limit: usize,
    ) -> Option<Vec<Partition>> {
        let mut partitions = Vec::new();
        for spec_id in spec_ids {
            let pruner = find_by_id(&self.partition_pruners, *spec_id)?;
            let partition_spec = metadata.partition_spec_by_id(*spec_id)?;
            let tuples = enumerate_tuples_from_filter(
                metadata,
                partition_spec,
                pruner.partition_filter(),
                limit - partitions.len(),
            )?;
            partitions.extend(
                tuples
                    .into_iter()
                    .map(|tuple| Partition::new(*spec_id, tuple)),
            );
        }
        Some(partitions)
    }

    /// The files of `index`, a partition whose tuple the filters keep, that
    /// are not empty and whose column bounds may hold a row satisfying
    /// every filter, in key order.
    pub(crate) fn select<'s>(
        &'s self,
        index: &'s PartitionIndex,
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
