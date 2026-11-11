//! A table version's live data files, held once: every file as one record,
//! ordered by partition, then by where the file starts on the table's sort
//! key, then by path. The path only makes the order total: files of one
//! partition that start at the same place on the sort key, which is every
//! file of an unpartitioned table without a sort key, come out in an order
//! that is fixed for the version and means nothing.
//!
//! The index is what a version keeps of its manifests. A manifest is decoded
//! and each of its live entries moves into a record: the entry's data file
//! as the spec crate decoded it, with its partition tuple and the bounds
//! and null counts of every column the manifest recorded something for, so
//! a record grows with the table's width. Planning selects files from the
//! records, the columns' nullability is counted from them, and a commit
//! rebuilds the entry of a file it removes from one.
//!
//! Maintenance walks the same records: a file's partition and the stretch of
//! the sort key it covers are kept as bytes that order as the values do, so
//! files are grouped and their ranges compared without decoding a bound.
//!
//! The index is persistent: the next version's is this one's with the files
//! of the manifests that left the list removed and those of the manifests
//! that joined it inserted, sharing every untouched record and tree node, so
//! a version costs what changed, and a query keeps the one it started on.

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_row::{RowConverter, SortField};
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

use crate::data_files::{DataFileEntry, ManifestOrigin, literal_to_scalar, read_partition_values};
use crate::layout::TableLocation;
use crate::pruning::{self, PartitionPruner, StoredColumn};
use crate::{Error, Result};

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
pub struct SortKeyEncoder {
    /// The sort columns, each by field id and Pivot type, with the converter
    /// over their Arrow types; `None` for a table without a sort key, or one
    /// sorted by a column whose type has no ordered byte form, whose files
    /// then order by commit alone.
    columns: Option<(Vec<(i32, Type)>, RowConverter)>,
}

impl SortKeyEncoder {
    /// The encoder for a table sorted by `sort_by`, columns among `columns`
    /// whose field ids `metadata`'s current schema gives.
    pub fn new(metadata: &TableMetadata, columns: &[Column], sort_by: &[String]) -> Result<Self> {
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
    /// The partition spec the file's manifest was written under, which
    /// its tuple is laid out by.
    pub partition_spec_id: i32,
    /// The schema the file's manifest was written under: the types its
    /// bounds are recorded in, and the fields the file has.
    pub schema_id: i32,
    /// Where the table's metadata records the file; `None` for a file this
    /// process swapped in that no manifest of the version records yet.
    pub origin: Option<ManifestOrigin>,
    key: FileKey,
    /// The sort key's upper bound across the file, in the byte form of the
    /// key's lower bound; `None` when the table has no sort key with an
    /// ordered byte form, or the entry does not bound the column the key
    /// leads with on both sides.
    sort_maximum: Option<Arc<[u8]>>,
}

impl IndexedFile {
    /// The file `data` describes, recorded by a manifest written under the
    /// schema `schema_id` and the spec `partition_spec_id`.
    fn new(
        data: DataFile,
        schema_id: i32,
        partition_spec_id: i32,
        origin: Option<ManifestOrigin>,
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
            partition: encode_partition(partition_spec_id, data.partition()),
            sort_minimum,
            path: Arc::from(path.as_str()),
        };
        Ok(Self {
            file: FileRef {
                path,
                size: data.file_size_in_bytes(),
            },
            data,
            partition_spec_id,
            schema_id,
            origin,
            key,
            sort_maximum,
        })
    }

    /// The live files of the data manifest `file`, taken out of `manifest`, with
    /// everything the manifest leaves implicit filled in. First row ids are
    /// assigned the way the spec inherits them: an entry without one gets the
    /// manifest's, advanced by the row counts of the entries assigned before it.
    /// A delete file among a data manifest's entries, or a file that is not
    /// Parquet, is refused.
    pub(crate) fn collect_live_files(
        table: &str,
        file: &ManifestFile,
        manifest: Manifest,
        encoder: &SortKeyEncoder,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<Vec<Arc<Self>>> {
        let manifest_path: Arc<str> = Arc::from(file.manifest_path.as_str());
        let schema_id = manifest.metadata().schema_id();
        let partition_spec_id = manifest.metadata().partition_spec().spec_id();
        let mut next_row_id = file.first_row_id.map(|first| first as i64);
        let mut files = Vec::new();
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
                    table: table.to_string(),
                    file: data.file_path().to_string(),
                    format: data.file_format().to_string(),
                });
            }
            // The manifest was parsed for this index alone, so its entries
            // are its own to give away.
            let entry = Arc::unwrap_or_clone(entry);
            files.push(Arc::new(Self::from_manifest_entry(
                entry,
                file,
                &manifest_path,
                schema_id,
                partition_spec_id,
                first_row_id,
                encoder,
                metadata,
                location,
            )?));
        }
        Ok(files)
    }

    /// The live `entry` of the manifest `manifest`, written under the schema
    /// `schema_id` and the spec `partition_spec_id`. Inherits what a manifest leaves implicit on its
    /// ADDED entries: the snapshot that added the file, its sequence numbers,
    /// and (`first_row_id` being the caller's running assignment) its first
    /// row id.
    #[allow(clippy::too_many_arguments)] // the pieces of one manifest entry
    pub fn from_manifest_entry(
        entry: ManifestEntry,
        manifest: &ManifestFile,
        manifest_path: &Arc<str>,
        schema_id: i32,
        partition_spec_id: i32,
        first_row_id: Option<i64>,
        encoder: &SortKeyEncoder,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<Self> {
        let added = entry.status() == ManifestStatus::Added;
        let snapshot_id = entry.snapshot_id().unwrap_or(manifest.added_snapshot_id);
        let sequence_number = match entry.sequence_number() {
            Some(number) => number,
            None if added => manifest.sequence_number,
            // A version 1 table has no sequence numbers: every entry reads as
            // the spec's inherited 0.
            None if metadata.format_version() == iceberg::spec::FormatVersion::V1 => 0,
            None => {
                return Err(Error::InvalidDataFile {
                    file: entry.data_file().file_path().to_string(),
                    message: "an existing entry carries no sequence number".to_string(),
                });
            }
        };
        let file_sequence_number = match entry.file_sequence_number {
            Some(number) => number,
            None if added => manifest.sequence_number,
            None => sequence_number,
        };
        Self::new(
            entry.data_file,
            schema_id,
            partition_spec_id,
            Some(ManifestOrigin {
                snapshot_id,
                sequence_number,
                file_sequence_number,
                first_row_id,
                manifest_path: Arc::clone(manifest_path),
            }),
            encoder,
            metadata,
            location,
        )
    }

    /// `entry`, a file this process committed that no manifest of the version
    /// records yet: it orders among the committed files as they do, by its
    /// partition, where it starts on the sort key, and its path.
    pub fn uncommitted(
        entry: &DataFileEntry,
        encoder: &SortKeyEncoder,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<Self> {
        Self::new(
            entry.create_data_file(metadata, location)?,
            metadata.current_schema().schema_id(),
            metadata.default_partition_spec().spec_id(),
            None,
            encoder,
            metadata,
            location,
        )
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

    /// The bound the entry recorded for `field_id`, below when `lower` and
    /// above otherwise, in the type the file's manifest was written with.
    pub fn bound(&self, metadata: &TableMetadata, field_id: i32, lower: bool) -> Option<&Datum> {
        find_bound(
            &self.data,
            metadata.schema_by_id(self.schema_id),
            field_id,
            lower,
        )
    }

    /// Whether the file was written with the field `field_id`. A file whose
    /// schema the table no longer lists is taken to have every field.
    pub fn has_field(&self, metadata: &TableMetadata, field_id: i32) -> bool {
        metadata
            .schema_by_id(self.schema_id)
            .is_none_or(|schema| schema.field_by_id(field_id).is_some())
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
            origin: self.origin.clone(),
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
            PrimitiveLiteral::String(value) => extend_with_bytes(&mut key, value.as_bytes()),
            PrimitiveLiteral::Binary(value) => extend_with_bytes(&mut key, value),
            PrimitiveLiteral::AboveMax => key.push(u8::MAX),
            PrimitiveLiteral::BelowMin => key.push(0),
        }
    }
    Arc::from(key)
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

/// The live data files of one table version. Cloning shares everything; an
/// edit copies only what it touches.
#[derive(Clone, Default)]
pub struct FileIndex {
    files: OrdMap<FileKey, Arc<IndexedFile>>,
    by_path: SharedMap<Arc<str>, Arc<IndexedFile>>,
    /// Per field id, how many files' entries record that the field holds no
    /// NULL in the file.
    null_free_files: SharedMap<i32, u64>,
    total_rows: u64,
    /// The files each decoded data manifest lists, by the manifest's path.
    manifests: SharedMap<Arc<str>, Arc<[Arc<IndexedFile>]>>,
}

impl FileIndex {
    /// The index of the version whose data manifests are `listed`, built on
    /// `previous`, an earlier version's: the files of the manifests no longer
    /// listed leave, and those of the manifests `previous` did not hold are
    /// decoded by `decode`, a few manifests at a time, and join. A manifest
    /// is never rewritten in place, so one both versions list is not read
    /// again.
    pub fn advance(
        previous: Option<&FileIndex>,
        listed: &[&ManifestFile],
        mut decode: impl FnMut(&[&ManifestFile]) -> Result<Vec<Vec<Arc<IndexedFile>>>>,
    ) -> Result<FileIndex> {
        /// How many manifests are decoded, and held decoded, at a time.
        const MANIFESTS_PER_READ: usize = 16;

        let mut index = previous.cloned().unwrap_or_default();
        let gone: Vec<Arc<str>> = index
            .manifests
            .keys()
            .filter(|path| {
                !listed
                    .iter()
                    .any(|manifest| manifest.manifest_path == path.as_ref())
            })
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
        let unread: Vec<&ManifestFile> = listed
            .iter()
            .filter(|manifest| {
                !index
                    .manifests
                    .contains_key(manifest.manifest_path.as_str())
            })
            .copied()
            .collect();
        for manifests in unread.chunks(MANIFESTS_PER_READ) {
            for (manifest, files) in manifests.iter().zip(decode(manifests)?) {
                for file in &files {
                    index.insert(Arc::clone(file));
                }
                index
                    .manifests
                    .insert(Arc::from(manifest.manifest_path.as_str()), Arc::from(files));
            }
        }
        Ok(index)
    }

    /// Add `file`, replacing a file already at its path.
    pub fn insert(&mut self, file: Arc<IndexedFile>) {
        self.remove(&file.file.path);
        for (field_id, null_count) in file.data.null_value_counts() {
            if *null_count == 0 {
                *self.null_free_files.entry(*field_id).or_default() += 1;
            }
        }
        self.total_rows += file.data.record_count();
        self.by_path
            .insert(Arc::clone(&file.key.path), Arc::clone(&file));
        self.files.insert(file.key.clone(), file);
    }

    /// Drop the file at `path`, if the index has one.
    pub fn remove(&mut self, path: &ObjectPath) -> Option<Arc<IndexedFile>> {
        let file = self.by_path.remove(path.as_str())?;
        self.files.remove(&file.key);
        for (field_id, null_count) in file.data.null_value_counts() {
            if *null_count == 0
                && let Some(count) = self.null_free_files.get_mut(field_id)
            {
                *count -= 1;
            }
        }
        self.total_rows -= file.data.record_count();
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

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The rows of every file, deleted ones included.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// The files that are not empty and may hold a row satisfying every one
    /// of `filters`, in key order, for the table `metadata` describes, whose
    /// version this index is of: a file is described by its column bounds,
    /// and by the partition values of the spec it was written under when the
    /// table lists that spec.
    pub fn select(
        &self,
        metadata: &TableMetadata,
        filters: &[Expression],
    ) -> iceberg::Result<Vec<&Arc<IndexedFile>>> {
        let schema = metadata.current_schema();
        let row_filters = pruning::bind_filters(schema, filters);
        let compared_columns = pruning::list_compared_fields(&row_filters);
        // What a file's partition spec and schema decide is resolved once
        // for each of them, not once per file.
        let mut partition_pruners = Vec::new();
        for partition_spec in metadata.partition_specs_iter() {
            let pruner = PartitionPruner::new(schema, partition_spec, &row_filters)?;
            partition_pruners.push((partition_spec.spec_id(), pruner));
        }
        partition_pruners.sort_unstable_by_key(|(spec_id, _)| *spec_id);
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

        let mut selected = Vec::new();
        let mut partition_values = Vec::new();
        let mut column_ranges = Vec::with_capacity(compared_columns.len());
        for file in self.files() {
            if file.data.record_count() == 0 {
                continue;
            }
            if let Some(pruner) = find_by_id(&partition_pruners, file.partition_spec_id)
                && !pruner.may_match(file.data.partition(), &mut partition_values)?
            {
                continue;
            }
            let columns = match find_by_id(&columns_per_schema, file.schema_id) {
                Some(columns) => columns,
                None => &columns_of_unlisted_schema,
            };
            column_ranges.clear();
            column_ranges.extend(columns.iter().map(|column| column.read_range(&file.data)));
            if row_filters
                .iter()
                .all(|row_filter| pruning::may_match(row_filter, &column_ranges))
            {
                selected.push(file);
            }
        }
        Ok(selected)
    }

    /// Whether some file may hold a NULL in the field `field_id`: one whose
    /// entry records a null in it, or records no count for it. A table
    /// without files may hold anything.
    pub fn may_hold_nulls(&self, field_id: i32) -> bool {
        self.is_empty()
            || self.null_free_files.get(&field_id).copied().unwrap_or(0) < self.len() as u64
    }

    /// The files the decoded manifest at `manifest_path` lists.
    pub fn manifest_files(&self, manifest_path: &str) -> Option<&[Arc<IndexedFile>]> {
        self.manifests
            .get(manifest_path)
            .map(|files| files.as_ref())
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
