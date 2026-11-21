//! One Iceberg table at one version: what every datastore reads a table
//! through and writes it through, given only the table's metadata
//! store.
//!
//! A table is loaded from its metadata document and its current snapshot's
//! manifest list, and that is all a version holds of its files: opening a
//! table decodes no manifest. What is read off the manifests of every table
//! lives in one cache ([`FileMetadataCache`]), which keeps it within a
//! budget of memory, and is read again once the cache let go of it:
//!
//! - A scan selects its files through the parts each data manifest is cut
//!   into, by partition and then by column
//!   ([`manifest_parts`](crate::files::manifest_parts)), of which it reads
//!   the columns its filters compare (`selection`).
//! - Maintenance takes a version's files a manifest at a time, through the
//!   same parts, each with its bounds on the sort key (`file_groups`), and
//!   keeps them itself between its rounds.
//! - A commit, a merge and a description find the files of the partitions
//!   they ask for through the same parts (`live_files`), and keep nothing.
//!
//! The same cache, within the same budget, keeps the footers a table's
//! files were found to have.
//!
//! # Which manifests hold a partition
//!
//! Nothing is kept of a manifest once it was decoded but what went into the
//! cache. Which manifests a partition's files are in is read off the
//! manifest list every time: an entry bounds each partition field's values
//! across its manifest, so it rules a partition in or out and says nothing
//! more.
//!
//! Which partitions a version has is not kept either. An entry that leaves
//! its manifest one partition, of a spec without fields or with every
//! field bounded to a single value, names it. Those any other manifest
//! holds are read off the part of the manifest that lists them, which the
//! cache holds; a scan asks only the manifests its filters cannot rule out
//! by their entries.
//!
//! # Footers
//!
//! A version holds no file's footer either. Whatever needs row groups, a
//! scan's selection, a merge, a layout measurement, a description, asks for
//! those of the files it works on. A footer is read and parsed once, over
//! the worker pool, and the row groups that read produced are then kept in
//! the cache, each with the open of its file, where every version that
//! reads the file as this one does finds them and takes them as they are,
//! with nothing opened: one the cache let go of is read again. Row groups
//! are shared between the cache and everything that took them, so nothing
//! changes one: a selection records its version's deleted rows on copies.
//!
//! # Which entries a version finds
//!
//! A version carries nothing of the versions before it. It finds the
//! footers of its table's files in the cache under a key computed from
//! itself: where the table is, which table it is, and what a file of it is
//! read under, its schema and the Pivot types its fields are read as. Every
//! version that computes an equal key finds the same entries, whether it
//! was derived from another by a refresh or a commit, or loaded on its own;
//! a version under another schema finds none of them. The
//! parts of a table's manifests are read off the manifests alone, so they
//! are found under a key of their own that every version of the table
//! computes alike.
//! Nothing leaves the cache because a table object went away: an entry
//! stays until the budget lets go of it, or until whoever deletes its file
//! says so ([`Table::forget_footers`], [`forget_cached_table`]).

use std::sync::Arc;

use crate::file_metadata_cache::{FileMetadataCache, FooterUse, TableKey};
use arrow_schema::{Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{Manifest, ManifestFile, TableMetadata, TableMetadataRef};
use object_storage::{DataFile, FileRef, ObjectPath, load_objects};
use parquet_engine::{
    FileRowGroups, ParquetTable, ScanSelection, TableColumns, load_file_row_groups,
};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::expression::Expression;
use planner::types::physical_arrow_type;
use uuid::Uuid;

use crate::files::deletes::{count_deleted_rows, read_deletion_vectors};
use crate::files::manifests::ManifestList;
use crate::files::partition::Partition;
use crate::files::sort_bounded_file::SortColumns;
use crate::metadata::layout::TableLocation;
use crate::metadata::metadata_store::MetadataFile;
use crate::metadata::metadata_store::MetadataStore;
use crate::types::columns::{LOGICAL_TYPES_PROPERTY, identity_columns, read_table_columns};
use crate::types::partitioning::{PartitionKeys, format_partition_fields};
use crate::types::schema::{UNSORTED_ORDER_ID, find_writer_sort_order_id};
use crate::{Error, Result};

mod file_groups;
mod live_files;
pub(crate) mod manifest_reader;
mod selection;

pub use file_groups::ManifestPartitionFiles;
pub(crate) use live_files::FoundFile;
pub(crate) use manifest_reader::ManifestReader;
pub(crate) use selection::FileSelector;

/// How many manifests are read, and held parsed, at a time.
pub(crate) const MANIFESTS_PER_READ: usize = 16;

/// An Iceberg table at one committed version, with what reading and writing
/// it needs: its metadata store, where its files live and the
/// store they are read through, and the pool its reads run on. Immutable as
/// a version: a refresh or a commit hands back the next version as a new
/// value.
pub struct Table {
    /// The name the table is known by in its datastore.
    name: SchemaQualifiedTableName,
    metadata_store: Arc<dyn MetadataStore>,
    location: TableLocation,
    dispatcher: DataFlowDispatcher,
    metadata_file: Arc<MetadataFile>,
    /// The declared columns, keyed by the field ids the files carry.
    columns: TableColumns,
    /// The fields of the default partition spec, as the terms `CREATE
    /// TABLE` spells them.
    partition_by: Vec<String>,
    sort_by: Vec<String>,
    /// The id of the default sort order when rows ordered by `sort_by`, each
    /// column ascending with NULLs first, are in that order; `None` for an
    /// unsorted table and for an order Pivot's writer does not produce.
    writer_sort_order_id: Option<i32>,
    manifest_list: ManifestList,
    /// The length of the current snapshot's manifest list, when the version
    /// was told it or read the list; `None` for a version with no snapshot.
    manifest_list_size: Option<u64>,
    /// The key the footers of this version's files are cached under
    /// ([`build_cache_key`]).
    cache_key: TableKey,
    /// The key the parts of this version's data manifests are cached under
    /// ([`build_manifest_cache_key`]).
    manifest_cache_key: TableKey,
}

/// The key the version `metadata` describes of the table at `location`
/// finds the cached footers of its files under: equal for the versions of
/// one table that read a file alike, however each came to be. The table is
/// named by its URI. The rest tells which table that is, by its UUID, and
/// what a footer was read under: the id of the schema, which a footer is
/// parsed against, and the [`LOGICAL_TYPES_PROPERTY`], the Pivot types the
/// schema's fields are read as. The id is written whole and the property
/// last, quoted, so two versions that differ in either never write the
/// same key.
fn build_cache_key(location: &TableLocation, metadata: &TableMetadata) -> TableKey {
    let logical_types = metadata.properties().get(LOGICAL_TYPES_PROPERTY);
    TableKey {
        table: Arc::from(location.table_uri()),
        reading: Arc::from(format!(
            "{} {} {logical_types:?}",
            metadata.uuid(),
            metadata.current_schema_id(),
        )),
    }
}

/// The key the version `metadata` describes of the table at `location`
/// finds the parts of its data manifests under
/// ([`manifest_parts`](crate::files::manifest_parts)). A part is read off
/// its manifest and nothing else of the table but what a file's path is
/// taken against and how its manifests number their commits, so every
/// version of the table finds the parts of the manifests it shares with
/// another, whatever schema, partition spec or sort order it is read under:
/// the key names the table by its URI and its UUID, and the format version
/// its commits are numbered under.
fn build_manifest_cache_key(location: &TableLocation, metadata: &TableMetadata) -> TableKey {
    TableKey {
        table: Arc::from(location.table_uri()),
        reading: Arc::from(format!(
            "{} manifests of format version {}",
            metadata.uuid(),
            metadata.format_version(),
        )),
    }
}

/// Let go of everything the process's cache holds of the table at
/// `location`, whichever schema its versions read it under: what whoever deletes the table's directory
/// calls first, since a footer held holds its file open, and a deleted
/// local file keeps its blocks for as long as it is open.
pub fn forget_cached_table(location: &TableLocation) {
    FileMetadataCache::process_wide().remove_tables(&location.table_uri());
}

/// What a scan over some filters selected of a table version.
pub(crate) struct SelectedFiles {
    /// The files, each located for reading, in the order they are scanned.
    pub(crate) files: Vec<DataFile>,
    /// The names of the parts of the table's manifests the files were
    /// selected through.
    read_parts: Vec<Arc<[u8]>>,
    /// Whether each declared column may hold a NULL in one of the files
    /// (see [`ScanSelection::nullability`]).
    pub(crate) nullability: Vec<bool>,
}

impl Table {
    /// Load the table `name` at the version `metadata_store` holds current:
    /// a cold open.
    pub fn load(
        dispatcher: &DataFlowDispatcher,
        metadata_store: Arc<dyn MetadataStore>,
        location: TableLocation,
        name: &SchemaQualifiedTableName,
    ) -> Result<Table> {
        let metadata_file = metadata_store.load_latest(name)?;
        Self::at(
            dispatcher,
            metadata_store,
            location,
            name,
            metadata_file,
            None,
        )
    }

    /// The table at the version `metadata_file` holds. The manifest list
    /// is read over the pool when `manifest_list_size` is known, else
    /// fetched whole. No manifest is read.
    pub fn at(
        dispatcher: &DataFlowDispatcher,
        metadata_store: Arc<dyn MetadataStore>,
        location: TableLocation,
        name: &SchemaQualifiedTableName,
        metadata_file: MetadataFile,
        manifest_list_size: Option<u64>,
    ) -> Result<Table> {
        let metadata = &metadata_file.metadata;
        let (manifest_list, manifest_list_size) = match metadata.current_snapshot() {
            None => (ManifestList::empty(metadata), None),
            Some(snapshot) => {
                let path = snapshot.manifest_list();
                let bytes = match manifest_list_size {
                    Some(size) => {
                        load_objects(dispatcher, &[location.data_file(path, size)?])?
                            .pop()
                            .expect("load_objects returns one object per file")
                            .bytes
                    }
                    None => location.read_metadata_object(path)?,
                };
                (
                    ManifestList::parse(&name.to_string(), metadata, &bytes)?,
                    Some(bytes.len() as u64),
                )
            }
        };
        Self::from_manifest_list(
            dispatcher,
            metadata_store,
            location,
            name,
            Arc::new(metadata_file),
            manifest_list,
            manifest_list_size,
        )
    }

    /// The table at `metadata_file` with its manifest list already in hand:
    /// it finds the manifest parts and the footers the other versions of
    /// the table left in the cache.
    pub(crate) fn from_manifest_list(
        dispatcher: &DataFlowDispatcher,
        metadata_store: Arc<dyn MetadataStore>,
        location: TableLocation,
        name: &SchemaQualifiedTableName,
        metadata_file: Arc<MetadataFile>,
        manifest_list: ManifestList,
        manifest_list_size: Option<u64>,
    ) -> Result<Table> {
        let metadata = &metadata_file.metadata;
        if metadata.encryption_keys_iter().len() > 0 {
            return Err(Error::EncryptedTable {
                table: name.to_string(),
            });
        }
        let schema = metadata.current_schema();
        let columns = read_table_columns(&name.to_string(), metadata)?;
        let partition_by = format_partition_fields(schema, metadata.default_partition_spec());
        let sort_by = identity_columns(
            schema,
            metadata
                .default_sort_order()
                .fields
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );
        let writer_sort_order_id = find_writer_sort_order_id(metadata, columns.columns());
        let cache_key = build_cache_key(&location, metadata);
        let manifest_cache_key = build_manifest_cache_key(&location, metadata);
        Ok(Table {
            name: name.clone(),
            metadata_store,
            location,
            dispatcher: dispatcher.clone(),
            metadata_file,
            columns,
            partition_by,
            sort_by,
            writer_sort_order_id,
            manifest_list,
            manifest_list_size,
            cache_key,
            manifest_cache_key,
        })
    }

    /// The latest committed version, derived from this one, or `None` when
    /// this one is still current.
    pub fn refresh(&self) -> Result<Option<Table>> {
        let latest = self.metadata_store.load_latest(&self.name)?;
        if latest.location == self.metadata_file.location {
            return Ok(None);
        }
        Self::at(
            &self.dispatcher,
            Arc::clone(&self.metadata_store),
            self.location.clone(),
            &self.name,
            latest,
            None,
        )
        .map(Some)
    }

    /// What a scan over `filters` reads: the row groups of the files
    /// [`select_scan_files`](Self::select_scan_files) selects, in its
    /// order, with the rows their
    /// deletion vectors delete marked on them, and whether each column may
    /// hold a NULL in those files. The order is fixed by the version, and
    /// with file-local row-group order makes the same filters yield the
    /// same row-group positions for scanning and materialization. Of the
    /// delete manifests, those the partition summaries of their
    /// manifest-list entries cannot rule out are read. The filters are
    /// bound once, and projected once onto each partition spec, for the
    /// data manifests, the partitions and the delete manifests alike.
    ///
    /// The footers are taken from the cache, where those read are kept: a
    /// file whose footer is held and which has no deleted row is read
    /// through the row groups the cache holds, as they are, and a file with
    /// deleted rows from copies of those that hold one. Once the footers
    /// are taken, the parts of the manifests the files were selected
    /// through count as used again, so the cache lets go of the footers
    /// before them.
    pub fn select_scan(&self, filters: &[Expression]) -> Result<ScanSelection> {
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: self.name.to_string(),
            source: Box::new(source),
        };
        let selector = FileSelector::new(&self.metadata_file.metadata, filters)
            .map_err(invalid_partition_metadata)?;
        let SelectedFiles {
            files,
            read_parts,
            nullability,
        } = self.select_scan_files(&selector)?;
        // A scan no delete manifest applies to reads none.
        let selected_deletes =
            selector.select_manifests(self.manifest_list.listed_delete_manifests());
        let deletion_vectors = if selected_deletes.is_empty() {
            Vec::new()
        } else {
            let selected_deletes = selected_deletes.iter().map(|manifest| manifest.file());
            let delete_manifests = self.read_manifests(selected_deletes)?;
            read_deletion_vectors(delete_manifests, &files, &self.location, &self.dispatcher)?
        };
        let mut footers = self.open_footers(&files, FooterUse::Kept)?;
        FileMetadataCache::process_wide()
            .mark_metadata_parts_used(&self.manifest_cache_key, &read_parts);
        for (footer, deleted) in footers.iter_mut().zip(deletion_vectors) {
            if let Some(deleted) = deleted {
                footer.mark_deleted_rows(deleted);
            }
        }
        let row_groups: Vec<_> = footers
            .into_iter()
            .flat_map(|footer| footer.row_groups)
            .collect();
        let parquet = if row_groups.is_empty() {
            ParquetTable::empty(self.declared_schema())
        } else {
            ParquetTable::new(row_groups)
        };
        Ok(ScanSelection {
            row_groups: Arc::new(parquet),
            nullability,
        })
    }

    /// The row groups of each of `files`, distinct files of the table each
    /// under the path the table names it by, in the order given, with no
    /// row deleted. The footers are taken from the cache, where those read
    /// are kept.
    pub fn load_footers(&self, files: &[FileRef]) -> Result<Vec<FileRowGroups>> {
        self.open_footers(&self.locate_files(files)?, FooterUse::Kept)
    }

    /// [`load_footers`](Self::load_footers) for a pass over more files than
    /// are worth keeping the footers of: one the cache holds is not read,
    /// and the cache is left as it was.
    pub(crate) fn read_footers_unrecorded(&self, files: &[FileRef]) -> Result<Vec<FileRowGroups>> {
        self.open_footers(&self.locate_files(files)?, FooterUse::Unrecorded)
    }

    /// Each of `files`, under the path the table names it by, located for
    /// reading.
    fn locate_files(&self, files: &[FileRef]) -> Result<Vec<DataFile>> {
        files
            .iter()
            .map(|file| self.location.data_file_at(&file.path, file.size))
            .collect()
    }

    /// The row groups of each of `files`, distinct files of the table, in
    /// the order given, with no row deleted, shared with the cache and with
    /// whatever else took them. A file whose footer the cache holds is
    /// neither opened nor read; the others' footers are read and parsed in
    /// one pass over the pool, which is not run when the cache holds them
    /// all, and, with [`FooterUse::Kept`], cached.
    fn open_footers(
        &self,
        files: &[DataFile],
        footer_use: FooterUse,
    ) -> Result<Vec<FileRowGroups>> {
        let cache = FileMetadataCache::process_wide();
        let paths = files.iter().map(|file| file.file.path.as_str());
        let held = cache.find_footers(&self.cache_key, paths, footer_use);
        let unheld: Vec<DataFile> = files
            .iter()
            .zip(&held)
            .filter(|(_, held)| held.is_none())
            .map(|(file, _)| file.clone())
            .collect();
        let read = load_file_row_groups(&self.dispatcher, &unheld, self.columns.clone())?;
        if footer_use == FooterUse::Kept {
            let kept = read
                .iter()
                .map(|footer| {
                    let path = Arc::from(footer.file.path.as_str());
                    (path, footer.row_groups.clone())
                })
                .collect();
            cache.store_footers(&self.cache_key, kept);
        }
        let mut read = read.into_iter();
        Ok(files
            .iter()
            .zip(held)
            .map(|(file, held)| match held {
                Some(row_groups) => FileRowGroups {
                    file: file.file.clone(),
                    row_groups,
                },
                None => read
                    .next()
                    .expect("every file whose footer is not held was read"),
            })
            .collect())
    }

    /// Let go of the footers the cache holds for `files`, each under the
    /// path the table names it by: what the deleter of a table's files
    /// calls for them, since a footer held holds its file open, and a
    /// deleted local file keeps its blocks for as long as it is open.
    pub fn forget_footers(&self, files: &[ObjectPath]) -> Result<()> {
        let uris = files
            .iter()
            .map(|path| self.location.uri_for(path))
            .collect::<Result<Vec<String>>>()?;
        FileMetadataCache::process_wide()
            .remove_footers(&self.cache_key, uris.iter().map(String::as_str));
        Ok(())
    }

    /// The partition `key` names, whether or not this version has a file
    /// in it: its spec and its tuple are read from the key. A key that
    /// names no partition of the table is an error.
    pub fn find_partition(&self, key: &[u8]) -> Result<Arc<Partition>> {
        Ok(Arc::new(Partition::from_key(key, self.metadata())?))
    }

    /// The columns this version is sorted by, as the records of the files
    /// read with them share them.
    fn resolve_sort_columns(&self) -> Arc<SortColumns> {
        Arc::new(SortColumns::new(
            &self.metadata_file.metadata,
            self.columns(),
            &self.sort_by,
        ))
    }

    /// The sort order id a manifest entry records for a file a writer of
    /// this process reports as `sorted_by`: its rows in order of those
    /// columns, each ascending with NULLs first, or in no order when there
    /// are none. That is the table's default sort order's id when the order
    /// is exactly that one, and the unsorted order's otherwise: an order
    /// with a transformed, descending or nulls-last field is not the order
    /// the rows are in, whatever columns it names.
    pub fn find_sort_order_id_of_rows_sorted_by(&self, sorted_by: &[String]) -> i32 {
        match self.writer_sort_order_id {
            Some(sort_order_id) if sorted_by == self.sort_by => sort_order_id,
            _ => UNSORTED_ORDER_ID,
        }
    }

    /// The rows of every live data file, deleted ones included, as the
    /// manifest list counts them: no manifest is read.
    pub fn total_rows(&self) -> Result<u64> {
        self.manifest_list
            .row_count()
            .and_then(|rows| u64::try_from(rows).ok())
            .ok_or_else(|| Error::UncountedRows {
                table: self.name.to_string(),
            })
    }

    /// The live data files, as the manifest list counts them: no manifest
    /// is read. `None` when the manifest list does not count them.
    pub fn count_data_files(&self) -> Option<u64> {
        self.manifest_list.count_data_files()
    }

    /// Read and parse `manifests` of the table, in the order given, over the
    /// pool: each read whole, at the length its manifest list records, so one
    /// read again comes from the pool's caches.
    pub(crate) fn read_manifests<'a>(
        &self,
        manifests: impl IntoIterator<Item = &'a ManifestFile>,
    ) -> Result<Vec<Manifest>> {
        let objects = manifests
            .into_iter()
            .map(|manifest| {
                let length = manifest.manifest_length.max(0) as u64;
                self.location.data_file(&manifest.manifest_path, length)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut parsed = Vec::with_capacity(objects.len());
        for object in load_objects(&self.dispatcher, &objects)? {
            parsed.push(self.parse_manifest(object.file.path.as_str(), &object.bytes)?);
        }
        Ok(parsed)
    }

    /// Parse `bytes`, the manifest of the table at `path`.
    fn parse_manifest(&self, path: &str, bytes: &[u8]) -> Result<Manifest> {
        Manifest::parse_avro(bytes).map_err(|source| Error::MalformedMetadataObject {
            table: self.name.to_string(),
            path: path.to_string(),
            source: Box::new(source),
        })
    }

    /// The rows of the data files less the rows their deletion vectors
    /// delete, read from the manifests alone. Older writers may omit counts,
    /// and a delete manifest may fail to read or hold a delete file that is
    /// not applied; planning then has no statistic and COUNT uses the
    /// ordinary scan, which reports what is wrong.
    pub fn row_count(&self) -> Option<i64> {
        let rows = self.manifest_list.row_count()?;
        let delete_manifests = self
            .read_manifests(self.manifest_list.delete_manifests())
            .ok()?;
        let deleted = count_deleted_rows(delete_manifests).ok()?;
        rows.checked_sub(i64::try_from(deleted).ok()?)
    }

    /// The row count as the manifest list alone has it, with nothing read:
    /// exact when every delete file is a deletion vector, an estimate
    /// otherwise.
    pub fn estimated_row_count(&self) -> Option<i64> {
        self.manifest_list.estimated_row_count()
    }

    /// Whether each declared column may hold a NULL in a file of the table,
    /// read from the metadata alone: every column but one the table's
    /// current schema has as a required field, which Iceberg lets no file
    /// hold a NULL in.
    pub fn declared_nullability(&self) -> Vec<bool> {
        let schema = self.metadata_file.metadata.current_schema();
        self.columns()
            .iter()
            .map(|column| {
                schema
                    .field_by_name(&column.name)
                    .is_none_or(|field| !field.required)
            })
            .collect()
    }

    /// Whether each declared column may hold a NULL, as the table's schema
    /// declares it.
    pub fn schema_nullability(&self) -> Vec<bool> {
        self.metadata_file
            .metadata
            .current_schema()
            .as_struct()
            .fields()
            .iter()
            .map(|field| !field.required)
            .collect()
    }

    /// The declared columns as an Arrow schema, in their declared nullability.
    pub(crate) fn declared_schema(&self) -> ArrowSchemaRef {
        Arc::new(ArrowSchema::new(
            self.columns()
                .iter()
                .zip(self.schema_nullability())
                .map(|(column, nullable)| {
                    Field::new(
                        &column.name,
                        physical_arrow_type(&column.col_type),
                        nullable,
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    /// The name the table is known by in its datastore.
    pub fn name(&self) -> &SchemaQualifiedTableName {
        &self.name
    }

    pub fn metadata_store(&self) -> &Arc<dyn MetadataStore> {
        &self.metadata_store
    }

    pub fn location(&self) -> &TableLocation {
        &self.location
    }

    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// The metadata file this version was loaded from: where it is, and
    /// the metadata it holds.
    pub fn metadata_file(&self) -> &Arc<MetadataFile> {
        &self.metadata_file
    }

    pub fn metadata(&self) -> &TableMetadataRef {
        &self.metadata_file.metadata
    }

    /// The metadata file this version was loaded from. Every commit writes a
    /// new one, so it is the table's version.
    pub fn metadata_location(&self) -> &str {
        &self.metadata_file.location
    }

    /// The table's identity across every metadata version.
    pub fn uuid(&self) -> Uuid {
        self.metadata_file.metadata.uuid()
    }

    /// The revision a plan over this table is cached under: the table's
    /// identity across versions, and the metadata file this version was
    /// loaded from, which every commit replaces.
    pub fn revision(&self) -> TableRevision {
        TableRevision {
            identity: self.uuid().to_string(),
            version: self.metadata_location().to_string(),
        }
    }

    pub fn columns(&self) -> &[Column] {
        self.columns.columns()
    }

    /// The terms the table's default partition spec partitions by, in spec
    /// order, each as `CREATE TABLE` spells it: a column, or a transform of
    /// one (`day(ts)`). A file written under an earlier spec sits in a
    /// partition of that spec's fields.
    pub fn partition_by(&self) -> &[String] {
        &self.partition_by
    }

    /// The key columns the default partition spec cuts an INSERT's files
    /// by, or why an INSERT cannot partition by it.
    pub fn plan_partition_keys(&self) -> Result<PartitionKeys> {
        PartitionKeys::plan(
            &self.name.to_string(),
            self.metadata().current_schema(),
            self.metadata().default_partition_spec(),
            self.columns(),
        )
    }

    /// The identity-sorted columns, in sort-order order.
    pub fn sort_by(&self) -> &[String] {
        &self.sort_by
    }

    pub fn manifest_list(&self) -> &ManifestList {
        &self.manifest_list
    }

    /// The length of the current snapshot's manifest list, when this
    /// version knows it.
    pub fn manifest_list_size(&self) -> Option<u64> {
        self.manifest_list_size
    }
}
