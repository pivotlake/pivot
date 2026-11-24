//! One Iceberg table at one version: what every datastore reads a table
//! through and writes it through, given only the table's metadata
//! store.
//!
//! A table is loaded from its metadata document and its current snapshot's
//! manifest list, and that is all a version holds of its files: opening a
//! table decodes no manifest. The files are indexed a partition at a time
//! ([`PartitionIndex`]), when a scan, a commit or a maintenance pass first asks
//! for a partition, and the indexes of every table live in one cache
//! ([`FileMetadataCache`]), which keeps them within a budget of memory:
//! an index the cache let go of is built again when next asked for. The
//! same cache, within the same budget, keeps the footers a table's files
//! were found to have.
//!
//! # Which manifests hold a partition
//!
//! Nothing is kept of a manifest once it was decoded but the files that
//! went into a partition's index, in the cache. Which manifests a
//! partition's files are in is read off the manifest list every time: an
//! entry bounds each partition field's values across its manifest, so it
//! rules a partition in or out and says nothing more.
//!
//! A partition's index is asked for by the partition, and is built from
//! the manifests whose entries do not rule the partition's tuple out, each
//! decoded for the partition's files in it, of which it may hold none. The
//! index records the manifests it was built from, and is this version's
//! exactly when they are the ones this version's list does not rule out: a
//! manifest never changes. The cached index of an earlier or a later
//! version is brought to this one by the manifests the two differ in, and
//! only those are decoded. A version therefore reads its own files
//! whatever the cache holds or let go of, for as long as its manifests are
//! in the store.
//!
//! Which partitions a version has is not kept either. An entry that leaves
//! its manifest one partition, of a spec without fields or with every
//! field bounded to a single value, names it; any other manifest is
//! decoded for the partitions it holds whenever they are asked for. A scan
//! asks the manifests its filters cannot rule out by their entries, so it
//! decodes none of the others, and decodes those of them that hold several
//! partitions every time it is planned.
//!
//! A scan indexes the partitions its filters keep together, so a manifest
//! is decoded once to index all of them, and it holds the indexes of every
//! one, whatever the cache keeps of them, until it has selected its files.
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
//! A version carries nothing of the versions before it. It finds its
//! table's entries in the cache under a key computed from itself: where
//! the table is, which table it is, and what a file of it is read and
//! indexed under, its schema, the Pivot types its fields are read as, and
//! its default partition spec and sort order. Every version that computes
//! an equal key finds the same entries, whether it was derived from another
//! by a refresh or a commit, or loaded on its own; a version under
//! another schema, partition spec or sort order finds none of them.
//! Nothing leaves the cache because a table object went away: an entry
//! stays until the budget lets go of it, or until whoever deletes its file
//! says so ([`Table::forget_footers`], [`forget_cached_table`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::{Arc, OnceLock};

use arrow_schema::{Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{
    EncodedManifest, FormatVersion, Manifest, ManifestFile, TableMetadata, TableMetadataRef,
};
use object_storage::{DataFile, FileRef, ObjectPath, load_objects};
use parquet_engine::file_entry::FileEntry;
use parquet_engine::{
    FileRowGroups, ParquetTable, ScanSelection, TableColumns, load_file_row_groups,
};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::expression::Expression;
use planner::types::physical_arrow_type;
use table_file_index::{FileMetadataCache, LookupUse, TableKey, VersionOrder};
use uuid::Uuid;

use crate::files::data_files::DataFileEntry;
use crate::files::deletes::{count_deleted_rows, read_deletion_vectors};
use crate::files::file_index::{
    self, DecodeContext, DecodedManifest, FileSelector, FilesByPartition, IndexedFile, Partition,
    PartitionIndex, SortColumns,
};
use crate::files::manifests::{EntryPartitions, ListedManifest, ManifestList};
use crate::files::partitions_from_filter::MAX_PARTITIONS_FROM_FILTER;
use crate::metadata::layout::TableLocation;
use crate::metadata::metadata_store::MetadataFile;
use crate::metadata::metadata_store::MetadataStore;
use crate::types::columns::{LOGICAL_TYPES_PROPERTY, identity_columns, read_table_columns};
use crate::types::partitioning::{PartitionKeys, format_partition_fields};
use crate::types::schema::{UNSORTED_ORDER_ID, find_writer_sort_order_id};
use crate::{Error, Result};

/// How many manifests are read, and held parsed, at a time.
pub(crate) const MANIFESTS_PER_READ: usize = 16;

/// How many partitions the walk over every partition of a table
/// ([`Table::visit_partitions`]) has indexed at a time: it goes over a
/// whole table and may end before it reaches the rest.
const WALKED_PARTITIONS_PER_LOAD: usize = 16;

/// How much of the files a pass decoded ahead of the partitions it will
/// index it keeps, by [`IndexedFile::estimate_heap_bytes`]; a manifest past
/// it is decoded again when its partitions are indexed.
const DECODED_FILES_HELD_BYTES: u64 = 256 * 1024 * 1024;

/// Which declared columns may hold a NULL in the files a scan selected.
struct SelectedNulls {
    /// The field id of each declared column; `None` for a column the
    /// table's schema gives none.
    field_ids: Vec<Option<i32>>,
    /// Whether the table's schema lets a file hold a NULL in each declared
    /// column.
    declared_nullability: Vec<bool>,
    /// Whether some selected file may hold a NULL in each declared column.
    may_hold_nulls: Vec<bool>,
    is_any_file_selected: bool,
}

impl SelectedNulls {
    fn new(table: &Table) -> Self {
        let schema = table.metadata().current_schema();
        Self {
            field_ids: table
                .columns()
                .iter()
                .map(|column| schema.field_id_by_name(&column.name))
                .collect(),
            declared_nullability: table.declared_nullability(),
            may_hold_nulls: vec![false; table.columns().len()],
            is_any_file_selected: false,
        }
    }

    /// Count `file` among the selected: a column may hold a NULL in it
    /// unless its entry counts none, which an entry of a file written
    /// without the column does not.
    fn observe(&mut self, file: &IndexedFile) {
        self.is_any_file_selected = true;
        let null_counts = file.data.null_value_counts();
        for (field_id, may_hold_nulls) in self.field_ids.iter().zip(&mut self.may_hold_nulls) {
            if *may_hold_nulls {
                continue;
            }
            let is_null_free =
                field_id.is_some_and(|field_id| null_counts.get(&field_id) == Some(&0));
            *may_hold_nulls = !is_null_free;
        }
    }

    /// Whether each declared column may hold a NULL in a selected file.
    fn into_nullability(self) -> Vec<bool> {
        self.declared_nullability
            .iter()
            .zip(self.may_hold_nulls)
            .map(|(is_declared_nullable, may_hold_nulls)| {
                *is_declared_nullable && (may_hold_nulls || !self.is_any_file_selected)
            })
            .collect()
    }
}

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
    /// The key this version's partition indexes and the footers of its
    /// files are cached under ([`build_cache_key`]).
    cache_key: TableKey,
    /// The files the commit on top of this version removed, once it has
    /// committed, whose footers are let go of when nothing holds this
    /// version or an earlier one any more.
    files_removed_next: Arc<FilesRemovedNext>,
}

/// The files the commit on top of one version of a table removed. No later
/// version lists them, so the cache lets go of their footers, but not while
/// a transaction that began before the commit may still plan over them: a
/// version holds its own of these, and each holds the one of the next
/// version derived from its version, so the footers are let go of once the
/// version and every earlier one of its line are dropped. The cache may
/// still let go of them sooner to stay within its budget.
struct FilesRemovedNext {
    /// Where the version the commit was on top of stands.
    version: VersionOrder,
    /// The key that version's footers are cached under.
    cache_key: TableKey,
    /// The URIs of the files removed; unset until the commit succeeds, and
    /// for a version another process committed on top of.
    uris: OnceLock<Vec<String>>,
    /// The same of the next version derived from this one's.
    of_later_version: OnceLock<Arc<FilesRemovedNext>>,
    /// The pool whose ring caches hold bytes of the files.
    dispatcher: DataFlowDispatcher,
}

impl FilesRemovedNext {
    fn new(
        version: VersionOrder,
        cache_key: TableKey,
        dispatcher: &DataFlowDispatcher,
    ) -> Arc<Self> {
        Arc::new(Self {
            version,
            cache_key,
            uris: OnceLock::new(),
            of_later_version: OnceLock::new(),
            dispatcher: dispatcher.clone(),
        })
    }

    /// Let go of everything this process caches of the files at `uris`,
    /// which `cache_key` holds the footers of: each footer, and with it
    /// what the ring's caches hold of the file's bytes. A file whose footer
    /// is not held keeps its bytes in the ring until their slots are
    /// reclaimed.
    fn forget_files<'u>(
        dispatcher: &DataFlowDispatcher,
        cache_key: &TableKey,
        uris: impl Iterator<Item = &'u str>,
    ) {
        let footers = FileMetadataCache::process_wide().remove_footers(cache_key, uris);
        dispatcher.forget_cached_files(
            footers
                .iter()
                .filter_map(|row_groups| row_groups.first())
                .map(|row_group| &row_group.open_file),
        );
    }

    /// The one of the version standing at `version`, derived from this
    /// one's: the one a version derived earlier started, so every object of
    /// one version shares it, else a new one after the last of the line.
    fn find_or_start_later(
        self: &Arc<Self>,
        version: VersionOrder,
        cache_key: &TableKey,
    ) -> Arc<Self> {
        let mut held = Arc::clone(self);
        while held.version < version {
            let later = Arc::clone(
                held.of_later_version
                    .get_or_init(|| Self::new(version, cache_key.clone(), &self.dispatcher)),
            );
            held = later;
        }
        held
    }
}

impl Drop for FilesRemovedNext {
    /// Let go of the footers of the files removed, and of those of each
    /// later version that nothing else holds, one after the other: a long
    /// line let go of at once is not walked by nested drops.
    fn drop(&mut self) {
        if let Some(uris) = self.uris.get() {
            Self::forget_files(
                &self.dispatcher,
                &self.cache_key,
                uris.iter().map(String::as_str),
            );
        }
        let mut later = self.of_later_version.take();
        while let Some(mut unheld) = later.and_then(Arc::into_inner) {
            later = unheld.of_later_version.take();
        }
    }
}

/// The key the version `metadata` describes of the table at `location`
/// finds its cached partition indexes and footers under: equal for exactly
/// the versions of one table that read a file alike, however each came to
/// be. The table is named by its URI. The rest tells which table that is,
/// by its UUID, and what an entry was built under: the ids of the schema,
/// which a footer is parsed against, and of the default partition spec and
/// sort order, which a partition's files are indexed and ordered under,
/// and the [`LOGICAL_TYPES_PROPERTY`], the Pivot types the schema's fields
/// are read as. Each id is written whole and the property last, quoted, so
/// two versions that differ in any of them never write the same key.
fn build_cache_key(location: &TableLocation, metadata: &TableMetadata) -> TableKey {
    let logical_types = metadata.properties().get(LOGICAL_TYPES_PROPERTY);
    TableKey {
        table: Arc::from(location.table_uri()),
        reading: Arc::from(format!(
            "{} {} {} {} {logical_types:?}",
            metadata.uuid(),
            metadata.current_schema_id(),
            metadata.default_partition_spec_id(),
            metadata.default_sort_order_id(),
        )),
    }
}

/// Let go of everything the process's cache holds of the table at
/// `location`, whichever schema, partition spec and sort order its
/// versions read it under: what whoever deletes the table's directory
/// calls first, since a footer held holds its file open, and a deleted
/// local file keeps its blocks for as long as it is open.
pub fn forget_cached_table(location: &TableLocation) {
    FileMetadataCache::process_wide().remove_tables(&location.table_uri());
    forget_held_manifests(&location.table_uri());
}

/// Let go of the manifests the format's crate holds, as the commits of this
/// process wrote them, of the tables under the directory at
/// `directory_uri`. They are then read from the store, as they are by a
/// process that has just started, which holds none.
pub fn forget_held_manifests(directory_uri: &str) {
    EncodedManifest::forget_under(directory_uri);
}

/// What a scan over some filters selected of a table version.
pub(crate) struct SelectedFiles {
    /// The files, each located for reading, in the order they are scanned.
    pub(crate) files: Vec<DataFile>,
    /// The keys of the partitions the files were selected from.
    partitions: Vec<Arc<[u8]>>,
    /// Whether each declared column may hold a NULL in one of the files
    /// (see [`ScanSelection::nullability`]).
    pub(crate) nullability: Vec<bool>,
}

/// What a taker of footers leaves of itself in the cache.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FooterUse {
    /// The footers found count as used, and those read are kept.
    Kept,
    /// The cache is left as it was: nothing counts as used and nothing
    /// read is kept.
    Unrecorded,
}

/// What a walk over a table's partitions does with each partition and its
/// files: `Break` ends the walk.
pub type VisitPartition<'v> =
    dyn FnMut(&Arc<Partition>, &PartitionIndex) -> Result<ControlFlow<()>> + 'v;

/// What a walk over the files a scan selected does with each file, which
/// it is handed with the key of the file's partition.
type VisitSelectedFile<'v> = dyn FnMut(&Arc<[u8]>, &IndexedFile) -> Result<()> + 'v;

/// A partition beside its key, encoded once for everything a pass does
/// with the partition: finding its index in the cache, taking its decoded
/// files, storing its index, and counting the index as used.
struct KeyedPartition {
    key: Arc<[u8]>,
    partition: Arc<Partition>,
}

impl KeyedPartition {
    /// `partition` with its key, encoded here.
    fn new(partition: &Arc<Partition>) -> Self {
        Self {
            key: partition.encode_key(),
            partition: Arc::clone(partition),
        }
    }
}

/// A partition whose index a load has to build: where the partition sits
/// among those asked for, the manifests whose manifest-list entries do not
/// rule it out, and the index the cache held for it.
struct UnbuiltIndex {
    position: usize,
    candidates: Vec<Arc<str>>,
    held: Option<Arc<PartitionIndex>>,
}

/// The files manifests were decoded for, ahead of the partition indexes
/// they go into, by manifest path and partition key. Every manifest held
/// was decoded for the same partitions, so a partition among them that a
/// held manifest has no files of has none in it.
#[derive(Default)]
struct DecodedFiles {
    files: HashMap<Arc<str>, FilesByPartition>,
    heap_bytes: u64,
}

impl DecodedFiles {
    /// Keep `files`, the files a decode of the manifest at `manifest_path`
    /// was asked for, by partition key. With `is_bounded`, they are kept
    /// only while what is held stays within
    /// [`DECODED_FILES_HELD_BYTES`].
    fn keep(&mut self, manifest_path: &Arc<str>, files: FilesByPartition, is_bounded: bool) {
        let heap_bytes: u64 = files
            .values()
            .flatten()
            .map(|file| file.estimate_heap_bytes())
            .sum();
        if is_bounded && self.heap_bytes + heap_bytes > DECODED_FILES_HELD_BYTES {
            return;
        }
        self.heap_bytes += heap_bytes;
        self.files.insert(Arc::clone(manifest_path), files);
    }

    /// Whether the manifest at `manifest_path` was decoded and kept.
    fn holds_manifest(&self, manifest_path: &str) -> bool {
        self.files.contains_key(manifest_path)
    }

    /// Hand over the files of the partition `key` in the manifest at
    /// `manifest_path`, which was decoded and kept: none when the manifest
    /// holds no file of the partition.
    fn take(&mut self, manifest_path: &str, key: &[u8]) -> Vec<Arc<IndexedFile>> {
        let files = self
            .files
            .get_mut(manifest_path)
            .and_then(|partitions| partitions.remove(key))
            .unwrap_or_default();
        let heap_bytes: u64 = files.iter().map(|file| file.estimate_heap_bytes()).sum();
        self.heap_bytes -= heap_bytes;
        files
    }
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
    /// it finds the partition indexes built and the footers parsed by the
    /// versions of the table that read a file as it does.
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
        let version = VersionOrder {
            sequence: metadata.last_sequence_number(),
            updated_ms: metadata.last_updated_ms(),
        };
        let files_removed_next = FilesRemovedNext::new(version, cache_key.clone(), dispatcher);
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
            files_removed_next,
        })
    }

    /// This version as one derived from `earlier`, by a refresh or a commit:
    /// the footers of the files a commit between the two removed are held
    /// for as long as `earlier` is.
    pub(crate) fn derived_from(mut self, earlier: &Table) -> Table {
        self.files_removed_next = earlier
            .files_removed_next
            .find_or_start_later(self.version_order(), &self.cache_key);
        self
    }

    /// Record the files, by their URIs, that the commit on top of this
    /// version removed.
    pub(crate) fn record_files_removed_next(&self, uris: Vec<String>) {
        self.files_removed_next
            .uris
            .set(uris)
            .expect("one commit succeeds on top of a version");
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
        .map(|latest| Some(latest.derived_from(self)))
    }

    /// What a scan over `filters` reads: the row groups of the files
    /// `select_scan_files` selects, in its order, with the rows their
    /// deletion vectors delete marked on them, and whether each column may
    /// hold a NULL in those files. The order is fixed by the version, and
    /// with file-local row-group order makes the same filters yield the
    /// same row-group positions for scanning and materialization. Of the
    /// delete manifests, those the partition summaries of their
    /// manifest-list entries cannot rule out are read.
    ///
    /// The footers are taken from the cache, where those read are kept: a
    /// file whose footer is held and which has no deleted row is read
    /// through the row groups the cache holds, as they are, and a file with
    /// deleted rows from copies of those that hold one. Once the footers
    /// are taken, the indexes the files were selected from count as used
    /// again, so the cache lets go of a partition's footers before its
    /// index.
    pub fn select_scan(&self, filters: &[Expression]) -> Result<ScanSelection> {
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: self.name.to_string(),
            source: Box::new(source),
        };
        let SelectedFiles {
            files,
            partitions,
            nullability,
        } = self.select_scan_files(filters)?;
        let selected_deletes = self
            .manifest_list
            .select_delete_manifests(filters)
            .map_err(invalid_partition_metadata)?;
        let delete_manifests = self.read_manifests(selected_deletes)?;
        let deletion_vectors =
            read_deletion_vectors(delete_manifests, &files, &self.location, &self.dispatcher)?;
        let mut footers = self.open_footers(&files, FooterUse::Kept)?;
        FileMetadataCache::process_wide().mark_indexes_used(&self.cache_key, &partitions);
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

    /// What a scan over `filters` selects: the files it reads, each located
    /// for reading, in the order
    /// [`visit_selected_files`](Self::visit_selected_files) visits them,
    /// the partitions they were selected from, and whether each declared
    /// column may hold a NULL in one of them.
    pub(crate) fn select_scan_files(&self, filters: &[Expression]) -> Result<SelectedFiles> {
        let mut files = Vec::new();
        let mut partitions: Vec<Arc<[u8]>> = Vec::new();
        let mut nulls = SelectedNulls::new(self);
        self.visit_selected_files(filters, &mut |partition_key, file| {
            nulls.observe(file);
            // Files are visited a partition at a time.
            if partitions.last() != Some(partition_key) {
                partitions.push(Arc::clone(partition_key));
            }
            files.push(
                self.location
                    .data_file_at(&file.facts.file().path, file.facts.file().size)?,
            );
            Ok(())
        })?;
        Ok(SelectedFiles {
            files,
            partitions,
            nullability: nulls.into_nullability(),
        })
    }

    /// Visit the files a scan over `filters` reads, each with the key of
    /// its partition: those that the files' partition values and column
    /// bounds cannot rule out, by partition, then in the order a
    /// partition's files stand in.
    ///
    /// Only the partitions the filters keep are indexed. Filters that leave
    /// every partition field only a few values name those partitions
    /// themselves, and a manifest is read only to index one of them the
    /// cache does not hold. Otherwise they are found in the manifests whose
    /// manifest-list entries the filters cannot rule out, and no other
    /// data manifest is read: of those manifests, the ones whose entries do
    /// not name their one partition are decoded for the partitions they
    /// hold.
    ///
    /// The partitions kept are indexed together, so a manifest that holds
    /// files of several of them is decoded once to index them all, and the
    /// indexes of every one of them are held, whatever the cache keeps of
    /// them, until the files of the last are visited.
    fn visit_selected_files(
        &self,
        filters: &[Expression],
        visit: &mut VisitSelectedFile<'_>,
    ) -> Result<()> {
        let selector =
            FileSelector::new(&self.metadata_file.metadata, filters).map_err(|source| {
                Error::InvalidPartitionMetadata {
                    table: self.name.to_string(),
                    source: Box::new(source),
                }
            })?;
        let mut file_count = 0_usize;
        let mut decoded = DecodedFiles::default();
        let partitions = self.select_partitions(&selector, &mut decoded)?;
        let indexes = self.load_partitions(&partitions, &mut decoded)?;
        for (keyed, index) in partitions.iter().zip(&indexes) {
            for file in selector.select(index) {
                visit(&keyed.key, file)?;
                file_count += 1;
            }
        }
        tracing::debug!(
            table = %self.name,
            partitions = partitions.len(),
            files = file_count,
            "selected Iceberg files"
        );
        Ok(())
    }

    /// The partitions whose tuple the filters of `selector` keep, in key
    /// order: those computed from the filters alone
    /// ([`compute_partitions_from_filter`](Self::compute_partitions_from_filter)),
    /// or else those listed from the data manifests whose manifest-list
    /// entries the filters cannot rule out
    /// ([`list_partitions_from_manifests`](Self::list_partitions_from_manifests)),
    /// each with its key. A manifest decoded to learn its partitions leaves
    /// the files of the kept ones in `ahead`.
    fn select_partitions(
        &self,
        selector: &FileSelector,
        ahead: &mut DecodedFiles,
    ) -> Result<Vec<KeyedPartition>> {
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: self.name.to_string(),
            source: Box::new(source),
        };
        let manifests = self
            .manifest_list
            .select_data_manifests(selector.row_filters())
            .map_err(invalid_partition_metadata)?;
        let mut tuple_values = Vec::new();
        let mut keeps = |partition: &Partition| {
            selector
                .may_partition_match(partition, &mut tuple_values)
                .map_err(invalid_partition_metadata)
        };
        if let Some(from_filter) =
            self.compute_partitions_from_filter(selector, &manifests, &mut keeps)?
        {
            return Ok(from_filter);
        }
        let mut partitions = self.list_partitions_from_manifests(&manifests, &mut keeps, ahead)?;
        let mut kept = Vec::with_capacity(partitions.len());
        while let Some((key, partition)) = partitions.pop_first() {
            if keeps(&partition)? {
                kept.push(KeyedPartition { key, partition });
            }
        }
        Ok(kept)
    }

    /// The partitions a scan over the filters of `selector` may read a
    /// file of, each with its key, in key order, computed from the filters
    /// with no manifest read: of the combinations of the values the filters
    /// leave the partition fields, under each spec `manifests` were written
    /// under, those whose tuple `keeps` accepts and that the manifest-list
    /// entry of one of `manifests`, the data manifests the filters cannot
    /// rule out, admits.
    /// One of them may hold no file. `None` when the filters leave a field
    /// of one of the specs unrestricted, or leave more than
    /// [`MAX_PARTITIONS_FROM_FILTER`] combinations: the partitions are then
    /// listed from the manifests.
    fn compute_partitions_from_filter(
        &self,
        selector: &FileSelector,
        manifests: &[&ListedManifest],
        keeps: &mut dyn FnMut(&Partition) -> Result<bool>,
    ) -> Result<Option<Vec<KeyedPartition>>> {
        let mut spec_ids: Vec<i32> = manifests
            .iter()
            .map(|manifest| manifest.file().partition_spec_id)
            .collect();
        spec_ids.sort_unstable();
        spec_ids.dedup();
        let Some(from_filter) = selector.enumerate_partitions_from_filter(
            self.metadata(),
            &spec_ids,
            MAX_PARTITIONS_FROM_FILTER,
        ) else {
            return Ok(None);
        };
        let mut partitions = BTreeMap::new();
        for partition in from_filter {
            let is_admitted = manifests
                .iter()
                .any(|manifest| manifest.may_hold_partition(&partition));
            if is_admitted && keeps(&partition)? {
                partitions.insert(partition.encode_key(), Arc::new(partition));
            }
        }
        let keyed = partitions
            .into_iter()
            .map(|(key, partition)| KeyedPartition { key, partition });
        Ok(Some(keyed.collect()))
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
        let lookup_use = match footer_use {
            FooterUse::Kept => LookupUse::Counted,
            FooterUse::Unrecorded => LookupUse::Uncounted,
        };
        let held = cache.find_footers(&self.cache_key, paths, lookup_use);
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
    /// path the table names it by, and of what the ring's caches hold of
    /// their bytes: what the deleter of a table's files calls for them,
    /// since a footer held holds its file open, and a deleted local file
    /// keeps its blocks for as long as it is open.
    pub fn forget_footers(&self, files: &[ObjectPath]) -> Result<()> {
        let uris = files
            .iter()
            .map(|path| self.location.uri_for(path))
            .collect::<Result<Vec<String>>>()?;
        FilesRemovedNext::forget_files(
            &self.dispatcher,
            &self.cache_key,
            uris.iter().map(String::as_str),
        );
        Ok(())
    }

    /// Every partition this version has a file in, in key order. The data
    /// manifests whose manifest-list entries do not name their one
    /// partition are read for the partitions they hold, a few at a time,
    /// and nothing of them is kept.
    pub fn partitions(&self) -> Result<Vec<Arc<Partition>>> {
        let manifests: Vec<&ListedManifest> =
            self.manifest_list.listed_data_manifests().iter().collect();
        let partitions = self.list_partitions_from_manifests(
            &manifests,
            &mut |_| Ok(false),
            &mut DecodedFiles::default(),
        )?;
        Ok(partitions.into_values().collect())
    }

    /// The partitions `manifests`, data manifests of this version, hold a
    /// live file of, by key. A manifest whose manifest-list entry names its
    /// one partition is not read. The others are read over the pool, a few at a time, and
    /// decoded: of each, the files of the partitions `wants_files` accepts
    /// are left in `ahead`, as far as its bound allows, and nothing else
    /// is kept.
    fn list_partitions_from_manifests(
        &self,
        manifests: &[&ListedManifest],
        wants_files: &mut dyn FnMut(&Partition) -> Result<bool>,
        ahead: &mut DecodedFiles,
    ) -> Result<BTreeMap<Arc<[u8]>, Arc<Partition>>> {
        let mut partitions = BTreeMap::new();
        let mut unnamed: Vec<&ListedManifest> = Vec::new();
        // The partition the manifest before named: manifests that name
        // their one partition mostly come in runs of one partition, whose
        // key is encoded once for the run.
        let mut latest: Option<Arc<Partition>> = None;
        for manifest in manifests {
            match manifest.read_partitions(self.metadata()) {
                EntryPartitions::None => {}
                EntryPartitions::One(partition) => {
                    let is_of_latest = latest.as_ref().is_some_and(|latest| {
                        latest.spec_id() == partition.spec_id()
                            && latest.tuple() == partition.tuple()
                    });
                    if !is_of_latest {
                        let partition = Arc::new(partition);
                        partitions
                            .entry(partition.encode_key())
                            .or_insert_with(|| Arc::clone(&partition));
                        latest = Some(partition);
                    }
                }
                EntryPartitions::Unknown => unnamed.push(manifest),
            }
        }
        for manifests in unnamed.chunks(MANIFESTS_PER_READ) {
            let parsed = self.read_manifests(manifests.iter().map(|manifest| manifest.file()))?;
            for (listed, manifest) in manifests.iter().zip(parsed) {
                let decoded = self.decode_manifest(listed, manifest, &mut |_, partition| {
                    wants_files(partition)
                })?;
                for (key, partition) in decoded.partitions {
                    partitions.entry(key).or_insert(partition);
                }
                ahead.keep(listed.path(), decoded.files, true);
            }
        }
        Ok(partitions)
    }

    /// The partition `key` names, whether or not this version has a file
    /// in it: its spec and its tuple are read from the key. A key that
    /// names no partition of the table is an error.
    pub fn find_partition(&self, key: &[u8]) -> Result<Arc<Partition>> {
        Ok(Arc::new(Partition::from_key(key, self.metadata())?))
    }

    /// The files this version reads in `partition`: those the current
    /// snapshot's data manifests list, none for a partition the version has
    /// no file in. Built from the manifests
    /// whose manifest-list entries do not rule the partition out, on the
    /// index the cache holds for it when there is one.
    pub fn partition_files(&self, partition: &Arc<Partition>) -> Result<Arc<PartitionIndex>> {
        self.load_partition(KeyedPartition::new(partition))
    }

    /// The index of `partition` at this version: see
    /// [`partition_files`](Self::partition_files).
    fn load_partition(&self, partition: KeyedPartition) -> Result<Arc<PartitionIndex>> {
        let mut indexes = self.load_partitions(&[partition], &mut DecodedFiles::default())?;
        Ok(indexes
            .pop()
            .expect("one index is loaded for each partition asked for"))
    }

    /// The partition `key` names, beside that key.
    fn find_keyed_partition(&self, key: &[u8]) -> Result<KeyedPartition> {
        Ok(KeyedPartition {
            key: Arc::from(key),
            partition: self.find_partition(key)?,
        })
    }

    /// Index the partitions with the keys `keys` together, so the cache
    /// holds their indexes for the calls that then ask for them one by
    /// one: a manifest is decoded once for all of them, not once for each.
    /// The files decoded for them are held until every index is built, so
    /// the caller bounds how many keys it hands over.
    pub fn index_partitions(&self, keys: &[Vec<u8>]) -> Result<()> {
        let partitions = keys
            .iter()
            .map(|key| self.find_keyed_partition(key))
            .collect::<Result<Vec<_>>>()?;
        self.load_partitions(&partitions, &mut DecodedFiles::default())?;
        Ok(())
    }

    /// Visit every partition this version has a file in with its files, in
    /// key order, a few partitions indexed at a time; `Break` ends the
    /// walk.
    pub fn visit_partitions(&self, visit: &mut VisitPartition<'_>) -> Result<()> {
        let mut ahead = DecodedFiles::default();
        let manifests: Vec<&ListedManifest> =
            self.manifest_list.listed_data_manifests().iter().collect();
        let partitions: Vec<KeyedPartition> = self
            .list_partitions_from_manifests(&manifests, &mut |_| Ok(true), &mut ahead)?
            .into_iter()
            .map(|(key, partition)| KeyedPartition { key, partition })
            .collect();
        for partitions in partitions.chunks(WALKED_PARTITIONS_PER_LOAD) {
            let indexes = self.load_partitions(partitions, &mut ahead)?;
            for (keyed, index) in partitions.iter().zip(&indexes) {
                if visit(&keyed.partition, index)?.is_break() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Decode `manifest`, the parsed data manifest `listed` names: the
    /// partitions it holds a live file of, and the files of those `wants`
    /// accepts, which is asked of each by its key and the partition.
    fn decode_manifest(
        &self,
        listed: &ListedManifest,
        manifest: Manifest,
        wants: &mut dyn FnMut(&[u8], &Partition) -> Result<bool>,
    ) -> Result<DecodedManifest> {
        let table_name = self.name.to_string();
        let sort_columns = self.resolve_sort_columns();
        let context = DecodeContext {
            table: &table_name,
            sort_columns: &sort_columns,
            metadata: self.metadata(),
            location: &self.location,
        };
        IndexedFile::decode_manifest(listed.file(), manifest, &context, wants)
    }

    /// The index of each of `partitions` at this version, in the order
    /// given.
    ///
    /// A partition's index is built from the data manifests whose
    /// manifest-list entries do not rule the partition out. The index the
    /// cache holds for the partition is this version's when it was built
    /// from exactly those manifests; otherwise this version's is built on
    /// it by the manifests the two differ in, and cached. The partition's
    /// files in a manifest that joins are taken from `ahead` when it holds
    /// the manifest, which was then decoded for a set of partitions these
    /// are among; the other manifests are decoded now, each once for all
    /// the partitions asked for. What is decoded now is held until every
    /// index is built, and only the partitions asked for bound it.
    fn load_partitions(
        &self,
        partitions: &[KeyedPartition],
        ahead: &mut DecodedFiles,
    ) -> Result<Vec<Arc<PartitionIndex>>> {
        let cache = FileMetadataCache::process_wide();
        let _indexing = cache.lock_table(&self.cache_key);
        let listed = self.manifest_list.listed_data_manifests();
        let held = cache.find_indexes::<PartitionIndex>(
            &self.cache_key,
            partitions.iter().map(|keyed| keyed.key.as_ref()),
            LookupUse::Counted,
        );
        let mut indexes: Vec<Option<Arc<PartitionIndex>>> = Vec::with_capacity(partitions.len());
        let mut unbuilt: Vec<UnbuiltIndex> = Vec::new();
        for (position, (keyed, held)) in partitions.iter().zip(held).enumerate() {
            let candidates: Vec<&ListedManifest> = listed
                .iter()
                .filter(|manifest| manifest.may_hold_partition(&keyed.partition))
                .collect();
            let held = held.map(|(index, _)| index);
            match held {
                Some(index)
                    if index.is_built_from(candidates.iter().map(|manifest| manifest.path())) =>
                {
                    indexes.push(Some(index));
                }
                held => {
                    indexes.push(None);
                    unbuilt.push(UnbuiltIndex {
                        position,
                        candidates: candidates
                            .iter()
                            .map(|manifest| Arc::clone(manifest.path()))
                            .collect(),
                        held,
                    });
                }
            }
        }

        let mut undecoded: HashSet<&str> = HashSet::new();
        for partition in &unbuilt {
            for manifest in &partition.candidates {
                let is_held = partition
                    .held
                    .as_ref()
                    .is_some_and(|index| index.holds_manifest(manifest));
                if !is_held && !ahead.holds_manifest(manifest) {
                    undecoded.insert(manifest);
                }
            }
        }
        let mut decoded = DecodedFiles::default();
        if !undecoded.is_empty() {
            let unbuilt_keys: HashSet<&[u8]> = unbuilt
                .iter()
                .map(|partition| partitions[partition.position].key.as_ref())
                .collect();
            let mut is_unbuilt = |key: &[u8], _: &Partition| Ok(unbuilt_keys.contains(key));
            let undecoded: Vec<&ListedManifest> = listed
                .iter()
                .filter(|manifest| undecoded.contains(manifest.path().as_ref()))
                .collect();
            // A manifest the format's crate holds, as one this process
            // wrote is, is neither fetched nor parsed, and a file of it
            // that the index being advanced has a record of keeps the
            // record.
            let held_indexes: HashMap<&[u8], &PartitionIndex> = unbuilt
                .iter()
                .filter_map(|partition| {
                    let key = partitions[partition.position].key.as_ref();
                    Some((key, partition.held.as_deref()?))
                })
                .collect();
            let find_record = |key: &[u8], path: &ObjectPath| {
                held_indexes
                    .get(key)
                    .and_then(|index| index.find(path))
                    .cloned()
            };
            let mut unheld: Vec<&ListedManifest> = Vec::new();
            for listed in undecoded {
                let Some(held) = EncodedManifest::find(&listed.file().manifest_path) else {
                    unheld.push(listed);
                    continue;
                };
                let table_name = self.name.to_string();
                let sort_columns = self.resolve_sort_columns();
                let context = DecodeContext {
                    table: &table_name,
                    sort_columns: &sort_columns,
                    metadata: self.metadata(),
                    location: &self.location,
                };
                let files = IndexedFile::decode_held_manifest(
                    listed.file(),
                    &held,
                    &context,
                    &mut is_unbuilt,
                    &find_record,
                )?
                .files;
                decoded.keep(listed.path(), files, false);
            }
            for manifests in unheld.chunks(MANIFESTS_PER_READ) {
                let parsed =
                    self.read_manifests(manifests.iter().map(|manifest| manifest.file()))?;
                for (listed, manifest) in manifests.iter().zip(parsed) {
                    let files = self
                        .decode_manifest(listed, manifest, &mut is_unbuilt)?
                        .files;
                    decoded.keep(listed.path(), files, false);
                }
            }
        }

        let built_at = self.version_order();
        for partition in unbuilt {
            let key = &partitions[partition.position].key;
            let index = Arc::new(PartitionIndex::advance(
                partition.held.as_deref(),
                &partition.candidates,
                |manifest| {
                    if ahead.holds_manifest(manifest) {
                        ahead.take(manifest, key)
                    } else {
                        decoded.take(manifest, key)
                    }
                },
            ));
            cache.store_index(&self.cache_key, key, &index, built_at);
            indexes[partition.position] = Some(index);
        }
        Ok(indexes
            .into_iter()
            .map(|index| index.expect("every partition asked for is indexed"))
            .collect())
    }

    /// Where this version stands among the versions of its table.
    fn version_order(&self) -> VersionOrder {
        VersionOrder {
            sequence: self.metadata().last_sequence_number(),
            updated_ms: self.metadata().last_updated_ms(),
        }
    }

    /// The partitions holding a file that a commit after the one numbered
    /// `sequence_number` added, in key order, whoever committed and
    /// whether the file is an insert's or a rewrite's output. Only the
    /// data manifests written after that commit, as the manifest list
    /// numbers them, are read, a few at a time, and nothing of them is
    /// kept: an entry of one records the commit that added its file, which
    /// a manifest merged or rewritten later carries over. A version 1
    /// table numbers no commit, so every partition counts.
    pub fn list_partitions_added_after(&self, sequence_number: i64) -> Result<Vec<Arc<Partition>>> {
        if self.metadata().format_version() == FormatVersion::V1 {
            return self.partitions();
        }
        let written_after: Vec<&ListedManifest> = self
            .manifest_list
            .listed_data_manifests()
            .iter()
            .filter(|manifest| manifest.file().sequence_number > sequence_number)
            .collect();
        let mut partitions = BTreeMap::new();
        for manifests in written_after.chunks(MANIFESTS_PER_READ) {
            let parsed = self.read_manifests(manifests.iter().map(|manifest| manifest.file()))?;
            for (listed, manifest) in manifests.iter().zip(&parsed) {
                partitions.append(&mut file_index::list_partitions_added_after(
                    listed.file(),
                    manifest,
                    self.metadata(),
                    sequence_number,
                )?);
            }
        }
        Ok(partitions.into_values().collect())
    }

    /// The columns this version is sorted by, as the records of the files
    /// decoded with them share them.
    fn resolve_sort_columns(&self) -> Arc<SortColumns> {
        Arc::new(SortColumns::new(
            &self.metadata_file.metadata,
            self.columns(),
            &self.sort_by,
        ))
    }

    /// The live file at each of `paths` in the partition with the key
    /// `partition` as a commit records it, in the order given: `None` for
    /// a path this version has no file at.
    pub fn find_entries(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> Result<Vec<Option<DataFileEntry>>> {
        // A commit that only adds files looks for none, and names no
        // partition.
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let partition = self.find_keyed_partition(partition)?;
        let index = self.load_partition(partition)?;
        let found: Vec<Option<&Arc<IndexedFile>>> =
            paths.iter().map(|path| index.find(path)).collect();
        let live: Vec<&Arc<IndexedFile>> = found.iter().flatten().copied().collect();
        let mut manifest_paths = index.find_manifests_of(&live).into_iter();
        found
            .into_iter()
            .map(|file| {
                file.map(|file| {
                    let manifest_path = manifest_paths
                        .next()
                        .expect("a manifest is looked for of every live file");
                    file.to_entry(self.metadata(), self.columns(), manifest_path)
                })
                .transpose()
            })
            .collect()
    }

    /// `file`, a live file of this version, as the datastore's planners
    /// read it: its tuple, its row count, its bounds on the columns the
    /// table is sorted by, and those columns as what its rows are sorted by
    /// when its manifest entry records the sort order Pivot's writer
    /// produces for them.
    pub fn build_datastore_entry(&self, file: &IndexedFile) -> Result<FileEntry> {
        let is_in_sort_order =
            file.sort_order_id().is_some() && file.sort_order_id() == self.writer_sort_order_id;
        let sorted_by: Arc<[String]> = if is_in_sort_order {
            Arc::from(self.sort_by.as_slice())
        } else {
            Arc::default()
        };
        file.to_datastore_entry(self.metadata(), self.columns(), &self.sort_by, sorted_by)
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
    pub fn read_manifests<'a>(
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
    pub fn declared_schema(&self) -> ArrowSchemaRef {
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
