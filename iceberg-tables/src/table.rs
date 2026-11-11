//! One Iceberg table at one version: what every datastore reads a table
//! through and writes it through, given only the table's metadata
//! store.
//!
//! A table is loaded from its metadata document and its current snapshot's
//! manifest list, and that is all a version holds of its files: opening a
//! table decodes no manifest. The files are indexed a partition at a time
//! ([`FileIndex`]), when a scan, a commit or a maintenance pass first asks
//! for a partition, and the indexes of every table live in one cache
//! ([`FileMetadataCache`]), which keeps them within a budget of memory:
//! an index the cache let go of is built again when next asked for. The
//! same cache, within the same budget, keeps the footers a table's files
//! were found to have.
//!
//! # Which manifests hold a partition
//!
//! A manifest-list entry bounds each partition field's values across its
//! manifest, so it says which manifests may hold a partition and nothing
//! of which partitions a manifest holds. That is learned by decoding the
//! manifest, once, and kept beside the entry as its summary, which every
//! version listing the manifest shares.
//!
//! A partition's index is asked for by the partition. The manifests whose
//! entries cannot rule the partition's tuple out are the ones that may
//! hold a file of it; those not yet decoded are decoded, and then the
//! summaries name exactly the manifests that do. Those manifests are what
//! the partition's index at this version is built from, and an index is
//! this version's exactly when it was built from them: a manifest never
//! changes, so the list of them names the partition's files. The cached
//! index of an earlier or a later version is brought to this one by the
//! manifests the two lists differ in, and only those are decoded. A
//! version therefore reads its own files whatever the cache holds or let
//! go of, for as long as its manifests are in the store.
//!
//! A scan finds its partitions from the manifests its filters cannot rule
//! out by their entries, so it decodes none of the others.
//!
//! # Footers
//!
//! A version holds no file's footer either. Whatever needs row groups, a
//! scan's plan, a merge, a layout measurement, a description, asks for
//! those of the files it works on. A footer is read and parsed once, over
//! the worker pool, and the row groups that read produced are then kept in
//! the cache, each with the open of its file, where every version that
//! reads the file as this one does finds them and takes them as they are,
//! with nothing opened: one the cache let go of is read again. Row groups
//! are shared between the cache and everything that took them, so nothing
//! changes one: a plan records its version's deleted rows on copies.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use arrow_schema::{Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{FormatVersion, Manifest, ManifestFile, TableMetadata, TableMetadataRef};
use object_storage::{DataFile, FileRef, ObjectPath, load_objects};
use parquet_engine::file_entry::FileEntry;
use parquet_engine::{FileRowGroups, ParquetTable, ScanPlan, TableColumns, load_file_row_groups};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::expression::Expression;
use planner::types::physical_arrow_type;
use uuid::Uuid;

use crate::files::data_files::DataFileEntry;
use crate::files::deletes::{count_deleted_rows, read_deletion_vectors};
use crate::files::file_index::{
    DecodeContext, FileIndex, FileSelector, FilesByPartition, IndexedFile, Partition,
    PartitionInterner, SortKeyEncoder,
};
use crate::files::file_metadata_cache::{CacheScope, FileMetadataCache, VersionOrder};
use crate::files::manifests::{ListedManifest, ManifestList};
use crate::io::storage::WrittenObjects;
use crate::metadata::layout::TableLocation;
use crate::metadata::metadata_store::MetadataStore;
use crate::metadata::state::CurrentMetadata;
use crate::types::columns::{LOGICAL_TYPES_PROPERTY, identity_columns, read_table_columns};
use crate::types::schema::{UNSORTED_ORDER_ID, find_writer_sort_order_id};
use crate::{Error, Result};

/// How many manifests are read, and held parsed, at a time.
pub(crate) const MANIFESTS_PER_READ: usize = 16;

/// How many partitions a walk or a scan of many has indexed at a time.
const PARTITIONS_PER_LOAD: usize = 64;

/// How much of the files a pass decoded ahead of the partitions it will
/// index it keeps, by [`IndexedFile::heap_bytes`]; a manifest past it is
/// decoded again when its partitions' turn comes.
const DECODED_FILES_HELD_BYTES: u64 = 256 * 1024 * 1024;

/// Which declared columns may hold a NULL in the files a scan selected.
struct SelectedNulls {
    /// The field id of each declared column; `None` for a column the
    /// table's schema gives none.
    field_ids: Vec<Option<i32>>,
    /// The columns no file holds a NULL in.
    is_never_null: Vec<bool>,
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
            is_never_null: table.list_never_null_columns(),
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
        self.is_never_null
            .iter()
            .zip(self.may_hold_nulls)
            .map(|(is_never_null, may_hold_nulls)| {
                !is_never_null && (may_hold_nulls || !self.is_any_file_selected)
            })
            .collect()
    }
}

/// An Iceberg table at one committed version, with what reading and writing
/// it needs: its metadata store, where its files live and the
/// store they are read through, and the pool its reads run on. Immutable as
/// a version: a refresh, a commit or a local file swap hands back the next
/// version as a new value.
pub struct Table {
    /// The name the table is known by in its datastore.
    name: SchemaQualifiedTableName,
    metadata_store: Arc<dyn MetadataStore>,
    location: TableLocation,
    dispatcher: DataFlowDispatcher,
    current: Arc<CurrentMetadata>,
    /// The declared columns, keyed by the field ids the files carry.
    columns: TableColumns,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
    /// The id of the default sort order when rows ordered by `sort_by`, each
    /// column ascending with NULLs first, are in that order; `None` for an
    /// unsorted table and for an order Pivot's writer does not produce.
    writer_sort_order_id: Option<i32>,
    manifest_list: ManifestList,
    /// What this version shares with the versions it was derived from and
    /// those derived from it.
    lineage: Arc<Lineage>,
    /// For a local view of file swaps this process committed through
    /// another copy of the table: per partition, the files swapped out of
    /// the committed ones and those swapped in, which no manifest of this
    /// version records. The next refresh reconciles the copy to the
    /// committed metadata.
    swapped: Option<Arc<SwappedFiles>>,
}

/// What the versions of a table that read a file alike share: of one
/// schema, one partition spec and one sort order, each derived from
/// another by a refresh, a commit or a swap.
struct Lineage {
    /// Where the versions' partition indexes and the footers of their
    /// files are cached.
    scope: CacheScope,
    /// Every partition a manifest of some version was seen to hold.
    partitions: PartitionInterner,
    /// Held while partitions are indexed, so scans that want one partition
    /// at once wait for one build of its index rather than each decoding
    /// its manifests.
    indexing: Mutex<()>,
}

impl Lineage {
    fn new(cache: Arc<FileMetadataCache>) -> Self {
        Self {
            scope: CacheScope::new(cache),
            partitions: PartitionInterner::default(),
            indexing: Mutex::new(()),
        }
    }
}

/// What a scan over some filters selected of a table version.
pub(crate) struct SelectedFiles {
    /// The files, each located for reading, in the order they are scanned.
    pub(crate) files: Vec<DataFile>,
    /// The keys of the partitions the files were selected from.
    partitions: Vec<Arc<[u8]>>,
    /// Whether each declared column may hold a NULL in one of the files
    /// (see [`ScanPlan::nullability`]).
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
    dyn FnMut(&Arc<Partition>, &FileIndex) -> Result<ControlFlow<()>> + 'v;

/// A partition whose index a load has to build: where the partition sits
/// among those asked for, the manifests that hold a file of it, and the
/// index the cache held for it.
struct UnbuiltIndex {
    position: usize,
    holding: Vec<Arc<str>>,
    held: Option<Arc<FileIndex>>,
}

/// The files a local view swapped, by the key of their partition.
type SwappedFiles = HashMap<Arc<[u8]>, PartitionSwap>;

/// What a local view swapped in one partition.
#[derive(Clone)]
struct PartitionSwap {
    partition: Arc<Partition>,
    /// The committed files swapped out.
    removed: Vec<ObjectPath>,
    /// The files swapped in.
    added: Vec<Arc<IndexedFile>>,
}

/// The files of manifests decoded ahead of the partition indexes they go
/// into, by manifest path and partition key.
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
        let heap_bytes: u64 = files.values().flatten().map(|file| file.heap_bytes()).sum();
        if is_bounded && self.heap_bytes + heap_bytes > DECODED_FILES_HELD_BYTES {
            return;
        }
        self.heap_bytes += heap_bytes;
        self.files
            .entry(Arc::clone(manifest_path))
            .or_default()
            .extend(files);
    }

    /// Whether the files of the partition `key` in the manifest at
    /// `manifest_path` are held.
    fn holds(&self, manifest_path: &str, key: &[u8]) -> bool {
        self.files
            .get(manifest_path)
            .is_some_and(|partitions| partitions.contains_key(key))
    }

    /// Hand over the files of the partition `key` in the manifest at
    /// `manifest_path`.
    fn take(&mut self, manifest_path: &str, key: &[u8]) -> Option<Vec<Arc<IndexedFile>>> {
        let files = self.files.get_mut(manifest_path)?.remove(key)?;
        let heap_bytes: u64 = files.iter().map(|file| file.heap_bytes()).sum();
        self.heap_bytes -= heap_bytes;
        Some(files)
    }
}

impl Table {
    /// Load the table at the version `metadata_store` holds current: a cold
    /// open.
    pub fn load(
        dispatcher: &DataFlowDispatcher,
        metadata_store: Arc<dyn MetadataStore>,
        location: TableLocation,
        name: &SchemaQualifiedTableName,
    ) -> Result<Table> {
        let current = metadata_store.load_latest()?;
        Self::at(dispatcher, metadata_store, location, name, current, None)
    }

    /// The table at the version `current` describes. The manifest list is
    /// read over the pool when `manifest_list_size` is known, else fetched
    /// whole. No manifest is read.
    pub fn at(
        dispatcher: &DataFlowDispatcher,
        metadata_store: Arc<dyn MetadataStore>,
        location: TableLocation,
        name: &SchemaQualifiedTableName,
        current: CurrentMetadata,
        manifest_list_size: Option<u64>,
    ) -> Result<Table> {
        Self::at_version(
            dispatcher,
            metadata_store,
            location,
            name,
            current,
            manifest_list_size,
            None,
        )
    }

    /// The table at the version `current` describes, derived from
    /// `predecessor` when there is one.
    #[allow(clippy::too_many_arguments)] // the pieces of one version
    fn at_version(
        dispatcher: &DataFlowDispatcher,
        metadata_store: Arc<dyn MetadataStore>,
        location: TableLocation,
        name: &SchemaQualifiedTableName,
        current: CurrentMetadata,
        manifest_list_size: Option<u64>,
        predecessor: Option<&Table>,
    ) -> Result<Table> {
        let metadata = &current.metadata;
        let manifest_list = match metadata.current_snapshot() {
            None => ManifestList::empty(metadata),
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
                ManifestList::parse(&name.to_string(), metadata, &bytes)?
            }
        };
        Self::from_manifest_list(
            dispatcher,
            metadata_store,
            location,
            name,
            Arc::new(current),
            manifest_list,
            predecessor,
        )
    }

    /// The table at `current` with its manifest list already in hand,
    /// derived from `predecessor` when there is one: a version that reads a
    /// file as its predecessor does shares what the predecessor learned of
    /// the manifests both list, its partition indexes and the footers it
    /// parsed.
    #[allow(clippy::too_many_arguments)] // the pieces of one version, built in two places
    pub(crate) fn from_manifest_list(
        dispatcher: &DataFlowDispatcher,
        metadata_store: Arc<dyn MetadataStore>,
        location: TableLocation,
        name: &SchemaQualifiedTableName,
        current: Arc<CurrentMetadata>,
        mut manifest_list: ManifestList,
        predecessor: Option<&Table>,
    ) -> Result<Table> {
        let metadata = &current.metadata;
        if metadata.encryption_keys_iter().len() > 0 {
            return Err(Error::EncryptedTable {
                table: name.to_string(),
            });
        }
        let schema = metadata.current_schema();
        let columns = read_table_columns(&name.to_string(), metadata)?;
        let partition_by = identity_columns(
            schema,
            metadata
                .default_partition_spec()
                .fields()
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );
        let sort_by = identity_columns(
            schema,
            metadata
                .default_sort_order()
                .fields
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );
        let writer_sort_order_id = find_writer_sort_order_id(metadata, columns.columns());
        let lineage = match predecessor {
            Some(previous) if reads_files_alike(metadata, previous.metadata()) => {
                manifest_list.share_summaries_with(&previous.manifest_list);
                Arc::clone(&previous.lineage)
            }
            Some(previous) => Arc::new(Lineage::new(Arc::clone(previous.lineage.scope.cache()))),
            None => Arc::new(Lineage::new(FileMetadataCache::process_wide())),
        };
        Ok(Table {
            name: name.clone(),
            metadata_store,
            location,
            dispatcher: dispatcher.clone(),
            current,
            columns,
            partition_by,
            sort_by,
            writer_sort_order_id,
            manifest_list,
            lineage,
            swapped: None,
        })
    }

    /// The latest committed version, derived from this one, or `None` when
    /// this one is still current.
    pub fn refresh(&self) -> Result<Option<Table>> {
        let latest = self.metadata_store.load_latest()?;
        if latest.metadata_location == self.current.metadata_location {
            return Ok(None);
        }
        Self::at_version(
            &self.dispatcher,
            Arc::clone(&self.metadata_store),
            self.location.clone(),
            &self.name,
            latest,
            None,
            Some(self),
        )
        .map(Some)
    }

    /// This version with `removed`, files of the partition with the key
    /// `removed_partition`, dropped and `added` swapped in, as a local view
    /// of a swap this process already committed through another copy. A
    /// swap rearranges the table's rows, so what the version counts
    /// stands.
    pub fn with_files_swapped(
        &self,
        removed_partition: &[u8],
        removed: &[ObjectPath],
        added: Vec<DataFileEntry>,
    ) -> Result<Table> {
        let mut swapped = self.swapped.as_deref().cloned().unwrap_or_default();
        if !removed.is_empty() {
            let partition =
                self.find_partition(removed_partition)?
                    .ok_or_else(|| Error::UnknownPartition {
                        table: self.name.to_string(),
                    })?;
            let swap = swapped
                .entry(Arc::clone(partition.key()))
                .or_insert_with(|| PartitionSwap {
                    partition,
                    removed: Vec::new(),
                    added: Vec::new(),
                });
            for path in removed {
                let swapped_in = swap.added.iter().position(|file| &file.file.path == path);
                match swapped_in {
                    Some(position) => {
                        swap.added.remove(position);
                    }
                    None => swap.removed.push(path.clone()),
                }
            }
        }
        let encoder = self.sort_key_encoder()?;
        for entry in &added {
            let file = Arc::new(IndexedFile::uncommitted(
                entry,
                &encoder,
                self.metadata(),
                &self.location,
                &self.lineage.partitions,
            )?);
            let swap = swapped
                .entry(Arc::clone(file.partition.key()))
                .or_insert_with(|| PartitionSwap {
                    partition: Arc::clone(&file.partition),
                    removed: Vec::new(),
                    added: Vec::new(),
                });
            swap.added
                .retain(|earlier| earlier.file.path != file.file.path);
            swap.added.push(file);
        }
        Ok(Table {
            name: self.name.clone(),
            metadata_store: Arc::clone(&self.metadata_store),
            location: self.location.clone(),
            dispatcher: self.dispatcher.clone(),
            current: Arc::clone(&self.current),
            columns: self.columns.clone(),
            partition_by: self.partition_by.clone(),
            sort_by: self.sort_by.clone(),
            writer_sort_order_id: self.writer_sort_order_id,
            manifest_list: self.manifest_list.clone(),
            lineage: Arc::clone(&self.lineage),
            swapped: Some(Arc::new(swapped)),
        })
    }

    /// Derive this version, just loaded, from `previous`, an earlier version
    /// of this table: it takes over what `previous` learned of the
    /// manifests both list, its partition indexes, which this version's
    /// are then built on by the manifests that changed, and the footers it
    /// parsed. Nothing carries over between versions that do not read a
    /// file alike: of one schema read as the same columns, one partition
    /// spec and one sort order.
    pub fn carry_over_from(&mut self, previous: &Table) {
        if reads_files_alike(self.metadata(), previous.metadata()) {
            self.manifest_list
                .share_summaries_with(&previous.manifest_list);
            self.lineage = Arc::clone(&previous.lineage);
        }
    }

    /// What a scan over `filters` reads: the row groups of the files
    /// [`select_scan`](Self::select_scan) selects, in its order, with the
    /// rows their deletion vectors delete marked on them, and whether each
    /// column may hold a NULL in those files. The order is fixed by the
    /// version, and with file-local row-group order makes the same filters
    /// yield the same row-group positions for scanning and
    /// materialization. Of the delete manifests, those the partition
    /// summaries of their manifest-list entries cannot rule out are read.
    ///
    /// The footers are taken from the cache, where those read are kept: a
    /// file whose footer is held and which has no deleted row is planned
    /// from the row groups the cache holds, as they are, and a file with
    /// deleted rows from copies of those that hold one. Once the footers
    /// are taken, the indexes the files were selected from count as used
    /// again, so the cache lets go of a partition's footers before its
    /// index.
    pub fn plan_scan(&self, filters: &[Expression]) -> Result<ScanPlan> {
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: self.name.to_string(),
            source: Box::new(source),
        };
        let SelectedFiles {
            files,
            partitions,
            nullability,
        } = self.select_scan(filters)?;
        let selected_deletes = self
            .manifest_list
            .select_delete_manifests(filters)
            .map_err(invalid_partition_metadata)?;
        let delete_manifests = self.read_manifests(selected_deletes)?;
        let deletion_vectors =
            read_deletion_vectors(delete_manifests, &files, &self.location, &self.dispatcher)?;
        let mut footers = self.open_footers(&files, FooterUse::Kept)?;
        self.lineage.scope.mark_indexes_used(&partitions);
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
        Ok(ScanPlan {
            parquet: Arc::new(parquet),
            nullability,
        })
    }

    /// What a scan over `filters` selects: the files it reads, each located
    /// for reading, in the order
    /// [`visit_selected_files`](Self::visit_selected_files) visits them,
    /// the partitions they were selected from, and whether each declared
    /// column may hold a NULL in one of them.
    pub(crate) fn select_scan(&self, filters: &[Expression]) -> Result<SelectedFiles> {
        let mut files = Vec::new();
        let mut partitions: Vec<Arc<[u8]>> = Vec::new();
        let mut nulls = SelectedNulls::new(self);
        self.visit_selected_files(filters, &mut |file| {
            nulls.observe(file);
            // Files are visited a partition at a time.
            if partitions.last() != Some(file.partition.key()) {
                partitions.push(Arc::clone(file.partition.key()));
            }
            files.push(
                self.location
                    .data_file_at(&file.file.path, file.file.size)?,
            );
            Ok(())
        })?;
        Ok(SelectedFiles {
            files,
            partitions,
            nullability: nulls.into_nullability(),
        })
    }

    /// Visit the files a scan over `filters` reads: those that the files'
    /// partition values and column bounds cannot rule out, by partition,
    /// then by where a file starts on the sort key, then by path.
    ///
    /// Only the partitions the filters keep are indexed, found in the
    /// manifests whose manifest-list entries the filters cannot rule out;
    /// no other data manifest is read.
    fn visit_selected_files(
        &self,
        filters: &[Expression],
        visit: &mut dyn FnMut(&IndexedFile) -> Result<()>,
    ) -> Result<()> {
        let selector = FileSelector::new(&self.current.metadata, filters).map_err(|source| {
            Error::InvalidPartitionMetadata {
                table: self.name.to_string(),
                source: Box::new(source),
            }
        })?;
        let mut file_count = 0_usize;
        let mut decoded = DecodedFiles::default();
        let partitions = self.select_partitions(&selector, &mut decoded)?;
        for partitions in partitions.chunks(PARTITIONS_PER_LOAD) {
            for index in self.load_partitions(partitions, &mut decoded)? {
                for file in selector.select(&index) {
                    visit(file)?;
                    file_count += 1;
                }
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
    /// order: those of the data manifests whose manifest-list entries the
    /// filters cannot rule out, and those a local swap put a file in. A
    /// manifest decoded to learn its partitions leaves the files of the
    /// kept ones in `decoded`.
    fn select_partitions(
        &self,
        selector: &FileSelector,
        decoded: &mut DecodedFiles,
    ) -> Result<Vec<Arc<Partition>>> {
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
        self.summarize_manifests(&manifests, &mut keeps, decoded)?;
        let mut partitions = BTreeMap::new();
        let summarized = manifests
            .iter()
            .filter_map(|manifest| manifest.summary())
            .flat_map(|summary| summary.partitions())
            .map(|(partition, _)| partition);
        let swapped = self
            .swapped
            .iter()
            .flat_map(|swapped| swapped.values())
            .map(|swap| &swap.partition);
        for partition in summarized.chain(swapped) {
            if !partitions.contains_key(partition.key()) && keeps(partition)? {
                partitions.insert(Arc::clone(partition.key()), Arc::clone(partition));
            }
        }
        Ok(partitions.into_values().collect())
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
        let scope = &self.lineage.scope;
        let paths = files.iter().map(|file| file.file.path.as_str());
        let held = match footer_use {
            FooterUse::Kept => scope.find_footers(paths),
            FooterUse::Unrecorded => scope.peek_footers(paths),
        };
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
            scope.store_footers(kept);
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
        self.lineage
            .scope
            .remove_footers(uris.iter().map(String::as_str));
        Ok(())
    }

    /// Every partition this version has a file in, in key order. The data
    /// manifests nothing decoded yet are read for the partitions they
    /// hold.
    pub fn partitions(&self) -> Result<Vec<Arc<Partition>>> {
        self.list_partitions(&mut |_| Ok(false), &mut DecodedFiles::default())
    }

    /// Every partition this version has a file in, in key order. A manifest
    /// decoded to learn its partitions leaves the files of those `keeps`
    /// accepts in `decoded`.
    fn list_partitions(
        &self,
        keeps: &mut dyn FnMut(&Partition) -> Result<bool>,
        decoded: &mut DecodedFiles,
    ) -> Result<Vec<Arc<Partition>>> {
        let manifests: Vec<&ListedManifest> =
            self.manifest_list.listed_data_manifests().iter().collect();
        self.summarize_manifests(&manifests, keeps, decoded)?;
        let mut partitions = BTreeMap::new();
        let summarized = manifests
            .iter()
            .filter_map(|manifest| manifest.summary())
            .flat_map(|summary| summary.partitions())
            .map(|(partition, _)| partition);
        let swapped = self
            .swapped
            .iter()
            .flat_map(|swapped| swapped.values())
            .map(|swap| &swap.partition);
        for partition in summarized.chain(swapped) {
            partitions
                .entry(Arc::clone(partition.key()))
                .or_insert_with(|| Arc::clone(partition));
        }
        Ok(partitions.into_values().collect())
    }

    /// The partition with the key `key`, or `None` when no manifest of the
    /// table was seen to hold it. A key no decode has met yet is looked for
    /// in the manifests nothing decoded.
    pub fn find_partition(&self, key: &[u8]) -> Result<Option<Arc<Partition>>> {
        if let Some(partition) = self.lineage.partitions.find(key) {
            return Ok(Some(partition));
        }
        let manifests: Vec<&ListedManifest> =
            self.manifest_list.listed_data_manifests().iter().collect();
        self.summarize_manifests(&manifests, &mut |_| Ok(false), &mut DecodedFiles::default())?;
        Ok(self.lineage.partitions.find(key))
    }

    /// The files this version reads in `partition`: those the current
    /// snapshot's data manifests list, or a local swap's view of them. Built
    /// from the manifests that hold a file of the partition, on the index
    /// the cache holds for it when there is one.
    pub fn partition_files(&self, partition: &Arc<Partition>) -> Result<Arc<FileIndex>> {
        let mut indexes = self.load_partitions(
            std::slice::from_ref(partition),
            &mut DecodedFiles::default(),
        )?;
        Ok(indexes
            .pop()
            .expect("one index is loaded for each partition asked for"))
    }

    /// Visit every partition this version has a file in with its files, in
    /// key order, a few partitions indexed at a time; `Break` ends the
    /// walk.
    pub fn visit_partitions(&self, visit: &mut VisitPartition<'_>) -> Result<()> {
        let mut decoded = DecodedFiles::default();
        let partitions = self.list_partitions(&mut |_| Ok(true), &mut decoded)?;
        for partitions in partitions.chunks(PARTITIONS_PER_LOAD) {
            let indexes = self.load_partitions(partitions, &mut decoded)?;
            for (partition, index) in partitions.iter().zip(&indexes) {
                if visit(partition, index)?.is_break() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Decode those of `manifests` nothing decoded yet, each kept
    /// summarized beside its manifest-list entry. The files of the
    /// partitions `keeps` accepts are left in `decoded`, as far as its
    /// bound allows.
    fn summarize_manifests(
        &self,
        manifests: &[&ListedManifest],
        keeps: &mut dyn FnMut(&Partition) -> Result<bool>,
        decoded: &mut DecodedFiles,
    ) -> Result<()> {
        let unread: Vec<&ListedManifest> = manifests
            .iter()
            .filter(|manifest| manifest.summary().is_none())
            .copied()
            .collect();
        self.decode_manifests(&unread, keeps, decoded, true)
    }

    /// Read `manifests`, data manifests of this version, over the pool, a
    /// few at a time, and decode each: its summary is kept beside its
    /// manifest-list entry, and the files of the partitions `keeps`
    /// accepts are left in `decoded`, within its bound when `is_bounded`.
    fn decode_manifests(
        &self,
        manifests: &[&ListedManifest],
        keeps: &mut dyn FnMut(&Partition) -> Result<bool>,
        decoded: &mut DecodedFiles,
        is_bounded: bool,
    ) -> Result<()> {
        for manifests in manifests.chunks(MANIFESTS_PER_READ) {
            let parsed = self.read_manifests(manifests.iter().map(|manifest| manifest.file()))?;
            for (listed, manifest) in manifests.iter().zip(parsed) {
                self.decode_manifest(listed, manifest, keeps, decoded, is_bounded)?;
            }
        }
        Ok(())
    }

    /// Decode `manifest`, the parsed data manifest `listed` names: see
    /// [`decode_manifests`](Self::decode_manifests).
    fn decode_manifest(
        &self,
        listed: &ListedManifest,
        manifest: Manifest,
        keeps: &mut dyn FnMut(&Partition) -> Result<bool>,
        decoded: &mut DecodedFiles,
        is_bounded: bool,
    ) -> Result<()> {
        let table_name = self.name.to_string();
        let encoder = self.sort_key_encoder()?;
        let context = DecodeContext {
            table: &table_name,
            encoder: &encoder,
            metadata: self.metadata(),
            location: &self.location,
            partitions: &self.lineage.partitions,
        };
        let manifest = IndexedFile::decode_manifest(listed.file(), manifest, &context, keeps)?;
        listed.learn(manifest.summary);
        decoded.keep(listed.path(), manifest.files, is_bounded);
        Ok(())
    }

    /// The index of each of `partitions` at this version, in the order
    /// given, with a local swap's files swapped.
    ///
    /// A partition's index is built from the data manifests that hold a
    /// file of it. The manifests whose manifest-list entries cannot rule
    /// the partition out and that nothing decoded yet are decoded first, so
    /// the summaries name every manifest that holds it. The index the cache
    /// holds for the partition is this version's when it was built from
    /// exactly those manifests; otherwise this version's is built on it by
    /// the manifests the two differ in, taken from `decoded` or decoded
    /// now, and cached.
    fn load_partitions(
        &self,
        partitions: &[Arc<Partition>],
        decoded: &mut DecodedFiles,
    ) -> Result<Vec<Arc<FileIndex>>> {
        let _indexing = self.lineage.indexing.lock().unwrap();
        let listed = self.manifest_list.listed_data_manifests();
        let wanted: HashSet<&[u8]> = partitions
            .iter()
            .map(|partition| partition.key().as_ref())
            .collect();
        let mut is_wanted = |partition: &Partition| Ok(wanted.contains(partition.key().as_ref()));
        let undecoded: Vec<&ListedManifest> = listed
            .iter()
            .filter(|manifest| {
                manifest.summary().is_none()
                    && partitions
                        .iter()
                        .any(|partition| manifest.may_hold_partition(partition))
            })
            .collect();
        self.decode_manifests(&undecoded, &mut is_wanted, decoded, false)?;

        let mut indexes: Vec<Option<Arc<FileIndex>>> = Vec::with_capacity(partitions.len());
        let mut unbuilt: Vec<UnbuiltIndex> = Vec::new();
        for (position, partition) in partitions.iter().enumerate() {
            let holding: Vec<Arc<str>> = listed
                .iter()
                .filter(|manifest| {
                    manifest
                        .summary()
                        .is_some_and(|summary| summary.holds_partition(partition.key()))
                })
                .map(|manifest| Arc::clone(manifest.path()))
                .collect();
            let held = self.lineage.scope.find_index(partition.key());
            match held {
                Some(index) if index.is_built_from(&holding) => indexes.push(Some(index)),
                held => {
                    indexes.push(None);
                    unbuilt.push(UnbuiltIndex {
                        position,
                        holding,
                        held,
                    });
                }
            }
        }

        let mut missing: HashSet<&str> = HashSet::new();
        for partition in &unbuilt {
            let key = partitions[partition.position].key();
            for manifest in &partition.holding {
                let is_held = partition
                    .held
                    .as_ref()
                    .is_some_and(|index| index.holds_manifest(manifest));
                if !is_held && !decoded.holds(manifest, key) {
                    missing.insert(manifest);
                }
            }
        }
        if !missing.is_empty() {
            let unbuilt_keys: HashSet<&[u8]> = unbuilt
                .iter()
                .map(|partition| partitions[partition.position].key().as_ref())
                .collect();
            let mut is_unbuilt =
                |partition: &Partition| Ok(unbuilt_keys.contains(partition.key().as_ref()));
            let missing: Vec<&ListedManifest> = listed
                .iter()
                .filter(|manifest| missing.contains(manifest.path().as_ref()))
                .collect();
            self.decode_manifests(&missing, &mut is_unbuilt, decoded, false)?;
        }

        let built_at = VersionOrder {
            last_sequence_number: self.metadata().last_sequence_number(),
            last_updated_ms: self.metadata().last_updated_ms(),
        };
        for partition in unbuilt {
            let key = partitions[partition.position].key();
            let index = Arc::new(FileIndex::advance(
                partition.held.as_deref(),
                &partition.holding,
                |manifest| {
                    decoded
                        .take(manifest, key)
                        .expect("the partition's files in the manifest were just decoded")
                },
            ));
            self.lineage.scope.store_index(key, &index, built_at);
            indexes[partition.position] = Some(index);
        }
        Ok(partitions
            .iter()
            .zip(indexes)
            .map(|(partition, index)| {
                let index = index.expect("every partition asked for is indexed");
                self.swap_files(partition, index)
            })
            .collect())
    }

    /// `index`, the committed files of `partition`, with what a local swap
    /// swapped in the partition.
    fn swap_files(&self, partition: &Partition, index: Arc<FileIndex>) -> Arc<FileIndex> {
        let swap = self
            .swapped
            .as_ref()
            .and_then(|swapped| swapped.get(partition.key()));
        let Some(swap) = swap else {
            return index;
        };
        let mut view = FileIndex::clone(&index);
        for path in &swap.removed {
            view.remove(path);
        }
        for file in &swap.added {
            view.insert(Arc::clone(file));
        }
        Arc::new(view)
    }

    /// Learn the data manifests of this version that a commit just wrote,
    /// from the bytes it `written`: each is decoded with nothing read, and
    /// the partition indexes the cache holds are brought to this version by
    /// them, so the partitions a writer keeps appending to stay indexed.
    pub(crate) fn learn_written_manifests(&self, written: &WrittenObjects) -> Result<()> {
        let unread: Vec<(&ListedManifest, Manifest)> = self
            .manifest_list
            .listed_data_manifests()
            .iter()
            .filter(|manifest| manifest.summary().is_none())
            .filter_map(|manifest| {
                let bytes = written.find(&manifest.file().manifest_path)?;
                Some(
                    self.parse_manifest(&manifest.file().manifest_path, &bytes)
                        .map(|parsed| (manifest, parsed)),
                )
            })
            .collect::<Result<_>>()?;
        self.learn_manifests(unread)
    }

    /// Decode `manifests`, data manifests of this version nothing decoded
    /// yet, and bring the partition indexes the cache holds to this version
    /// by them.
    fn learn_manifests(&self, manifests: Vec<(&ListedManifest, Manifest)>) -> Result<()> {
        let indexed: HashSet<Arc<[u8]>> = self
            .lineage
            .scope
            .list_indexed_partitions()
            .into_iter()
            .collect();
        let mut is_indexed = |partition: &Partition| Ok(indexed.contains(partition.key()));
        let mut decoded = DecodedFiles::default();
        let mut stale = BTreeMap::new();
        for (listed, manifest) in manifests {
            self.decode_manifest(listed, manifest, &mut is_indexed, &mut decoded, false)?;
            let summary = listed
                .summary()
                .expect("a decoded manifest is kept summarized");
            for (partition, _) in summary.partitions() {
                if indexed.contains(partition.key()) {
                    stale.insert(Arc::clone(partition.key()), Arc::clone(partition));
                }
            }
        }
        let stale: Vec<Arc<Partition>> = stale.into_values().collect();
        for partitions in stale.chunks(PARTITIONS_PER_LOAD) {
            self.load_partitions(partitions, &mut decoded)?;
        }
        Ok(())
    }

    /// The partitions a file was added to since `previous`, an earlier
    /// version of this table: those holding, in a data manifest this
    /// version lists and `previous` did not, a file committed after
    /// `previous`. The manifests that joined the list and nothing decoded
    /// yet are read. A version 1 table numbers no commit, so every
    /// partition of a manifest that joined counts.
    pub fn list_partitions_added_since(&self, previous: &Table) -> Result<Vec<Arc<Partition>>> {
        let earlier: HashSet<&str> = previous
            .manifest_list
            .data_manifests()
            .map(|manifest| manifest.manifest_path.as_str())
            .collect();
        let joined: Vec<&ListedManifest> = self
            .manifest_list
            .listed_data_manifests()
            .iter()
            .filter(|manifest| !earlier.contains(manifest.path().as_ref()))
            .collect();
        let unread: Vec<&ListedManifest> = joined
            .iter()
            .filter(|manifest| manifest.summary().is_none())
            .copied()
            .collect();
        for manifests in unread.chunks(MANIFESTS_PER_READ) {
            let parsed = self.read_manifests(manifests.iter().map(|manifest| manifest.file()))?;
            self.learn_manifests(manifests.iter().copied().zip(parsed).collect())?;
        }
        let numbers_commits = self.metadata().format_version() != FormatVersion::V1;
        let last_commit = previous.metadata().last_sequence_number();
        let mut added = BTreeMap::new();
        for manifest in joined {
            let summary = manifest
                .summary()
                .expect("a manifest that joined the list was just decoded");
            for (partition, newest_commit) in summary.partitions() {
                if !numbers_commits || *newest_commit > last_commit {
                    added.insert(Arc::clone(partition.key()), Arc::clone(partition));
                }
            }
        }
        Ok(added.into_values().collect())
    }

    fn sort_key_encoder(&self) -> Result<SortKeyEncoder> {
        SortKeyEncoder::new(&self.current.metadata, self.columns(), &self.sort_by)
    }

    /// The live file at `path` in the partition with the key `partition`,
    /// or `None` when this version has no such file.
    fn find_file(&self, partition: &[u8], path: &ObjectPath) -> Result<Option<Arc<IndexedFile>>> {
        let Some(partition) = self.find_partition(partition)? else {
            return Ok(None);
        };
        Ok(self.partition_files(&partition)?.find(path).cloned())
    }

    /// The live file at `path` in the partition with the key `partition`
    /// as a commit records it, or `None` when this version has no such
    /// file.
    pub fn find_entry(&self, partition: &[u8], path: &ObjectPath) -> Result<Option<DataFileEntry>> {
        self.find_file(partition, path)?
            .map(|file| file.to_entry(self.metadata(), self.columns()))
            .transpose()
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
    /// is read. `None` when the manifest list does not count them, or when
    /// a local swap changed the files without a manifest list of its own.
    pub fn count_data_files(&self) -> Option<u64> {
        if self.swapped.is_some() {
            return None;
        }
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
        if self.swapped.is_some() {
            return None;
        }
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

    /// Whether each declared column holds a NULL in no file of the table,
    /// read from the metadata alone: a column the table's current schema
    /// has as a required field, which Iceberg lets no file hold a NULL in.
    pub fn list_never_null_columns(&self) -> Vec<bool> {
        let schema = self.current.metadata.current_schema();
        self.columns()
            .iter()
            .map(|column| {
                schema
                    .field_by_name(&column.name)
                    .is_some_and(|field| field.required)
            })
            .collect()
    }

    /// Whether each declared column may hold a NULL, as the table's schema
    /// declares it.
    pub fn schema_nullability(&self) -> Vec<bool> {
        self.current
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

    /// The metadata document this version was loaded from.
    pub fn current(&self) -> &Arc<CurrentMetadata> {
        &self.current
    }

    pub fn metadata(&self) -> &TableMetadataRef {
        &self.current.metadata
    }

    /// The metadata file this version was loaded from. Every commit writes a
    /// new one, so it is the table's version.
    pub fn metadata_location(&self) -> &str {
        &self.current.metadata_location
    }

    /// The table's identity across every metadata version.
    pub fn uuid(&self) -> Uuid {
        self.current.metadata.uuid()
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

    /// The identity-partitioned columns, in spec order.
    pub fn partition_by(&self) -> &[String] {
        &self.partition_by
    }

    /// The identity-sorted columns, in sort-order order.
    pub fn sort_by(&self) -> &[String] {
        &self.sort_by
    }

    pub fn manifest_list(&self) -> &ManifestList {
        &self.manifest_list
    }
}

/// Whether a version of a table described by `metadata` reads a file as
/// one described by `earlier` does: under one schema, whose fields are
/// read as the same Pivot types, one partition spec and one sort order.
fn reads_files_alike(metadata: &TableMetadata, earlier: &TableMetadata) -> bool {
    metadata.current_schema_id() == earlier.current_schema_id()
        && metadata.properties().get(LOGICAL_TYPES_PROPERTY)
            == earlier.properties().get(LOGICAL_TYPES_PROPERTY)
        && metadata.default_partition_spec_id() == earlier.default_partition_spec_id()
        && metadata.default_sort_order_id() == earlier.default_sort_order_id()
}

#[cfg(test)]
mod tests;
