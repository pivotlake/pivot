//! One Iceberg table at one version: what every datastore reads a table
//! through and writes it through, given only the catalog that holds the
//! table's current metadata pointer.
//!
//! A table is loaded from its metadata document and its current snapshot's
//! manifest list. Its manifests are decoded when something first needs them,
//! a scan over some filters only the ones the list's partition summaries
//! cannot rule out, and a commit, a compaction or a vacuum all of them, once,
//! for the life of the version. Footers are read over the worker pool when a
//! scan first needs them and remembered the same way.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arrow_schema::{Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{
    Manifest, ManifestContentType, ManifestEntry, ManifestFile, ManifestMetadata, ManifestStatus,
    TableMetadataRef,
};
use object_storage::{FileRef, ObjectPath, load_objects};
use parquet_engine::{
    FileRowGroups, ParquetTable, RowGroupMetadata, TableColumns, load_file_row_groups,
};
use planner::catalog::Column;
use planner::expression::Expression;
use planner::types::physical_arrow_type;
use uuid::Uuid;

use crate::catalog::TableCatalog;
use crate::columns::{read_logical_types, table_columns};
use crate::commit::{self, TableCreation};
use crate::data_files::{DataFileEntry, EntryContext, to_data_file};
use crate::deletes::{count_deleted_rows, read_deletion_vectors};
use crate::layout::TableLocation;
use crate::load::{decode_manifests, identity_columns, read_metadata_object};
use crate::manifest_writer::LoadedManifest;
use crate::manifests::{DataFiles, DecodedManifest, FileDescriptor, ManifestList, read_manifests};
use crate::reachability::{self, ReachableObjects};
use crate::state::CurrentMetadata;
use crate::store::TableStore;
use crate::{Error, Result};

/// The footers read so far for one version's files, by the file's URI.
type Footers = HashMap<String, Vec<Arc<RowGroupMetadata>>>;

/// An Iceberg table at one committed version, with what reading and writing
/// it needs: the catalog that holds its current-metadata pointer, where its
/// files live and the store they are read through, and the pool its reads
/// run on. Immutable as a version: a refresh, a commit or a local file swap
/// hands back the next version as a new value.
pub struct Table {
    /// The qualified name the table is known by, for messages.
    name: String,
    catalog: Arc<dyn TableCatalog>,
    location: TableLocation,
    store: TableStore,
    dispatcher: DataFlowDispatcher,
    current: Arc<CurrentMetadata>,
    /// The declared columns, keyed by the field ids the files carry.
    columns: TableColumns,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
    manifest_list: ManifestList,
    /// The size of the manifest list object, which a pool read of it needs;
    /// `None` for a table with no current snapshot.
    manifest_list_size: Option<u64>,
    /// Every data manifest of the current snapshot, decoded, once something
    /// needed all of them.
    decoded: Mutex<Option<Arc<Vec<Arc<DecodedManifest>>>>>,
    /// Files this process committed through another copy of the table and
    /// swapped into this one: part of every read until the next refresh
    /// reconciles the copy to the committed metadata, recorded in no manifest
    /// of this version.
    uncommitted: Vec<DataFileEntry>,
    footers: Mutex<Footers>,
}

impl Table {
    /// Create the table at `location` through `catalog`: publish its first
    /// version, holding the adopted files as the first snapshot when there
    /// are any.
    #[allow(clippy::too_many_arguments)] // one call per table creation; each piece is needed
    pub fn create(
        dispatcher: &DataFlowDispatcher,
        catalog: Arc<dyn TableCatalog>,
        location: TableLocation,
        store: TableStore,
        name: &str,
        table_uuid: Uuid,
        creation: TableCreation<'_>,
        now_ms: i64,
    ) -> Result<Table> {
        commit::initialize_table(
            dispatcher, catalog, location, store, name, table_uuid, creation, now_ms,
        )
    }

    /// Load the table at the version `catalog` holds current: a cold open.
    pub fn load(
        dispatcher: &DataFlowDispatcher,
        catalog: Arc<dyn TableCatalog>,
        location: TableLocation,
        store: TableStore,
        name: &str,
    ) -> Result<Table> {
        let current = catalog.load()?;
        Self::at(dispatcher, catalog, location, store, name, current, None)
    }

    /// The table at the version `current` describes. The manifest list is
    /// read over the pool when `manifest_list_size` is known, else fetched
    /// whole.
    pub fn at(
        dispatcher: &DataFlowDispatcher,
        catalog: Arc<dyn TableCatalog>,
        location: TableLocation,
        store: TableStore,
        name: &str,
        current: CurrentMetadata,
        manifest_list_size: Option<u64>,
    ) -> Result<Table> {
        let metadata = &current.metadata;
        let (manifest_list, manifest_list_size) = match metadata.current_snapshot() {
            None => (ManifestList::empty(metadata), None),
            Some(snapshot) => {
                let path = snapshot.manifest_list();
                let bytes = match manifest_list_size {
                    Some(size) => {
                        load_objects(dispatcher, &[store.data_file(path, size)?])?
                            .pop()
                            .expect("load_objects returns one object per file")
                            .bytes
                    }
                    None => read_metadata_object(&location, name, path)?,
                };
                let size = bytes.len() as u64;
                (ManifestList::parse(name, metadata, &bytes)?, Some(size))
            }
        };
        Self::assemble(
            dispatcher,
            catalog,
            location,
            store,
            name,
            Arc::new(current),
            manifest_list,
            manifest_list_size,
            None,
        )
    }

    /// The table at `current` with its manifest list already in hand, and its
    /// manifests when `decoded`: what a commit hands back without rereading
    /// what it wrote.
    #[allow(clippy::too_many_arguments)] // the pieces of one version, built in two places
    pub(crate) fn assemble(
        dispatcher: &DataFlowDispatcher,
        catalog: Arc<dyn TableCatalog>,
        location: TableLocation,
        store: TableStore,
        name: &str,
        current: Arc<CurrentMetadata>,
        manifest_list: ManifestList,
        manifest_list_size: Option<u64>,
        decoded: Option<Vec<Arc<DecodedManifest>>>,
    ) -> Result<Table> {
        let metadata = &current.metadata;
        if metadata.encryption_keys_iter().len() > 0 {
            return Err(Error::EncryptedTable {
                table: name.to_string(),
            });
        }
        let schema = metadata.current_schema();
        let logical_types = read_logical_types(metadata.properties())?;
        let columns = table_columns(name, schema, &logical_types)?;
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
        Ok(Table {
            name: name.to_string(),
            catalog,
            location,
            store,
            dispatcher: dispatcher.clone(),
            current,
            columns,
            partition_by,
            sort_by,
            manifest_list,
            manifest_list_size,
            decoded: Mutex::new(decoded.map(Arc::new)),
            uncommitted: Vec::new(),
            footers: Mutex::new(Footers::new()),
        })
    }

    /// The latest committed version, or `None` when this one is still
    /// current. The new version starts with no footers; see
    /// [`carry_footers_from`](Self::carry_footers_from).
    pub fn refresh(&self) -> Result<Option<Table>> {
        let Some(current) = self.catalog.refresh(&self.current)? else {
            return Ok(None);
        };
        Self::at(
            &self.dispatcher,
            Arc::clone(&self.catalog),
            self.location.clone(),
            self.store.clone(),
            &self.name,
            current,
            None,
        )
        .map(Some)
    }

    /// Commit, on top of this version, the removal of `removed` (entries of
    /// this version) and the addition of `added`, as one new version dated
    /// `now_ms`. `data_change` is `false` for a rearrangement (compaction)
    /// that leaves the table's rows as they were. `None` means another writer
    /// landed a version first: the caller refreshes and retries.
    pub fn commit(
        &self,
        removed: &[DataFileEntry],
        added: Vec<DataFileEntry>,
        data_change: bool,
        now_ms: i64,
    ) -> Result<Option<Table>> {
        commit::commit_file_changes(self, removed, added, data_change, now_ms)
    }

    /// Drop every snapshot that has outlived the table's retention window as
    /// of `now_ms`, as one metadata-only commit with no change to the current
    /// snapshot. `None` when nothing has expired, or when another writer
    /// landed the next version first (that writer expired them itself).
    pub fn expire_snapshots(&self, now_ms: i64) -> Result<Option<Table>> {
        commit::expire_snapshots(self, now_ms)
    }

    /// The data files some retained snapshot names, and the metadata objects
    /// the retained snapshots and the metadata log still reach: what a sweep
    /// of the table's storage must keep.
    pub fn reachable_objects(&self) -> Result<ReachableObjects> {
        reachability::reachable_objects(self)
    }

    /// This version with `removed` dropped and `added` swapped in, as a local
    /// view of a swap this process already committed through another copy.
    /// The footers held carry over.
    pub fn with_files_swapped(
        &self,
        removed: &[ObjectPath],
        added: Vec<DataFileEntry>,
    ) -> Result<Table> {
        let removed_uris: HashSet<String> = removed
            .iter()
            .map(|path| self.location.uri_for(path))
            .collect::<Result<_>>()?;
        let decoded = self
            .decoded_manifests()?
            .iter()
            .map(|manifest| Arc::new(manifest.without_files(removed, &removed_uris)))
            .collect();
        let mut uncommitted: Vec<DataFileEntry> = self
            .uncommitted
            .iter()
            .filter(|entry| !removed.contains(&entry.file.path))
            .cloned()
            .collect();
        uncommitted.extend(added);
        let footers = self
            .footers
            .lock()
            .unwrap()
            .iter()
            .filter(|(uri, _)| !removed_uris.contains(*uri))
            .map(|(uri, row_groups)| (uri.clone(), row_groups.clone()))
            .collect();
        Ok(Table {
            name: self.name.clone(),
            catalog: Arc::clone(&self.catalog),
            location: self.location.clone(),
            store: self.store.clone(),
            dispatcher: self.dispatcher.clone(),
            current: Arc::clone(&self.current),
            columns: self.columns.clone(),
            partition_by: self.partition_by.clone(),
            sort_by: self.sort_by.clone(),
            manifest_list: self.manifest_list.clone(),
            manifest_list_size: self.manifest_list_size,
            decoded: Mutex::new(Some(Arc::new(decoded))),
            uncommitted,
            footers: Mutex::new(footers),
        })
    }

    /// Keep the footers `previous` holds for the files this version still
    /// has, so a scan after a refresh or a commit does not read them back.
    pub fn carry_footers_from(&self, previous: &Table) -> Result<()> {
        let live: HashSet<String> = self
            .entries()?
            .iter()
            .map(|entry| self.location.uri_for(&entry.file.path))
            .collect::<Result<_>>()?;
        let previous = previous.footers.lock().unwrap();
        let mut footers = self.footers.lock().unwrap();
        for (uri, row_groups) in previous.iter() {
            if live.contains(uri) {
                footers
                    .entry(uri.clone())
                    .or_insert_with(|| row_groups.clone());
            }
        }
        Ok(())
    }

    /// Keep the footer a writer holds for the file at `path`, so a scan after
    /// its commit does not read it back.
    pub fn remember_footer(
        &self,
        path: &ObjectPath,
        row_groups: Vec<Arc<RowGroupMetadata>>,
    ) -> Result<()> {
        let uri = self.location.uri_for(path)?;
        self.footers.lock().unwrap().insert(uri, row_groups);
        Ok(())
    }

    /// The row groups a scan over `filters` reads, with the rows their
    /// deletion vectors delete marked on them: those of the files the
    /// snapshot's metadata cannot rule out, a manifest by the partition
    /// summaries of its manifest-list entry, a file by its partition values
    /// and column bounds. Manifest order, then file-local row-group order,
    /// makes the same filters yield the same row-group positions for
    /// scanning and materialization.
    pub fn scan(&self, filters: &[Expression]) -> Result<ParquetTable> {
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: self.name.clone(),
            source: Box::new(source),
        };
        let selected = self
            .manifest_list
            .select(filters)
            .map_err(invalid_partition_metadata)?;
        let mut manifests = self.manifests_for(&selected.data)?;
        if !self.uncommitted.is_empty() {
            manifests.push(Arc::new(self.uncommitted_manifest()?));
        }
        let files = DataFiles::new(&self.name, &self.current.metadata, manifests)?
            .select(filters)
            .map_err(invalid_partition_metadata)?;
        tracing::debug!(table = %self.name, manifests = selected.data.len(), files = files.len(), "selected Iceberg files");
        let delete_manifests =
            read_manifests(&self.name, selected.deletes, &self.store, &self.dispatcher)?;
        let deletion_vectors =
            read_deletion_vectors(delete_manifests, &files, &self.store, &self.dispatcher)?;
        let mut footers = self.read_footers(&files)?;
        for (footer, deleted) in footers.iter_mut().zip(deletion_vectors) {
            if let Some(deleted) = deleted {
                // The held row groups are shared with every other scan of
                // this version; the deleted rows go on this scan's own copy.
                footer.row_groups = footer
                    .row_groups
                    .iter()
                    .map(|row_group| Arc::new(RowGroupMetadata::clone(row_group)))
                    .collect();
                footer.mark_deleted_rows(deleted);
            }
        }
        let row_groups: Vec<_> = footers
            .into_iter()
            .flat_map(|footer| footer.row_groups)
            .collect();
        Ok(if row_groups.is_empty() {
            ParquetTable::empty(self.declared_schema())
        } else {
            ParquetTable::new(row_groups)
        })
    }

    /// The row groups of each of `files`, in the order given: read over the
    /// pool the first time, held for the life of the version after.
    pub fn read_footers(&self, files: &[FileDescriptor]) -> Result<Vec<FileRowGroups>> {
        let mut footers = self.footers.lock().unwrap();
        let missing: Vec<&FileDescriptor> = files
            .iter()
            .filter(|file| !footers.contains_key(&file.path))
            .collect();
        if !missing.is_empty() {
            let data_files = missing
                .iter()
                .map(|file| self.store.data_file(&file.path, file.length))
                .collect::<Result<Vec<_>>>()?;
            let loaded = load_file_row_groups(&self.dispatcher, &data_files, self.columns.clone())?;
            for (file, footer) in missing.iter().zip(loaded) {
                footers.insert(file.path.clone(), footer.row_groups);
            }
        }
        Ok(files
            .iter()
            .map(|file| FileRowGroups {
                file: FileRef {
                    path: ObjectPath::new(&file.path),
                    size: file.length,
                },
                row_groups: footers
                    .get(&file.path)
                    .expect("every requested footer was held or just read")
                    .clone(),
            })
            .collect())
    }

    /// Every live data file of this version, in manifest-list then manifest
    /// order, each stamped with where its manifest records it; the files
    /// swapped in locally, which no manifest records yet, last.
    pub fn entries(&self) -> Result<Vec<DataFileEntry>> {
        let mut entries: Vec<DataFileEntry> = self
            .decoded_manifests()?
            .iter()
            .flat_map(|manifest| manifest.entries.iter().cloned())
            .collect();
        entries.extend(self.uncommitted.iter().cloned());
        Ok(entries)
    }

    /// Every data manifest of the current snapshot, decoded: read over the
    /// pool the first time, held for the life of the version after.
    pub fn decoded_manifests(&self) -> Result<Arc<Vec<Arc<DecodedManifest>>>> {
        let mut decoded = self.decoded.lock().unwrap();
        if let Some(decoded) = decoded.as_ref() {
            return Ok(Arc::clone(decoded));
        }
        let files: Vec<ManifestFile> = self.manifest_list.data_manifests().cloned().collect();
        let all = Arc::new(decode_manifests(
            &self.dispatcher,
            &self.store,
            &self.name,
            &files,
            &self.entry_context(),
        )?);
        *decoded = Some(Arc::clone(&all));
        Ok(all)
    }

    /// The current snapshot's manifests in manifest-list order, each with the
    /// committed entries it records: what a commit rewrites from.
    pub(crate) fn ordered_manifests(&self) -> Result<Vec<Arc<LoadedManifest>>> {
        Ok(self
            .decoded_manifests()?
            .iter()
            .map(|manifest| {
                Arc::new(LoadedManifest {
                    file: manifest.file.clone(),
                    entries: manifest.entries.clone(),
                })
            })
            .collect())
    }

    /// The manifests among `files` (data manifests of the current snapshot),
    /// decoded: from the ones held when all were, else read over the pool.
    fn manifests_for(&self, files: &[&ManifestFile]) -> Result<Vec<Arc<Manifest>>> {
        if let Some(decoded) = self.decoded.lock().unwrap().as_ref() {
            let by_path: HashMap<&str, &Arc<DecodedManifest>> = decoded
                .iter()
                .map(|manifest| (manifest.file.manifest_path.as_str(), manifest))
                .collect();
            return files
                .iter()
                .map(|file| {
                    by_path
                        .get(file.manifest_path.as_str())
                        .map(|manifest| Arc::clone(&manifest.manifest))
                        .ok_or_else(|| Error::UnknownManifest {
                            manifest: file.manifest_path.clone(),
                        })
                })
                .collect();
        }
        Ok(read_manifests(
            &self.name,
            files.iter().copied(),
            &self.store,
            &self.dispatcher,
        )?
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    /// The files swapped in locally as the manifest a commit of them would
    /// write, so a scan selects among them as it does among committed files.
    fn uncommitted_manifest(&self) -> Result<Manifest> {
        let metadata = &self.current.metadata;
        let ctx = self.entry_context();
        let entries = self
            .uncommitted
            .iter()
            .map(|entry| {
                Ok(ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(to_data_file(entry, &ctx)?)
                    .build())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Manifest::new(
            ManifestMetadata {
                schema: metadata.current_schema().clone(),
                schema_id: metadata.current_schema().schema_id(),
                partition_spec: metadata.default_partition_spec().as_ref().clone(),
                format_version: metadata.format_version(),
                content: ManifestContentType::Data,
            },
            entries,
        ))
    }

    /// The rows of the data files less the rows their deletion vectors
    /// delete, read from the manifests alone. Older writers may omit counts,
    /// and a delete manifest may fail to read or hold a delete file that is
    /// not applied; planning then has no statistic and COUNT uses the
    /// ordinary scan, which reports what is wrong.
    pub fn row_count(&self) -> Option<i64> {
        if !self.uncommitted.is_empty() {
            return None;
        }
        let rows = self.manifest_list.row_count()?;
        let delete_manifests = read_manifests(
            &self.name,
            self.manifest_list.delete_manifests(),
            &self.store,
            &self.dispatcher,
        )
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

    /// Whether each declared column may hold a NULL, as the table's schema
    /// declares it.
    pub fn declared_nullability(&self) -> Vec<bool> {
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
                .zip(self.declared_nullability())
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

    /// The qualified name the table is known by.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn catalog(&self) -> &Arc<dyn TableCatalog> {
        &self.catalog
    }

    pub fn location(&self) -> &TableLocation {
        &self.location
    }

    pub fn store(&self) -> &TableStore {
        &self.store
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

    pub fn columns(&self) -> &[Column] {
        self.columns.columns()
    }

    /// The declared columns keyed by the field ids the files carry.
    pub fn table_columns(&self) -> &TableColumns {
        &self.columns
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

    pub fn manifest_list_size(&self) -> Option<u64> {
        self.manifest_list_size
    }

    /// How a manifest entry of this table is read and written.
    pub(crate) fn entry_context(&self) -> EntryContext<'_> {
        let metadata = &self.current.metadata;
        EntryContext {
            schema: metadata.current_schema(),
            partition_spec: metadata.default_partition_spec(),
            columns: self.columns(),
            location: &self.location,
            format_version: metadata.format_version(),
        }
    }
}
