//! One Iceberg table at one version: what every datastore reads a table
//! through and writes it through, given only the table's current
//! metadata pointer.
//!
//! A table is loaded from its metadata document and its current snapshot's
//! manifest list. A datastore that plans, compacts and commits against the
//! table asks for its [`FileIndex`], every live file as one record:
//! the manifests are decoded into it a few at a time and dropped, and the
//! next version's index is built on this one's by the manifests that
//! changed. The index is built when a scan, a commit or a statistic first
//! asks for it, so a table that is only listed reads no manifest.
//!
//! A table holds no file's footer. Whatever needs row groups, a scan's
//! plan, a merge, a layout measurement, a description, reads the footers of
//! the files it works on over the worker pool and owns what it is handed;
//! the bytes it read stay in the pool's caches, where the next read of the
//! same footer finds them.

use std::sync::{Arc, Mutex};

use arrow_schema::{Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{Manifest, ManifestFile, TableMetadataRef};
use object_storage::{ObjectPath, load_objects};
use parquet_engine::file_entry::FileEntry;
use parquet_engine::{FileRowGroups, FileStats, ParquetTable, TableColumns, load_file_row_groups};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::expression::Expression;
use planner::types::physical_arrow_type;
use uuid::Uuid;

use crate::columns::{identity_columns, read_table_columns};
use crate::data_files::DataFileEntry;
use crate::deletes::{count_deleted_rows, read_deletion_vectors};
use crate::file_index::{FileIndex, IndexedFile, SortKeyEncoder};
use crate::layout::TableLocation;
use crate::manifests::{FileDescriptor, ManifestList};
use crate::metadata_pointer::MetadataPointer;
use crate::schema::{UNSORTED_ORDER_ID, find_writer_sort_order_id};
use crate::state::CurrentMetadata;
use crate::store::TableStore;
use crate::{Error, Result};

/// An Iceberg table at one committed version, with what reading and writing
/// it needs: its current-metadata pointer, where its files live and the
/// store they are read through, and the pool its reads run on. Immutable as
/// a version: a refresh, a commit or a local file swap hands back the next
/// version as a new value.
pub struct Table {
    /// The name the table is known by in its datastore.
    name: SchemaQualifiedTableName,
    metadata_pointer: Arc<dyn MetadataPointer>,
    location: TableLocation,
    store: TableStore,
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
    /// The size of the manifest list object, which a pool read of it needs;
    /// `None` for a table with no current snapshot.
    manifest_list_size: Option<u64>,
    /// The files the current snapshot's data manifests list, once something
    /// asked for them.
    committed: Mutex<Option<Arc<FileIndex>>>,
    /// The committed files of an earlier version of the table, which this
    /// version's index is built on.
    inherited: Mutex<Option<Arc<FileIndex>>>,
    /// For a local view of a file swap this process committed through
    /// another copy of the table: the files the view reads, the committed
    /// ones less those swapped out, with those swapped in, which no manifest
    /// of this version records. The next refresh reconciles the copy to the
    /// committed metadata.
    swapped: Option<Arc<FileIndex>>,
}

impl Table {
    /// Load the table at the version `metadata_pointer` holds current: a cold
    /// open.
    pub fn load(
        dispatcher: &DataFlowDispatcher,
        metadata_pointer: Arc<dyn MetadataPointer>,
        location: TableLocation,
        store: TableStore,
        name: &SchemaQualifiedTableName,
    ) -> Result<Table> {
        let current = metadata_pointer.load()?;
        Self::at(
            dispatcher,
            metadata_pointer,
            location,
            store,
            name,
            current,
            None,
        )
    }

    /// The table at the version `current` describes. The manifest list is
    /// read over the pool when `manifest_list_size` is known, else fetched
    /// whole.
    pub fn at(
        dispatcher: &DataFlowDispatcher,
        metadata_pointer: Arc<dyn MetadataPointer>,
        location: TableLocation,
        store: TableStore,
        name: &SchemaQualifiedTableName,
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
                    None => location.read_metadata_object(&name.to_string(), path)?,
                };
                let size = bytes.len() as u64;
                (
                    ManifestList::parse(&name.to_string(), metadata, &bytes)?,
                    Some(size),
                )
            }
        };
        Self::from_manifest_list(
            dispatcher,
            metadata_pointer,
            location,
            store,
            name,
            Arc::new(current),
            manifest_list,
            manifest_list_size,
        )
    }

    /// The table at `current` with its manifest list already in hand.
    #[allow(clippy::too_many_arguments)] // the pieces of one version, built in two places
    pub(crate) fn from_manifest_list(
        dispatcher: &DataFlowDispatcher,
        metadata_pointer: Arc<dyn MetadataPointer>,
        location: TableLocation,
        store: TableStore,
        name: &SchemaQualifiedTableName,
        current: Arc<CurrentMetadata>,
        manifest_list: ManifestList,
        manifest_list_size: Option<u64>,
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
        Ok(Table {
            name: name.clone(),
            metadata_pointer,
            location,
            store,
            dispatcher: dispatcher.clone(),
            current,
            columns,
            partition_by,
            sort_by,
            writer_sort_order_id,
            manifest_list,
            manifest_list_size,
            committed: Mutex::new(None),
            inherited: Mutex::new(None),
            swapped: None,
        })
    }

    /// The latest committed version, or `None` when this one is still
    /// current. The new version starts with no index; see
    /// [`carry_over_from`](Self::carry_over_from).
    pub fn refresh(&self) -> Result<Option<Table>> {
        let Some(current) = self.metadata_pointer.refresh(&self.current)? else {
            return Ok(None);
        };
        Self::at(
            &self.dispatcher,
            Arc::clone(&self.metadata_pointer),
            self.location.clone(),
            self.store.clone(),
            &self.name,
            current,
            None,
        )
        .map(Some)
    }

    /// This version with `removed` dropped and `added` swapped in, as a local
    /// view of a swap this process already committed through another copy.
    pub fn with_files_swapped(
        &self,
        removed: &[ObjectPath],
        added: Vec<DataFileEntry>,
    ) -> Result<Table> {
        let mut view = FileIndex::clone(self.file_index()?.as_ref());
        for path in removed {
            view.remove(path);
        }
        let encoder = self.sort_key_encoder()?;
        for entry in &added {
            view.insert(Arc::new(IndexedFile::uncommitted(
                entry,
                &encoder,
                self.metadata(),
                &self.location,
            )?));
        }
        Ok(Table {
            name: self.name.clone(),
            metadata_pointer: Arc::clone(&self.metadata_pointer),
            location: self.location.clone(),
            store: self.store.clone(),
            dispatcher: self.dispatcher.clone(),
            current: Arc::clone(&self.current),
            columns: self.columns.clone(),
            partition_by: self.partition_by.clone(),
            sort_by: self.sort_by.clone(),
            writer_sort_order_id: self.writer_sort_order_id,
            manifest_list: self.manifest_list.clone(),
            manifest_list_size: self.manifest_list_size,
            committed: Mutex::new(self.committed.lock().unwrap().clone()),
            inherited: Mutex::new(self.inherited.lock().unwrap().clone()),
            swapped: Some(Arc::new(view)),
        })
    }

    /// Keep what `previous`, an earlier version of this table, read that this
    /// version still has: its file index, which this version's is then built
    /// on by the manifests that changed. The index carries over only between
    /// versions that read a file alike: of one schema, one partition spec
    /// and one sort order.
    pub fn carry_over_from(&self, previous: &Table) {
        let (metadata, earlier) = (self.metadata(), previous.metadata());
        if metadata.current_schema_id() == earlier.current_schema_id()
            && metadata.default_partition_spec_id() == earlier.default_partition_spec_id()
            && metadata.default_sort_order_id() == earlier.default_sort_order_id()
        {
            // A version nothing read the files of built no index of its
            // own: the one it would have been built on carries over.
            let earlier_index = previous
                .committed
                .lock()
                .unwrap()
                .clone()
                .or_else(|| previous.inherited.lock().unwrap().clone());
            *self.inherited.lock().unwrap() = earlier_index;
        }
    }

    /// The row groups a scan over `filters` reads, with the rows their
    /// deletion vectors delete marked on them: those of the files of the
    /// table's index that the files' partition values and column bounds
    /// cannot rule out, in the index's order: by partition, then by where a
    /// file starts on the sort key, then by path. The order is fixed by the
    /// version, and with file-local row-group order makes the same filters
    /// yield the same row-group positions for scanning and materialization.
    /// Of the delete manifests, those the partition summaries of their
    /// manifest-list entries cannot rule out are read.
    pub fn plan_scan(&self, filters: &[Expression]) -> Result<ParquetTable> {
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: self.name.to_string(),
            source: Box::new(source),
        };
        let index = self.file_index()?;
        let files = index
            .select(&self.current.metadata, filters)
            .map_err(invalid_partition_metadata)?
            .into_iter()
            .map(|file| {
                Ok(FileDescriptor {
                    path: self.location.uri_for(&file.file.path)?,
                    length: file.file.size,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        tracing::debug!(table = %self.name, files = files.len(), "selected Iceberg files");
        let selected_deletes = self
            .manifest_list
            .select_delete_manifests(filters)
            .map_err(invalid_partition_metadata)?;
        let delete_manifests = self.read_manifests(selected_deletes)?;
        let deletion_vectors =
            read_deletion_vectors(delete_manifests, &files, &self.store, &self.dispatcher)?;
        let mut footers = self.read_footers(&files)?;
        for (footer, deleted) in footers.iter_mut().zip(deletion_vectors) {
            if let Some(deleted) = deleted {
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

    /// The row groups of each of `files`, distinct files, in the order
    /// given: their footers read over the pool for this call and handed to
    /// the caller, the table keeping none.
    pub fn read_footers(&self, files: &[FileDescriptor]) -> Result<Vec<FileRowGroups>> {
        let data_files = files
            .iter()
            .map(|file| self.store.data_file(&file.path, file.length))
            .collect::<Result<Vec<_>>>()?;
        Ok(load_file_row_groups(
            &self.dispatcher,
            &data_files,
            self.columns.clone(),
        )?)
    }

    /// The files this version reads, each as one record: those the
    /// current snapshot's data manifests list, or a local swap's view of
    /// them. Built when first asked for, on the index of the version this one
    /// carried over from when there is one, and held for the life of the
    /// version.
    pub fn file_index(&self) -> Result<Arc<FileIndex>> {
        match &self.swapped {
            Some(view) => Ok(Arc::clone(view)),
            None => self.committed_index(),
        }
    }

    /// The files the current snapshot's data manifests list, whatever a local
    /// swap made of them.
    pub(crate) fn committed_index(&self) -> Result<Arc<FileIndex>> {
        let mut committed = self.committed.lock().unwrap();
        if let Some(index) = committed.as_ref() {
            return Ok(Arc::clone(index));
        }
        let listed: Vec<&ManifestFile> = self.manifest_list.data_manifests().collect();
        let inherited = self.inherited.lock().unwrap().take();
        let encoder = self.sort_key_encoder()?;
        let index = Arc::new(FileIndex::advance(
            inherited.as_deref(),
            &listed,
            |manifests| self.decode_manifests(manifests, &encoder),
        )?);
        *committed = Some(Arc::clone(&index));
        Ok(index)
    }

    /// Read `files`, data manifests of the table, over the pool and decode
    /// each into the files it lists, in the order given.
    fn decode_manifests(
        &self,
        files: &[&ManifestFile],
        encoder: &SortKeyEncoder,
    ) -> Result<Vec<Vec<Arc<IndexedFile>>>> {
        let table_name = self.name.to_string();
        self.read_manifests(files.iter().copied())?
            .into_iter()
            .zip(files)
            .map(|(manifest, file)| {
                IndexedFile::collect_live_files(
                    &table_name,
                    file,
                    manifest,
                    encoder,
                    self.metadata(),
                    &self.location,
                )
            })
            .collect()
    }

    fn sort_key_encoder(&self) -> Result<SortKeyEncoder> {
        SortKeyEncoder::new(&self.current.metadata, self.columns(), &self.sort_by)
    }

    /// Every live data file of this version as a commit records it, in the
    /// index's order.
    pub fn entries(&self) -> Result<Vec<DataFileEntry>> {
        self.file_index()?
            .files()
            .map(|file| file.to_entry(self.metadata(), self.columns()))
            .collect()
    }

    /// The live file at `path` as a commit records it, or `None` when this
    /// version has no such file.
    pub fn find_entry(&self, path: &ObjectPath) -> Result<Option<DataFileEntry>> {
        self.file_index()?
            .find(path)
            .map(|file| file.to_entry(self.metadata(), self.columns()))
            .transpose()
    }

    /// The live file at `path` as the datastore's planners read it: its
    /// tuple, its row count, its bounds on the columns the table is sorted
    /// by, and those columns as what its rows are sorted by when its
    /// manifest entry records the sort order Pivot's writer produces for
    /// them. `None` when this version has no such file.
    pub fn find_datastore_entry(&self, path: &ObjectPath) -> Result<Option<FileEntry>> {
        self.file_index()?
            .find(path)
            .map(|file| {
                let is_in_sort_order = file.sort_order_id().is_some()
                    && file.sort_order_id() == self.writer_sort_order_id;
                let sorted_by: Arc<[String]> = if is_in_sort_order {
                    Arc::from(self.sort_by.as_slice())
                } else {
                    Arc::default()
                };
                file.to_datastore_entry(self.metadata(), self.columns(), &self.sort_by, sorted_by)
            })
            .transpose()
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

    /// The statistics the manifest entry of the live file at `path` records
    /// for every declared column, or `None` when this version has no such
    /// file.
    pub fn file_stats(&self, path: &ObjectPath) -> Result<Option<FileStats>> {
        Ok(self
            .file_index()?
            .find(path)
            .map(|file| file.file_stats(self.metadata(), self.columns().iter())))
    }

    /// The rows of every live data file, deleted ones included, as the
    /// files' manifest entries count them.
    pub fn total_rows(&self) -> Result<u64> {
        Ok(self.file_index()?.total_rows())
    }

    /// Whether each declared column may hold a NULL in some file, as the
    /// files' manifest entries record it. A table without files may hold
    /// anything.
    pub fn nullability(&self) -> Result<Vec<bool>> {
        let index = self.file_index()?;
        let schema = self.current.metadata.current_schema();
        Ok(self
            .columns()
            .iter()
            .map(|column| {
                schema
                    .field_id_by_name(&column.name)
                    .is_none_or(|field_id| index.may_hold_nulls(field_id))
            })
            .collect())
    }

    /// Every data manifest of the current snapshot, read over the pool and
    /// kept by no one: what a description of the whole table reads.
    pub fn read_data_manifests(&self) -> Result<Vec<Manifest>> {
        self.read_manifests(self.manifest_list.data_manifests())
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
                self.store.data_file(&manifest.manifest_path, length)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut parsed = Vec::with_capacity(objects.len());
        for object in load_objects(&self.dispatcher, &objects)? {
            let manifest = Manifest::parse_avro(&object.bytes).map_err(|source| {
                Error::MalformedMetadataObject {
                    table: self.name.to_string(),
                    path: object.file.path.as_str().to_string(),
                    source: Box::new(source),
                }
            })?;
            parsed.push(manifest);
        }
        Ok(parsed)
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

    pub fn metadata_pointer(&self) -> &Arc<dyn MetadataPointer> {
        &self.metadata_pointer
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
}
