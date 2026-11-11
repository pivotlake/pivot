//! Committing to a table: the spec crate's transaction builds the new
//! snapshot's manifests and manifest list and the changes to the metadata
//! document, and the table's [`MetadataPointer`](crate::MetadataPointer)
//! makes the result current.
//!
//! A commit from one version writes its Avro objects under new names, then
//! asks the pointer to make the new metadata the successor of the version
//! it was built on. Exactly one of several writers from the same base
//! succeeds; the rest learn they lost, withdraw their objects, reload, and
//! try again on top of the winner.
//!
//! A commit of files drops no snapshot and no entry of the metadata log.
//! Snapshots and previous metadata files are expired by a commit of its own
//! ([`Table::expire_snapshots`]), which hands back what it dropped so the
//! objects only those snapshots used, and the metadata files, can be
//! deleted: finding them reads manifests, which no INSERT should wait for.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dispatch::DataFlowDispatcher;
use iceberg::Runtime;
use iceberg::spec::{DataFile, FormatVersion, TableMetadata, TableMetadataBuilder};
use iceberg::table::Table as SpecTable;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{ErrorKind, NamespaceIdent, TableIdent};
use object_storage::StoreError;
use planner::catalog::{Column, SchemaQualifiedTableName};
use uuid::Uuid;

use crate::data_files::DataFileEntry;
use crate::directory::DirectoryMetadataPointer;
use crate::expired_objects::ExpiredHistory;
use crate::layout::TableLocation;
use crate::manifests::ManifestList;
use crate::retention::{DEFAULT_SNAPSHOT_RETENTION, SNAPSHOT_RETENTION_PROPERTY};
use crate::runtime::block_on;
use crate::schema::{
    LOGICAL_TYPES_PROPERTY, NAME_MAPPING_PROPERTY, build_partition_spec, build_schema,
    build_sort_order, logical_types_property, name_mapping_property,
};
use crate::state::CurrentMetadata;
use crate::storage::WrittenObjects;
use crate::store::TableStore;
use crate::table::Table;
use crate::{Error, Result};

/// How many times a store write that failed with an IO error is retried
/// before the commit gives up, and how many times a writer that lost the
/// race rebuilds its commit on the winner before giving up.
pub const COMMIT_RETRIES: usize = 10;
const RETRY_BACKOFF_BASE: Duration = Duration::from_millis(10);
const RETRY_BACKOFF_CAP: Duration = Duration::from_secs(1);

/// What `CREATE TABLE` asks a new table to be.
pub struct TableCreation<'a> {
    pub columns: &'a [Column],
    pub partition_by: &'a [String],
    pub sort_by: &'a [String],
    /// Files the table adopts as its initial data, each with its statistics
    /// and, for a partitioned table, its constant partition tuple.
    pub adopted: Vec<DataFileEntry>,
}

impl TableCreation<'_> {
    /// The metadata document of a new table at `location` with the shape
    /// asked for, before any version of it is published: format
    /// version 3, every field optional, the table's retention window set,
    /// and the Pivot types Iceberg has no spelling for recorded.
    pub fn build_metadata(
        &self,
        location: &TableLocation,
        table_uuid: Uuid,
    ) -> Result<TableMetadata> {
        let schema = build_schema(self.columns)?;
        let partition_spec = build_partition_spec(&schema, self.partition_by)?;
        let sort_order = build_sort_order(&schema, self.sort_by)?;
        let built = TableMetadataBuilder::new(
            schema,
            partition_spec,
            sort_order,
            location.table_uri(),
            FormatVersion::V3,
            HashMap::new(),
        )
        .map_err(Error::spec)?
        .assign_uuid(table_uuid)
        .build()
        .map_err(Error::spec)?
        .metadata;
        // The properties key on field ids, so they are taken from the schema as
        // built, after the builder's own id assignment.
        let mut properties = HashMap::from([(
            SNAPSHOT_RETENTION_PROPERTY.to_string(),
            DEFAULT_SNAPSHOT_RETENTION.as_millis().to_string(),
        )]);
        if let Some(logical_types) = logical_types_property(built.current_schema(), self.columns) {
            properties.insert(LOGICAL_TYPES_PROPERTY.to_string(), logical_types);
        }
        if !self.adopted.is_empty() {
            properties.insert(
                NAME_MAPPING_PROPERTY.to_string(),
                name_mapping_property(built.current_schema()),
            );
        }
        Ok(built
            .into_builder(None)
            .set_properties(properties)
            .map_err(Error::spec)?
            .build()
            .map_err(Error::spec)?
            .metadata)
    }
}

impl Table {
    /// Create the table at `location`, numbered by `directory`: publish its
    /// first version, holding the adopted files as the first snapshot when
    /// there are any. The version is created, never replaced: a location that
    /// already has one is not this table's to take over.
    #[allow(clippy::too_many_arguments)] // one call per table creation; each piece is needed
    pub fn create(
        dispatcher: &DataFlowDispatcher,
        directory: DirectoryMetadataPointer,
        location: TableLocation,
        store: TableStore,
        table_name: &SchemaQualifiedTableName,
        table_uuid: Uuid,
        creation: TableCreation<'_>,
    ) -> Result<Table> {
        let metadata = Arc::new(creation.build_metadata(&location, table_uuid)?);
        let already_initialized = Error::TableAlreadyInitialized {
            table: table_name.to_string(),
            location: location.table_uri(),
        };
        if creation.adopted.is_empty() {
            let current = directory
                .publish(None, metadata)?
                .ok_or(already_initialized)?;
            return Table::at(
                dispatcher,
                Arc::new(directory),
                location,
                store,
                table_name,
                current,
                None,
            );
        }
        // The document has no location until it is published, and no base:
        // the commit of the adopted files publishes it as the first version.
        let unpublished = Table::from_manifest_list(
            dispatcher,
            Arc::new(directory),
            location,
            store,
            table_name,
            Arc::new(CurrentMetadata {
                metadata_location: String::new(),
                metadata: Arc::clone(&metadata),
            }),
            ManifestList::empty(&metadata),
            None,
        )?;
        unpublished
            .commit(&[], creation.adopted)?
            .ok_or(already_initialized)
    }

    /// Commit, on top of this version, the removal of `removed` (entries of
    /// this version) and the addition of `added`, as one new version: an
    /// append when nothing is removed, else a replace, which leaves the
    /// table's rows as they were. `None` means another writer committed a
    /// version first: the caller refreshes and retries.
    pub fn commit(
        &self,
        removed: &[DataFileEntry],
        added: Vec<DataFileEntry>,
    ) -> Result<Option<Table>> {
        if let Some(manifest) = self.manifest_list().delete_manifests().next() {
            return Err(Error::DeleteFiles {
                table: self.name().to_string(),
                manifest: manifest.manifest_path.clone(),
            });
        }
        // A version 1 table's manifests have no sequence numbers, and Pivot
        // commits to none.
        let format_version = self.metadata().format_version();
        if format_version < FormatVersion::V2 {
            return Err(Error::UnsupportedFormatVersion {
                table: self.name().to_string(),
                version: format_version as u8,
            });
        }
        let to_data_files = |entries: &[DataFileEntry]| -> Result<Vec<DataFile>> {
            entries
                .iter()
                .map(|entry| entry.create_data_file(self.metadata(), self.location()))
                .collect()
        };
        let removed = to_data_files(removed)?;
        let added = to_data_files(&added)?;
        self.commit_transaction(Vec::new(), move |transaction| {
            // The files were just written under fresh names, so none is in
            // the table already; the check for one would read every manifest.
            // Small manifests are merged as they pile up, so the table's
            // manifests grow with its files and not with its commits.
            if removed.is_empty() {
                return transaction
                    .fast_append()
                    .with_check_duplicate(false)
                    .with_manifest_merge(true)
                    .add_data_files(added)
                    .apply(transaction);
            }
            let mut rewrite = transaction
                .rewrite_files()
                .with_check_duplicate(false)
                .with_manifest_merge(true);
            for file in removed {
                rewrite = rewrite.delete_file(file);
            }
            for file in added {
                rewrite = rewrite.add_file(file);
            }
            rewrite.apply(transaction)
        })
    }

    /// Drop every snapshot that has outlived the table's retention window as
    /// of `now_ms`, and from the metadata log every previous metadata file
    /// that has, as one metadata-only commit with no change to the current
    /// snapshot. `None` when nothing has expired, or when another writer
    /// committed the next version first: the caller refreshes and looks again.
    ///
    /// The expiry handed back is of a commit that succeeded. An error leaves
    /// open whether it did ([`Error::may_have_committed`]) and hands back
    /// nothing to delete by.
    pub fn expire_snapshots(&self, now_ms: i64) -> Result<Option<SnapshotExpiry>> {
        let expired_snapshot_ids = self.expired_snapshot_ids(now_ms)?;
        let expired_metadata_files = self.expired_metadata_files(now_ms)?;
        if expired_snapshot_ids.is_empty() && expired_metadata_files.is_empty() {
            return Ok(None);
        }
        let committed = self.commit_transaction(expired_metadata_files, |transaction| {
            expire(transaction, expired_snapshot_ids)
        })?;
        let Some(table) = committed else {
            return Ok(None);
        };
        let expired = ExpiredHistory::between(self.metadata(), table.metadata());
        Ok(Some(SnapshotExpiry { table, expired }))
    }

    /// Commit the transaction `build_transaction` builds on this version
    /// through the table's metadata pointer, the new version's metadata log
    /// dropping `expired_metadata_files`. `Ok(None)` when another writer
    /// committed first: the objects this commit wrote are withdrawn. They
    /// are withdrawn in no other case: a commit whose outcome is unknown may
    /// have succeeded, and its error says so
    /// ([`Error::may_have_committed`]). A commit that succeeded is loaded
    /// from what it wrote, with no read of the store.
    fn commit_transaction(
        &self,
        expired_metadata_files: Vec<String>,
        build_transaction: impl FnOnce(Transaction) -> iceberg::Result<Transaction>,
    ) -> Result<Option<Table>> {
        let (file_io, written) = self.store().build_recording_file_io();
        let base = SpecTable::builder()
            .metadata(Arc::clone(self.metadata()))
            .identifier(TableIdent::new(
                NamespaceIdent::new(self.name().schema.clone()),
                self.name().table.clone(),
            ))
            .file_io(file_io)
            // A commit reads the manifests it rewrites once.
            .disable_cache();
        // A table being created has no published document to succeed.
        let base = match self.metadata_location() {
            "" => base,
            location => base.metadata_location(location),
        };
        let committed = block_on(async {
            let base = base
                .runtime(Runtime::try_current().map_err(Error::spec)?)
                .build()
                .map_err(Error::spec)?;
            let transaction = build_transaction(Transaction::new(&base)).map_err(Error::spec)?;
            let catalog = self
                .metadata_pointer()
                .commit_catalog(base, expired_metadata_files)?;
            transaction
                .commit(catalog.as_ref())
                .await
                .map_err(|error| Error::Catalog(Box::new(error)))
        });
        let committed = match committed {
            Ok(committed) => committed,
            Err(Error::Catalog(error)) if error.kind() == ErrorKind::CatalogCommitConflicts => {
                written.withdraw(self.store());
                return Ok(None);
            }
            Err(Error::Catalog(error)) if is_outcome_unknown(&error) => {
                return Err(Error::CommitOutcomeUnknown {
                    source: Box::new(Error::Catalog(error)),
                });
            }
            Err(error) => return Err(error),
        };
        self.load_committed(&committed, &written)
            .map(Some)
            .map_err(|error| Error::CommittedTableUnloaded {
                table: self.name().to_string(),
                source: Box::new(error),
            })
    }

    /// The table at the version `committed`, which a commit on top of this
    /// version just made current having written `written`. Its manifest list is
    /// the one the commit wrote, or this version's when the commit added no
    /// snapshot, so nothing is read from the store.
    fn load_committed(&self, committed: &SpecTable, written: &WrittenObjects) -> Result<Table> {
        let metadata = committed.metadata_ref();
        let metadata_location = committed
            .metadata_location_result()
            .map_err(Error::spec)?
            .to_string();
        let table_name = self.name().to_string();
        let held_manifest_list = self
            .metadata()
            .current_snapshot()
            .map(|snapshot| snapshot.manifest_list());
        let (manifest_list, manifest_list_size) = match metadata.current_snapshot() {
            None => (ManifestList::empty(&metadata), None),
            Some(snapshot) if Some(snapshot.manifest_list()) == held_manifest_list => {
                (self.manifest_list().clone(), self.manifest_list_size())
            }
            Some(snapshot) => {
                let path = snapshot.manifest_list();
                let bytes = match written.find(path) {
                    Some(bytes) => bytes.to_vec(),
                    None => self.location().read_metadata_object(&table_name, path)?,
                };
                (
                    ManifestList::parse(&table_name, &metadata, &bytes)?,
                    Some(bytes.len() as u64),
                )
            }
        };
        Table::from_manifest_list(
            self.dispatcher(),
            Arc::clone(self.metadata_pointer()),
            self.location().clone(),
            self.store().clone(),
            self.name(),
            Arc::new(CurrentMetadata {
                metadata_location,
                metadata,
            }),
            manifest_list,
            manifest_list_size,
        )
    }
}

/// Whether `error`, a failed commit's, comes from a publish whose outcome is
/// unknown: the catalog passes the publish's own error along as the source.
fn is_outcome_unknown(error: &iceberg::Error) -> bool {
    std::error::Error::source(error)
        .and_then(|source| source.downcast_ref::<Error>())
        .is_some_and(Error::may_have_committed)
}

/// An expiry this process committed: the version it produced and the snapshots
/// and previous metadata files it dropped, which
/// [`Table::collect_expired_objects`] of that version turns into the objects
/// to delete.
pub struct SnapshotExpiry {
    pub table: Table,
    pub expired: ExpiredHistory,
}

/// `transaction` with the removal of the snapshots `expired` staged on it;
/// with none to remove it still commits a version, the one that drops
/// expired metadata files from the log.
/// Which snapshots have expired is Pivot's to say (see
/// [`Table::expired_snapshot_ids`]), so the spec crate's own selection by age is
/// switched off: it keeps every snapshot newer than a cutoff nothing is
/// older than.
fn expire(transaction: Transaction, expired: Vec<i64>) -> iceberg::Result<Transaction> {
    transaction
        .expire_snapshots()
        .expire_snapshot_ids(expired)
        .expire_older_than_ms(i64::MIN)
        .apply(transaction)
}

/// Run `write` until it succeeds, retrying an IO or HTTP failure up to
/// [`COMMIT_RETRIES`] times with a jittered backoff.
pub fn retry_io<T>(mut write: impl FnMut() -> object_storage::Result<T>) -> Result<T> {
    let mut last_error = None;
    for attempt in 0..COMMIT_RETRIES {
        match write() {
            Ok(value) => return Ok(value),
            Err(error @ (StoreError::Io { .. } | StoreError::Http(_))) => {
                tracing::warn!(attempt, error = %error, "a commit write failed; retrying");
                last_error = Some(error);
            }
            Err(error) => return Err(error.into()),
        }
        let ceiling = RETRY_BACKOFF_BASE
            .saturating_mul(1 << attempt.min(16))
            .min(RETRY_BACKOFF_CAP);
        std::thread::sleep(jittered(ceiling));
    }
    Err(Error::CommitRetriesExhausted {
        attempts: COMMIT_RETRIES,
        source: last_error.expect("a retry only follows a recorded failure"),
    })
}

/// A sleep drawn from `[0, ceiling)`, seeded from the clock's subsecond
/// nanoseconds: enough to keep two writers from retrying in lockstep.
fn jittered(ceiling: Duration) -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .subsec_nanos() as u64;
    Duration::from_nanos(nanos % (ceiling.as_nanos() as u64).max(1))
}

pub fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_files::ManifestOrigin;
    use crate::directory::metadata_location_version;
    use crate::layout::{VERSION_HINT_FILE, parse_hint, parse_metadata_file_version};
    use crate::metadata_pointer::MetadataPointer;
    use crate::test_support::test_pool;
    use arrow_array::{ArrayRef, Int64Array, Scalar};
    use iceberg::spec::Operation;
    use object_storage::{FileRef, LocalStore, ObjectPath, ObjectStore, StoreError};
    use parquet_engine::{FileStats, PartitionValues, scalar_equal};
    use planner::types::Type;

    const HOUR_MS: i64 = 60 * 60 * 1000;

    fn columns() -> Vec<Column> {
        vec![Column {
            name: "v".to_string(),
            col_type: Type::Int64,
        }]
    }

    /// An entry for `name` holding `rows` rows of `v` in `low..=high`.
    fn entry(name: &str, rows: i64, low: i64, high: i64) -> DataFileEntry {
        let bound = |value: i64| -> ArrayRef { Arc::new(Int64Array::from(vec![value])) };
        DataFileEntry {
            file: FileRef {
                path: ObjectPath::new(format!("{name}.parquet")),
                size: 10,
            },
            partition: None,
            stats: Some(Arc::new(FileStats {
                num_records: Some(rows),
                min_values: HashMap::from([("v".to_string(), bound(low))]),
                max_values: HashMap::from([("v".to_string(), bound(high))]),
                null_counts: HashMap::from([("v".to_string(), 0)]),
            })),
            sort_order_id: None,
            origin: None,
        }
    }

    fn int_scalar(value: i64) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef)
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        location: TableLocation,
    }

    /// The name every fixture's table is known by.
    fn table_name() -> SchemaQualifiedTableName {
        SchemaQualifiedTableName::new("main", "t1")
    }

    impl Fixture {
        fn metadata_pointer(&self) -> Arc<dyn MetadataPointer> {
            Arc::new(DirectoryMetadataPointer::new(self.location.clone()))
        }

        fn store(&self) -> TableStore {
            TableStore::over_datastore_store(
                &table_name().to_string(),
                Arc::clone(self.location.store()),
            )
            .unwrap()
        }

        /// The table as another copy loads it through `metadata_pointer`.
        fn load_through(&self, metadata_pointer: Arc<dyn MetadataPointer>) -> Table {
            Table::load(
                &test_pool(),
                metadata_pointer,
                self.location.clone(),
                self.store(),
                &table_name(),
            )
            .unwrap()
        }
    }

    fn create_table(partition_by: &[String]) -> (Fixture, Table) {
        create_table_laid_out_by(partition_by, &[])
    }

    /// `table` after `change` to its metadata is published as its next
    /// version, the way another tool alters a table.
    fn publish_changed(
        fixture: &Fixture,
        table: &Table,
        change: impl FnOnce(TableMetadataBuilder) -> TableMetadataBuilder,
    ) -> Table {
        let builder = TableMetadata::clone(table.metadata())
            .into_builder(Some(table.metadata_location().to_string()));
        let changed = change(builder).build().unwrap().metadata;
        DirectoryMetadataPointer::new(fixture.location.clone())
            .publish(Some(table.current()), Arc::new(changed))
            .unwrap()
            .expect("no other writer");
        fixture.load_through(fixture.metadata_pointer())
    }

    fn with_retention_of_one_millisecond(fixture: &Fixture, table: &Table) -> Table {
        publish_changed(fixture, table, |builder| {
            builder
                .set_properties(HashMap::from([(
                    SNAPSHOT_RETENTION_PROPERTY.to_string(),
                    "1".to_string(),
                )]))
                .unwrap()
        })
    }

    /// `table` with the tag `release` on its current snapshot.
    fn with_tag_on_current_snapshot(fixture: &Fixture, table: &Table) -> Table {
        use iceberg::spec::{SnapshotReference, SnapshotRetention};
        let tagged_snapshot_id = table.metadata().current_snapshot_id().unwrap();
        publish_changed(fixture, table, |builder| {
            builder
                .set_ref(
                    "release",
                    SnapshotReference::new(
                        tagged_snapshot_id,
                        SnapshotRetention::Tag {
                            max_ref_age_ms: None,
                        },
                    ),
                )
                .unwrap()
        })
    }

    fn create_table_laid_out_by(partition_by: &[String], sort_by: &[String]) -> (Fixture, Table) {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location = TableLocation::new(store, ObjectPath::new("t1")).unwrap();
        let fixture = Fixture {
            _dir: dir,
            location,
        };
        let table = Table::create(
            &test_pool(),
            DirectoryMetadataPointer::new(fixture.location.clone()),
            fixture.location.clone(),
            fixture.store(),
            &table_name(),
            Uuid::new_v4(),
            TableCreation {
                columns: &columns(),
                partition_by,
                sort_by,
                adopted: Vec::new(),
            },
        )
        .unwrap();
        (fixture, table)
    }

    /// What a [`FaultyStore`] does to the calls a commit makes.
    #[derive(Debug, Default)]
    struct Faults {
        /// How many writes of a metadata file are applied and then answered
        /// with an error, as when the store's answer is lost on the way.
        lost_put_answers: std::sync::atomic::AtomicUsize,
        /// Whether every write of a metadata file fails without being applied.
        fail_metadata_puts: std::sync::atomic::AtomicBool,
        /// Whether every whole-object read fails.
        fail_gets: std::sync::atomic::AtomicBool,
    }

    /// A store that fails the calls its [`Faults`] name and passes the rest on.
    #[derive(Debug)]
    struct FaultyStore {
        inner: Arc<dyn ObjectStore>,
        faults: Arc<Faults>,
    }

    fn injected_failure(key: &ObjectPath) -> StoreError {
        StoreError::Io {
            key: key.to_string(),
            source: std::io::Error::other("injected failure"),
        }
    }

    impl ObjectStore for FaultyStore {
        fn source(
            &self,
            key: &ObjectPath,
        ) -> object_storage::Result<object_storage::DataFileLocation> {
            self.inner.source(key)
        }

        fn get(&self, key: &ObjectPath) -> object_storage::Result<Option<Vec<u8>>> {
            if self
                .faults
                .fail_gets
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                return Err(injected_failure(key));
            }
            self.inner.get(key)
        }

        fn put(&self, key: &ObjectPath, data: &[u8]) -> object_storage::Result<()> {
            self.inner.put(key, data)
        }

        fn put_if_absent(&self, key: &ObjectPath, data: &[u8]) -> object_storage::Result<()> {
            use std::sync::atomic::Ordering;
            if !key.as_str().ends_with(".metadata.json") {
                return self.inner.put_if_absent(key, data);
            }
            if self.faults.fail_metadata_puts.load(Ordering::SeqCst) {
                return Err(injected_failure(key));
            }
            let is_answer_lost = self
                .faults
                .lost_put_answers
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                    left.checked_sub(1)
                })
                .is_ok();
            self.inner.put_if_absent(key, data)?;
            if is_answer_lost {
                return Err(injected_failure(key));
            }
            Ok(())
        }

        fn update(
            &self,
            key: &ObjectPath,
            apply: &mut dyn FnMut(Option<Vec<u8>>) -> Option<Vec<u8>>,
        ) -> object_storage::Result<()> {
            self.inner.update(key, apply)
        }

        fn delete(&self, key: &ObjectPath) -> object_storage::Result<()> {
            self.inner.delete(key)
        }

        fn list_filtered(
            &self,
            directory: &ObjectPath,
            name_prefix: &str,
            after: Option<&str>,
        ) -> object_storage::Result<object_storage::DirectoryListing> {
            self.inner.list_filtered(directory, name_prefix, after)
        }

        fn sink(
            &self,
            key: &ObjectPath,
        ) -> object_storage::Result<object_storage::DataFileLocation> {
            self.inner.sink(key)
        }

        fn create_dir(&self, prefix: &ObjectPath) -> object_storage::Result<()> {
            self.inner.create_dir(prefix)
        }

        fn absolute_key(&self, key: &ObjectPath) -> object_storage::Result<ObjectPath> {
            self.inner.absolute_key(key)
        }

        fn location_uri(&self) -> String {
            self.inner.location_uri()
        }

        fn connection(&self) -> object_storage::StoreConnection {
            self.inner.connection()
        }
    }

    /// The fixture's table as a writer loads it whose store fails the way
    /// `faults` says.
    fn load_over_faulty_store(fixture: &Fixture, faults: &Arc<Faults>) -> Table {
        let store: Arc<dyn ObjectStore> = Arc::new(FaultyStore {
            inner: Arc::clone(fixture.location.store()),
            faults: Arc::clone(faults),
        });
        let location = TableLocation::new(Arc::clone(&store), ObjectPath::new("t1")).unwrap();
        Table::load(
            &test_pool(),
            Arc::new(DirectoryMetadataPointer::new(location.clone())),
            location,
            TableStore::over_datastore_store(&table_name().to_string(), store).unwrap(),
            &table_name(),
        )
        .unwrap()
    }

    fn append(table: &Table, entries: Vec<DataFileEntry>) -> Table {
        table
            .commit(&[], entries)
            .unwrap()
            .expect("no other writer")
    }

    fn version(table: &Table) -> u64 {
        metadata_location_version(table.metadata_location()).unwrap()
    }

    fn load(fixture: &Fixture) -> Table {
        fixture.load_through(fixture.metadata_pointer())
    }

    fn metadata_versions(location: &TableLocation) -> Vec<u64> {
        let mut versions: Vec<u64> = location
            .store()
            .list(&location.metadata_dir())
            .unwrap()
            .objects
            .iter()
            .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
            .collect();
        versions.sort_unstable();
        versions
    }

    fn avro_objects(location: &TableLocation) -> usize {
        location
            .store()
            .list(&location.metadata_dir())
            .unwrap()
            .objects
            .iter()
            .filter(|object| object.file.path.as_str().ends_with(".avro"))
            .count()
    }

    fn hint(location: &TableLocation) -> u64 {
        parse_hint(
            &location
                .store()
                .get(&location.metadata_key(VERSION_HINT_FILE))
                .unwrap()
                .unwrap(),
        )
        .unwrap()
    }

    /// The live files' names, sorted: a table lists new manifests before
    /// carried ones, so the order is not what a test asserts on.
    fn paths(table: &Table) -> Vec<String> {
        let mut paths: Vec<String> = table
            .entries()
            .unwrap()
            .iter()
            .map(|entry| entry.file.path.to_string())
            .collect();
        paths.sort_unstable();
        paths
    }

    /// The live files' names in the order the table holds them.
    fn paths_in_order(table: &Table) -> Vec<String> {
        table
            .entries()
            .unwrap()
            .iter()
            .map(|entry| entry.file.path.to_string())
            .collect()
    }

    fn live_entry(table: &Table, path: &str) -> DataFileEntry {
        table
            .entries()
            .unwrap()
            .into_iter()
            .find(|entry| entry.file.path.as_str() == path)
            .unwrap_or_else(|| panic!("{path} is live"))
    }

    fn origin_of(table: &Table, path: &str) -> ManifestOrigin {
        live_entry(table, path).origin.unwrap()
    }

    #[test]
    fn a_new_table_is_version_one_with_a_hint_and_no_snapshot() {
        let (fixture, table) = create_table(&[]);

        let loaded = load(&fixture);

        assert_eq!(version(&table), 1);
        assert_eq!(metadata_versions(&fixture.location), [1]);
        assert_eq!(hint(&fixture.location), 1);
        assert_eq!(version(&loaded), 1);
        assert_eq!(loaded.metadata().format_version(), FormatVersion::V3);
        assert_eq!(loaded.uuid(), table.uuid());
        assert!(loaded.metadata().current_snapshot().is_none());
        assert!(loaded.entries().unwrap().is_empty());
        assert_eq!(loaded.columns(), columns().as_slice());
    }

    #[test]
    fn appends_commit_at_successive_versions_with_contiguous_row_ids() {
        let (fixture, created) = create_table(&[]);

        let after_a = append(&created, vec![entry("a", 100, 1, 100)]);
        let after_b = append(&after_a, vec![entry("b", 50, 101, 150)]);
        let loaded = load(&fixture);

        assert_eq!((version(&after_a), version(&after_b)), (2, 3));
        assert_eq!(paths(&loaded), ["a.parquet", "b.parquet"]);
        let (a, b) = (
            origin_of(&loaded, "a.parquet"),
            origin_of(&loaded, "b.parquet"),
        );
        assert_eq!((a.sequence_number, b.sequence_number), (1, 2));
        assert_eq!((a.first_row_id, b.first_row_id), (Some(0), Some(100)));
        assert_eq!(a, origin_of(&after_b, "a.parquet"));
        assert_eq!(b, origin_of(&after_b, "b.parquet"));
        assert_eq!(loaded.metadata().next_row_id(), 150);
        assert_eq!(loaded.metadata().snapshots().len(), 2);
    }

    #[test]
    fn the_second_writer_from_one_base_loses_and_leaves_no_objects() {
        let (fixture, created) = create_table(&[]);
        let after_a = append(&created, vec![entry("a", 1, 1, 1)]);
        let objects_before = avro_objects(&fixture.location);

        let lost = created.commit(&[], vec![entry("b", 1, 2, 2)]).unwrap();

        assert!(lost.is_none());
        assert_eq!(metadata_versions(&fixture.location), [1, 2]);
        assert_eq!(avro_objects(&fixture.location), objects_before);
        assert_eq!(paths(&after_a), ["a.parquet"]);
    }

    #[test]
    fn a_commit_whose_write_succeeded_behind_a_lost_answer_reports_success() {
        use std::sync::atomic::Ordering;
        let (fixture, _created) = create_table(&[]);
        let faults = Arc::new(Faults::default());
        let table = load_over_faulty_store(&fixture, &faults);
        faults.lost_put_answers.store(1, Ordering::SeqCst);

        let committed = table.commit(&[], vec![entry("a", 1, 1, 1)]).unwrap();

        assert_eq!(version(&committed.expect("the commit succeeded")), 2);
        assert_eq!(paths(&load(&fixture)), ["a.parquet"]);
        assert_eq!(avro_objects(&fixture.location), 2);
    }

    #[test]
    fn a_commit_whose_write_never_answers_keeps_its_objects_and_says_so() {
        use std::sync::atomic::Ordering;
        let (fixture, _created) = create_table(&[]);
        let faults = Arc::new(Faults::default());
        let table = load_over_faulty_store(&fixture, &faults);
        faults.fail_metadata_puts.store(true, Ordering::SeqCst);

        let error = table
            .commit(&[], vec![entry("a", 1, 1, 1)])
            .err()
            .expect("the commit's outcome is unknown");

        assert!(error.may_have_committed());
        assert!(error.to_string().contains("outcome is unknown"), "{error}");
        assert_eq!(avro_objects(&fixture.location), 2);
    }

    #[test]
    fn a_committed_version_is_loaded_without_reading_the_store() {
        use std::sync::atomic::Ordering;
        let (fixture, _created) = create_table(&[]);
        let faults = Arc::new(Faults::default());
        let table = load_over_faulty_store(&fixture, &faults);
        faults.fail_gets.store(true, Ordering::SeqCst);

        let committed = table.commit(&[], vec![entry("a", 1, 1, 1)]).unwrap();

        let committed = committed.expect("the commit succeeded");
        assert_eq!(version(&committed), 2);
        assert_eq!(paths(&committed), ["a.parquet"]);
        assert_eq!(avro_objects(&fixture.location), 2);
    }

    #[test]
    fn a_removal_retires_the_file_and_carries_the_rest_with_their_numbers() {
        let (fixture, created) = create_table(&[]);
        let table = append(
            &created,
            vec![entry("a", 10, 1, 10), entry("b", 10, 11, 20)],
        );
        let table = append(&table, vec![entry("c", 10, 21, 30)]);
        let removed = live_entry(&table, "a.parquet");
        let b_origin = origin_of(&table, "b.parquet");

        let replaced = table
            .commit(&[removed], vec![entry("d", 10, 1, 10)])
            .unwrap()
            .unwrap();
        let loaded = load(&fixture);

        assert_eq!(paths(&replaced), ["b.parquet", "c.parquet", "d.parquet"]);
        assert_eq!(paths(&loaded), paths(&replaced));
        let b = origin_of(&loaded, "b.parquet");
        assert_eq!(b.sequence_number, b_origin.sequence_number);
        assert_eq!(b.first_row_id, b_origin.first_row_id);
        assert_ne!(b.manifest_path, b_origin.manifest_path);
        let snapshot = loaded.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.summary().operation, Operation::Replace);
        assert_eq!(
            snapshot.summary().additional_properties["deleted-data-files"],
            "1"
        );
        assert_eq!(loaded.manifest_list().manifests().count(), 3);
    }

    #[test]
    fn refresh_sees_a_version_committed_from_another_copy() {
        let (fixture, created) = create_table(&[]);
        let reader = load(&fixture);
        let writer = append(&created, vec![entry("a", 1, 1, 1)]);

        let refreshed = reader.refresh().unwrap().unwrap();
        let unchanged = refreshed.refresh().unwrap();

        assert_eq!(version(&refreshed), version(&writer));
        assert_eq!(paths(&refreshed), ["a.parquet"]);
        assert!(unchanged.is_none());
    }

    #[test]
    fn a_commit_drops_no_snapshot_however_far_past_the_window() {
        let (fixture, created) = create_table(&[]);
        let table = with_retention_of_one_millisecond(&fixture, &created);
        let table = append(&table, vec![entry("a", 1, 1, 1)]);
        let table = append(&table, vec![entry("b", 1, 2, 2)]);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(table.expired_snapshot_ids(now_unix_ms()).unwrap().len(), 1);

        let table = append(&table, vec![entry("c", 1, 3, 3)]);

        assert_eq!(table.metadata().snapshots().len(), 3);
        assert_eq!(paths(&table), ["a.parquet", "b.parquet", "c.parquet"]);
    }

    #[test]
    fn a_tagged_snapshot_outlives_the_window_and_commits_keep_landing() {
        let (fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)]);
        let tagged_snapshot_id = table.metadata().current_snapshot_id().unwrap();
        let table = with_tag_on_current_snapshot(&fixture, &table);
        let table = append(&table, vec![entry("b", 1, 2, 2)]);
        let untagged_snapshot_id = table.metadata().current_snapshot_id().unwrap();
        let table = append(&table, vec![entry("c", 1, 3, 3)]);

        let much_later = now_unix_ms() + 100 * HOUR_MS;
        let table = table.expire_snapshots(much_later).unwrap().unwrap().table;
        let table = append(&table, vec![entry("d", 1, 4, 4)]);

        let snapshot_ids: Vec<i64> = table
            .metadata()
            .snapshots()
            .map(|snapshot| snapshot.snapshot_id())
            .collect();
        assert!(snapshot_ids.contains(&tagged_snapshot_id));
        assert!(!snapshot_ids.contains(&untagged_snapshot_id));
        assert_eq!(
            paths(&table),
            ["a.parquet", "b.parquet", "c.parquet", "d.parquet"]
        );
    }

    #[test]
    fn expiry_alone_commits_a_metadata_only_version() {
        // Every commit is dated by the wall clock, so only the sweep's clock
        // moves.
        let (_fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)]);
        let table = append(&table, vec![entry("b", 1, 2, 2)]);

        let nothing_yet = table.expire_snapshots(now_unix_ms() + 2 * HOUR_MS).unwrap();
        let expiry = table
            .expire_snapshots(now_unix_ms() + 6 * HOUR_MS)
            .unwrap()
            .unwrap();

        assert!(nothing_yet.is_none());
        assert_eq!(expiry.expired.snapshot_count(), 1);
        let expired = expiry.table;
        assert_eq!(version(&expired), version(&table) + 1);
        assert_eq!(expired.metadata().snapshots().len(), 1);
        assert_eq!(
            expired.metadata().current_snapshot_id(),
            table.metadata().current_snapshot_id()
        );
        assert_eq!(paths(&expired), paths(&table));
    }

    /// The table's names for the objects `expiry` leaves to delete, sorted.
    fn objects_of(expiry: &SnapshotExpiry) -> Vec<String> {
        let mut objects: Vec<String> = expiry
            .table
            .collect_expired_objects(&expiry.expired)
            .unwrap()
            .iter()
            .map(|path| path.to_string())
            .collect();
        objects.sort_unstable();
        objects
    }

    /// The table's names for the objects an expiry `far_later` leaves to
    /// delete for the snapshots it dropped, sorted: the metadata files it
    /// dropped are left out.
    fn expired_snapshot_objects(table: &Table) -> Vec<String> {
        let far_later = now_unix_ms() + 100 * HOUR_MS;
        let expiry = table.expire_snapshots(far_later).unwrap().unwrap();
        let mut objects = objects_of(&expiry);
        objects.retain(|object| !object.ends_with(".metadata.json"));
        objects
    }

    /// The table's names for the metadata files `expiry` leaves to delete,
    /// by version.
    fn expired_metadata_versions(expiry: &SnapshotExpiry) -> Vec<u64> {
        objects_of(expiry)
            .iter()
            .filter_map(|object| parse_metadata_file_version(object.rsplit('/').next()?))
            .collect()
    }

    /// Delete from the fixture's store every object `expiry` leaves to
    /// delete, as the process that committed it does.
    fn delete_objects_of(fixture: &Fixture, expiry: &SnapshotExpiry) {
        let location = &fixture.location;
        for path in expiry
            .table
            .collect_expired_objects(&expiry.expired)
            .unwrap()
        {
            location
                .store()
                .delete(&location.data_file_key(&path))
                .unwrap();
        }
    }

    /// The versions `table`'s metadata log names, in the log's order.
    fn logged_versions(table: &Table) -> Vec<u64> {
        table
            .metadata()
            .metadata_log()
            .iter()
            .map(|entry| metadata_location_version(&entry.metadata_file).unwrap())
            .collect()
    }

    /// The wall clock at a moment no commit is dated at: every commit before
    /// the call is dated earlier and every commit after it later.
    fn pause_between_commits() -> i64 {
        std::thread::sleep(Duration::from_millis(5));
        let paused_at = now_unix_ms();
        std::thread::sleep(Duration::from_millis(5));
        paused_at
    }

    const WINDOW_MS: i64 = DEFAULT_SNAPSHOT_RETENTION.as_millis() as i64;

    /// The table's names for the manifest list and the manifests of
    /// `table`'s current snapshot.
    fn current_snapshot_objects(table: &Table) -> Vec<String> {
        let location = table.location();
        let list = table.metadata().current_snapshot().unwrap().manifest_list();
        std::iter::once(list)
            .chain(
                table
                    .manifest_list()
                    .manifests()
                    .map(|manifest| manifest.manifest_path.as_str()),
            )
            .map(|uri| location.file_path_for_uri(uri).unwrap().to_string())
            .collect()
    }

    #[test]
    fn an_expiry_leaves_to_delete_what_only_its_snapshots_used() {
        let (_fixture, created) = create_table(&[]);
        let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
        let removed = live_entry(&appended, "a.parquet");
        let replaced = appended
            .commit(&[removed], vec![entry("c", 1, 1, 1)])
            .unwrap()
            .unwrap();

        let objects = expired_snapshot_objects(&replaced);

        let mut expected = current_snapshot_objects(&appended);
        expected.push("a.parquet".to_string());
        expected.sort_unstable();
        assert_eq!(objects, expected);
    }

    #[test]
    fn an_expiry_leaves_what_a_retained_snapshot_shares_with_an_expired_one() {
        let (_fixture, created) = create_table(&[]);
        let first = append(&created, vec![entry("a", 1, 1, 1)]);
        let second = append(&first, vec![entry("b", 1, 2, 2)]);

        let objects = expired_snapshot_objects(&second);

        let first_list = first.metadata().current_snapshot().unwrap().manifest_list();
        assert_eq!(
            objects,
            [second
                .location()
                .file_path_for_uri(first_list)
                .unwrap()
                .to_string()],
            "the first snapshot's manifest, and the file in it, are the second's too"
        );
    }

    #[test]
    fn an_expiry_leaves_what_a_tagged_snapshot_uses() {
        let (fixture, created) = create_table(&[]);
        let appended = append(&created, vec![entry("a", 1, 1, 1)]);
        let tagged = with_tag_on_current_snapshot(&fixture, &appended);
        let removed = live_entry(&tagged, "a.parquet");
        let replaced = tagged
            .commit(&[removed], vec![entry("b", 1, 1, 1)])
            .unwrap()
            .unwrap();
        let replaced = append(&replaced, vec![entry("c", 1, 2, 2)]);

        let objects = expired_snapshot_objects(&replaced);

        let kept = current_snapshot_objects(&appended);
        assert!(!objects.contains(&"a.parquet".to_string()), "{objects:?}");
        assert!(
            kept.iter().all(|object| !objects.contains(object)),
            "{objects:?}"
        );
        assert!(!objects.is_empty(), "the untagged replace expired");
    }

    #[test]
    fn a_commit_drops_no_metadata_log_entry_however_many_there_are() {
        let (fixture, created) = create_table(&[]);
        let mut table = created;

        for number in 0..120 {
            table = append(&table, vec![entry(&format!("f{number:03}"), 1, 1, 1)]);
        }

        let every_earlier_version: Vec<u64> = (1..=120).collect();
        assert_eq!(logged_versions(&table), every_earlier_version);
        assert_eq!(logged_versions(&load(&fixture)), every_earlier_version);
    }

    #[test]
    fn a_table_that_caps_its_previous_versions_at_100_keeps_every_log_entry_across_commits() {
        use iceberg::spec::TableProperties;
        let (fixture, created) = create_table(&[]);
        let mut table = publish_changed(&fixture, &created, |builder| {
            builder
                .set_properties(HashMap::from([(
                    TableProperties::PROPERTY_METADATA_PREVIOUS_VERSIONS_MAX.to_string(),
                    "100".to_string(),
                )]))
                .unwrap()
        });

        for number in 0..105 {
            table = append(&table, vec![entry(&format!("f{number:03}"), 1, 1, 1)]);
        }

        assert_eq!(logged_versions(&table), (1..=106).collect::<Vec<u64>>());
    }

    #[test]
    fn an_expiry_drops_the_metadata_files_replaced_before_the_window_and_no_other() {
        let (fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)]);
        let table = append(&table, vec![entry("b", 1, 2, 2)]);
        let window_start = pause_between_commits();
        let table = append(&table, vec![entry("c", 1, 3, 3)]);
        let table = append(&table, vec![entry("d", 1, 4, 4)]);

        let expiry = table
            .expire_snapshots(window_start + WINDOW_MS)
            .unwrap()
            .unwrap();
        let expired_versions = expired_metadata_versions(&expiry);
        delete_objects_of(&fixture, &expiry);

        // Versions 2 and 3 replaced 1 and 2 before the window; 4 replaced 3
        // inside it.
        assert_eq!(expiry.expired.metadata_file_count(), 2);
        assert_eq!(expired_versions, [1, 2]);
        assert_eq!(logged_versions(&expiry.table), [3, 4, 5]);
        assert_eq!(metadata_versions(&fixture.location), [3, 4, 5, 6]);
        assert_eq!(paths(&load(&fixture)).len(), 4);
    }

    #[test]
    fn an_expiry_with_only_metadata_files_past_the_window_commits_and_drops_them() {
        let (_fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)]);

        let expiry = table
            .expire_snapshots(now_unix_ms() + 100 * HOUR_MS)
            .unwrap()
            .unwrap();

        assert_eq!(expiry.expired.snapshot_count(), 0);
        assert_eq!(expired_metadata_versions(&expiry), [1]);
        assert_eq!(logged_versions(&expiry.table), [2]);
        assert_eq!(version(&expiry.table), 3);
        assert_eq!(paths(&expiry.table), ["a.parquet"]);
    }

    #[test]
    fn an_expiry_with_nothing_past_the_window_commits_nothing() {
        let (fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)]);
        let table = append(&table, vec![entry("b", 1, 2, 2)]);

        let expiry = table
            .expire_snapshots(now_unix_ms() + WINDOW_MS - HOUR_MS)
            .unwrap();

        assert!(expiry.is_none());
        assert_eq!(metadata_versions(&fixture.location), [1, 2, 3]);
        assert_eq!(logged_versions(&load(&fixture)), [1, 2]);
    }

    #[test]
    fn a_freshly_loaded_table_expires_the_metadata_files_its_log_names() {
        let (fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)]);
        append(&table, vec![entry("b", 1, 2, 2)]);
        let window_start = pause_between_commits();
        let reloaded = load(&fixture);
        append(&reloaded, vec![entry("c", 1, 3, 3)]);

        let expiry = load(&fixture)
            .expire_snapshots(window_start + WINDOW_MS)
            .unwrap()
            .unwrap();
        delete_objects_of(&fixture, &expiry);

        assert_eq!(metadata_versions(&fixture.location), [3, 4, 5]);
    }

    #[test]
    fn a_table_with_gc_disabled_drops_no_metadata_file() {
        let (fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)]);
        let table = append(&table, vec![entry("b", 1, 2, 2)]);
        let table = publish_changed(&fixture, &table, |builder| {
            builder
                .set_properties(HashMap::from([(
                    "gc.enabled".to_string(),
                    "false".to_string(),
                )]))
                .unwrap()
        });

        let expiry = table
            .expire_snapshots(now_unix_ms() + 100 * HOUR_MS)
            .unwrap();

        assert!(expiry.is_none());
        assert_eq!(metadata_versions(&fixture.location), [1, 2, 3, 4]);
        assert_eq!(logged_versions(&load(&fixture)), [1, 2, 3]);
    }

    #[test]
    fn an_expiry_whose_outcome_is_unknown_hands_back_no_metadata_file_to_delete() {
        use std::sync::atomic::Ordering;
        let (fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)]);
        append(&table, vec![entry("b", 1, 2, 2)]);
        let faults = Arc::new(Faults::default());
        let table = load_over_faulty_store(&fixture, &faults);
        faults.fail_metadata_puts.store(true, Ordering::SeqCst);

        let error = table
            .expire_snapshots(now_unix_ms() + 100 * HOUR_MS)
            .err()
            .expect("the expiry's outcome is unknown");

        assert!(error.may_have_committed());
        assert_eq!(metadata_versions(&fixture.location), [1, 2, 3]);
    }

    #[test]
    fn a_writer_whose_base_outlived_the_window_loses_after_its_successor_is_deleted() {
        let (fixture, created) = create_table(&[]);
        let table = with_retention_of_one_millisecond(&fixture, &created);
        let stale_writer = load(&fixture);
        let table = append(&table, vec![entry("a", 1, 1, 1)]);
        let table = append(&table, vec![entry("b", 1, 2, 2)]);
        std::thread::sleep(Duration::from_millis(5));
        let expiry = table.expire_snapshots(now_unix_ms()).unwrap().unwrap();
        delete_objects_of(&fixture, &expiry);
        assert_eq!(metadata_versions(&fixture.location), [4, 5]);

        let lost = stale_writer.commit(&[], vec![entry("c", 1, 3, 3)]).unwrap();

        // Version 3, the name the writer's commit takes, was free.
        assert!(lost.is_none());
        assert_eq!(metadata_versions(&fixture.location), [4, 5]);
        assert_eq!(paths(&load(&fixture)), ["a.parquet", "b.parquet"]);
    }

    #[test]
    fn the_hint_follows_every_version() {
        let (fixture, mut table) = create_table(&[]);

        for version in 2..=11u64 {
            table = append(&table, vec![entry(&format!("f{version}"), 1, 1, 1)]);
            assert_eq!(hint(&fixture.location), version, "after version {version}");
        }
    }

    #[test]
    fn a_partitioned_table_records_each_files_tuple() {
        let (fixture, created) = create_table(&["v".to_string()]);
        let mut partitioned = entry("a", 3, 7, 7);
        partitioned.partition = Some(PartitionValues::from([("v".to_string(), int_scalar(7))]));

        let table = append(&created, vec![partitioned]);
        let loaded = load(&fixture);

        assert_eq!(table.partition_by(), ["v"]);
        let tuple = live_entry(&loaded, "a.parquet").partition.unwrap();
        assert_eq!(scalar_equal(&tuple["v"], &int_scalar(7)), Some(true));
    }

    #[test]
    fn a_version_built_on_an_earlier_one_reads_only_the_manifests_that_joined() {
        let (fixture, created) = create_table(&[]);
        let first = append(&created, vec![entry("a", 10, 1, 10)]);
        let first_manifest = origin_of(&first, "a.parquet").manifest_path;
        let second = append(&first, vec![entry("b", 10, 11, 20)]);
        second.carry_over_from(&first);
        let location = &fixture.location;
        let manifest_key =
            location.data_file_key(&location.file_path_for_uri(&first_manifest).unwrap());
        location.store().delete(&manifest_key).unwrap();

        let files = paths(&second);

        assert_eq!(files, ["a.parquet", "b.parquet"]);
    }

    #[test]
    fn a_sorted_tables_files_are_held_in_the_order_they_start_on_the_sort_key() {
        let (_fixture, created) = create_table_laid_out_by(&[], &["v".to_string()]);
        let table = append(&created, vec![entry("late", 10, 50, 60)]);

        let table = append(
            &table,
            vec![entry("middle", 10, 20, 30), entry("early", 10, -5, 10)],
        );

        assert_eq!(
            paths_in_order(&table),
            ["early.parquet", "middle.parquet", "late.parquet"]
        );
    }

    #[test]
    fn an_unsorted_tables_files_are_held_in_path_order() {
        let (fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("zebra", 10, 50, 60)]);

        let table = append(
            &table,
            vec![entry("mango", 10, 20, 30), entry("apple", 10, 1, 10)],
        );

        let by_path = ["apple.parquet", "mango.parquet", "zebra.parquet"];
        assert_eq!(paths_in_order(&table), by_path);
        assert_eq!(paths_in_order(&load(&fixture)), by_path);
    }

    #[test]
    fn a_column_may_hold_nulls_only_while_a_live_file_records_one() {
        let (_fixture, created) = create_table(&[]);
        let mut holed = entry("holed", 10, 11, 20);
        Arc::get_mut(holed.stats.as_mut().unwrap())
            .unwrap()
            .null_counts
            .insert("v".to_string(), 3);

        let dense = append(&created, vec![entry("dense", 10, 1, 10)]);
        let with_holes = append(&dense, vec![holed]);
        let removed = live_entry(&with_holes, "holed.parquet");
        let rewritten = with_holes
            .commit(&[removed], vec![entry("filled", 10, 11, 20)])
            .unwrap()
            .unwrap();

        assert_eq!(created.nullability().unwrap(), [true]);
        assert_eq!(dense.nullability().unwrap(), [false]);
        assert_eq!(with_holes.nullability().unwrap(), [true]);
        assert_eq!(rewritten.nullability().unwrap(), [false]);
    }

    #[test]
    fn many_appends_leave_few_manifests_and_every_file() {
        let (fixture, created) = create_table(&[]);
        let mut table = created;
        let appends = 100;

        for number in 0..appends {
            let low = number * 10;
            table = append(
                &table,
                vec![entry(&format!("f{number:03}"), 10, low, low + 9)],
            );
        }
        let loaded = load(&fixture);

        assert_eq!(paths(&loaded).len(), appends as usize);
        assert_eq!(paths_in_order(&loaded), paths_in_order(&table));
        assert!(loaded.manifest_list().manifests().count() < 10);
    }
}
