//! Committing to a table: the spec crate's transaction builds the new
//! snapshot's manifests and manifest list and the changes to the metadata
//! document, and the table's [`TableCatalog`](crate::TableCatalog) makes the
//! result current.
//!
//! A commit from one version writes its Avro objects under new names, then
//! asks the catalog to land the new metadata as the successor of the version
//! it was built on. Exactly one of several writers from the same base
//! succeeds; the rest learn they lost, withdraw their objects, reload, and
//! try again on top of the winner.

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
use planner::catalog::Column;
use uuid::Uuid;

use crate::data_files::DataFileEntry;
use crate::directory::DirectoryCatalog;
use crate::layout::TableLocation;
use crate::manifests::ManifestList;
use crate::retention::{
    DEFAULT_PREVIOUS_VERSIONS, DEFAULT_SNAPSHOT_RETENTION, PREVIOUS_VERSIONS_PROPERTY,
    SNAPSHOT_RETENTION_PROPERTY, expired_snapshot_ids,
};
use crate::runtime::block_on;
use crate::schema::{
    LOGICAL_TYPES_PROPERTY, NAME_MAPPING_PROPERTY, build_partition_spec, build_schema,
    build_sort_order, logical_types_property, name_mapping_property,
};
use crate::state::CurrentMetadata;
use crate::storage::build_recording_file_io;
use crate::store::TableStore;
use crate::table::Table;
use crate::{Error, Result};

/// How many times a store write that failed with an IO error is retried
/// before the commit gives up, and how many times a writer that lost the
/// race rebuilds its commit on the winner before giving up.
pub const COMMIT_RETRIES: usize = 10;
const RETRY_BACKOFF_BASE: Duration = Duration::from_millis(10);
const RETRY_BACKOFF_CAP: Duration = Duration::from_secs(1);

/// The namespace a table is addressed under in the spec crate's transaction,
/// which names a table by an identifier and never reads it back: a table here
/// is found by its catalog, not by name.
const TRANSACTION_NAMESPACE: &str = "pivot";

/// What `CREATE TABLE` asks a new table to be.
pub struct TableCreation<'a> {
    pub columns: &'a [Column],
    pub partition_by: &'a [String],
    pub sort_by: &'a [String],
    /// Files the table adopts as its initial data, each with its statistics
    /// and, for a partitioned table, its constant partition tuple.
    pub adopted: Vec<DataFileEntry>,
}

/// The metadata document of a new table at `location` with the shape
/// `creation` asks for, before any version of it is published: format
/// version 3, every field optional, the table's retention properties set,
/// and the Pivot types Iceberg has no spelling for recorded.
pub fn new_table_metadata(
    location: &TableLocation,
    table_uuid: Uuid,
    creation: &TableCreation<'_>,
) -> Result<TableMetadata> {
    let schema = build_schema(creation.columns)?;
    let partition_spec = build_partition_spec(&schema, creation.partition_by)?;
    let sort_order = build_sort_order(&schema, creation.sort_by)?;
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
    let mut properties = HashMap::from([
        (
            SNAPSHOT_RETENTION_PROPERTY.to_string(),
            DEFAULT_SNAPSHOT_RETENTION.as_millis().to_string(),
        ),
        (
            PREVIOUS_VERSIONS_PROPERTY.to_string(),
            DEFAULT_PREVIOUS_VERSIONS.to_string(),
        ),
    ]);
    if let Some(logical_types) = logical_types_property(built.current_schema(), creation.columns) {
        properties.insert(LOGICAL_TYPES_PROPERTY.to_string(), logical_types);
    }
    if !creation.adopted.is_empty() {
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

impl Table {
    /// Create the table at `location`, numbered by `directory`: publish its
    /// first version, holding the adopted files as the first snapshot when
    /// there are any. The version is created, never replaced: a location that
    /// already has one is not this table's to take over.
    #[allow(clippy::too_many_arguments)] // one call per table creation; each piece is needed
    pub fn create(
        dispatcher: &DataFlowDispatcher,
        directory: DirectoryCatalog,
        location: TableLocation,
        store: TableStore,
        table_name: &str,
        table_uuid: Uuid,
        creation: TableCreation<'_>,
        now_ms: i64,
    ) -> Result<Table> {
        let metadata = Arc::new(new_table_metadata(&location, table_uuid, &creation)?);
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
        let unpublished = Table::assemble(
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
            Some(Vec::new()),
        )?;
        unpublished
            .commit(&[], creation.adopted, now_ms)?
            .ok_or(already_initialized)
    }

    /// Commit, on top of this version, the removal of `removed` (entries of
    /// this version) and the addition of `added`, as one new version: an
    /// append when nothing is removed, else a replace, which leaves the
    /// table's rows as they were. Every snapshot that has expired by `now_ms`
    /// is dropped by the same commit. `None` means another writer landed a
    /// version first: the caller refreshes and retries.
    pub fn commit(
        &self,
        removed: &[DataFileEntry],
        added: Vec<DataFileEntry>,
        now_ms: i64,
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
        let ctx = self.entry_context();
        let to_data_files = |entries: &[DataFileEntry]| -> Result<Vec<DataFile>> {
            entries
                .iter()
                .map(|entry| entry.create_data_file(&ctx))
                .collect()
        };
        let removed = to_data_files(removed)?;
        let added = to_data_files(&added)?;
        let expired = expired_snapshot_ids(self.metadata(), now_ms)?;
        self.land(move |transaction| {
            let transaction = expire(transaction, expired)?;
            if removed.is_empty() {
                // The files were just written under fresh names, so none is
                // in the table already; the check for one would read every
                // manifest.
                return transaction
                    .fast_append()
                    .with_check_duplicate(false)
                    .add_data_files(added)
                    .apply(transaction);
            }
            let mut rewrite = transaction.rewrite_files();
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
    /// of `now_ms`, as one metadata-only commit with no change to the current
    /// snapshot. `None` when nothing has expired, or when another writer
    /// landed the next version first (that writer expired them itself).
    pub fn expire_snapshots(&self, now_ms: i64) -> Result<Option<Table>> {
        let expired = expired_snapshot_ids(self.metadata(), now_ms)?;
        if expired.is_empty() {
            return Ok(None);
        }
        self.land(|transaction| expire(transaction, expired))
    }

    /// Commit the transaction `stage` builds on this version through the
    /// table's catalog. `Ok(None)` when another writer landed first: the
    /// objects this commit wrote are withdrawn.
    fn land(
        &self,
        stage: impl FnOnce(Transaction) -> iceberg::Result<Transaction>,
    ) -> Result<Option<Table>> {
        let (file_io, written) = build_recording_file_io(self.store());
        let base = SpecTable::builder()
            .metadata(Arc::clone(self.metadata()))
            .identifier(TableIdent::new(
                NamespaceIdent::new(TRANSACTION_NAMESPACE.to_string()),
                self.name().to_string(),
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
            let transaction = stage(Transaction::new(&base)).map_err(Error::spec)?;
            let catalog = self.catalog().commit_catalog(base)?;
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
            Err(error) => return Err(error),
        };
        let current = CurrentMetadata {
            metadata_location: committed
                .metadata_location_result()
                .map_err(Error::spec)?
                .to_string(),
            metadata: committed.metadata_ref(),
        };
        Table::at(
            self.dispatcher(),
            Arc::clone(self.catalog()),
            self.location().clone(),
            self.store().clone(),
            self.name(),
            current,
            None,
        )
        .map(Some)
    }
}

/// `transaction` with the removal of the snapshots `expired` staged on it.
/// Which snapshots have expired is Pivot's to say (see
/// [`expired_snapshot_ids`]), so the spec crate's own selection by age is
/// switched off: it keeps every snapshot newer than a cutoff nothing is
/// older than.
fn expire(transaction: Transaction, expired: Vec<i64>) -> iceberg::Result<Transaction> {
    if expired.is_empty() {
        return Ok(transaction);
    }
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
    use crate::catalog::TableCatalog;
    use crate::data_files::ManifestOrigin;
    use crate::directory::metadata_location_version;
    use crate::layout::{VERSION_HINT_FILE, parse_hint, parse_metadata_file_version};
    use arrow_array::{ArrayRef, Int64Array, Scalar};
    use iceberg::spec::Operation;
    use object_storage::{FileRef, LocalStore, ObjectPath, ObjectStore};
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

    impl Fixture {
        fn catalog(&self) -> Arc<dyn TableCatalog> {
            Arc::new(DirectoryCatalog::new(self.location.clone()))
        }

        fn store(&self) -> TableStore {
            TableStore::over_datastore_store("main.t1", Arc::clone(self.location.store())).unwrap()
        }

        /// The table as another copy loads it through `catalog`.
        fn load_through(&self, catalog: Arc<dyn TableCatalog>) -> Table {
            Table::load(
                &test_pool(),
                catalog,
                self.location.clone(),
                self.store(),
                "main.t1",
            )
            .unwrap()
        }
    }

    /// One worker pool shared by every test in the binary, for the reads a
    /// load makes over it.
    fn test_pool() -> dispatch::DataFlowDispatcher {
        static POOL: std::sync::OnceLock<dispatch::Dispatch> = std::sync::OnceLock::new();
        POOL.get_or_init(|| dispatch::Dispatch::spin_up(2, 64, None))
            .dispatcher()
            .clone()
    }

    fn create_table(partition_by: &[String]) -> (Fixture, Table) {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location = TableLocation::new(store, ObjectPath::new("t1")).unwrap();
        let fixture = Fixture {
            _dir: dir,
            location,
        };
        let table = Table::create(
            &test_pool(),
            DirectoryCatalog::new(fixture.location.clone()),
            fixture.location.clone(),
            fixture.store(),
            "main.t1",
            Uuid::new_v4(),
            TableCreation {
                columns: &columns(),
                partition_by,
                sort_by: &[],
                adopted: Vec::new(),
            },
            0,
        )
        .unwrap();
        (fixture, table)
    }

    fn append(table: &Table, entries: Vec<DataFileEntry>, now_ms: i64) -> Table {
        table
            .commit(&[], entries, now_ms)
            .unwrap()
            .expect("no other writer")
    }

    fn version(table: &Table) -> u64 {
        metadata_location_version(table.metadata_location()).unwrap()
    }

    fn load(fixture: &Fixture) -> Table {
        fixture.load_through(fixture.catalog())
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
    fn appends_land_at_successive_versions_with_contiguous_row_ids() {
        let (fixture, created) = create_table(&[]);

        let after_a = append(&created, vec![entry("a", 100, 1, 100)], 1_000);
        let after_b = append(&after_a, vec![entry("b", 50, 101, 150)], 2_000);
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
        let after_a = append(&created, vec![entry("a", 1, 1, 1)], 1_000);
        let objects_before = avro_objects(&fixture.location);

        let lost = created
            .commit(&[], vec![entry("b", 1, 2, 2)], 1_000)
            .unwrap();

        assert!(lost.is_none());
        assert_eq!(metadata_versions(&fixture.location), [1, 2]);
        assert_eq!(avro_objects(&fixture.location), objects_before);
        assert_eq!(paths(&after_a), ["a.parquet"]);
    }

    #[test]
    fn a_removal_retires_the_file_and_carries_the_rest_with_their_numbers() {
        let (fixture, created) = create_table(&[]);
        let table = append(
            &created,
            vec![entry("a", 10, 1, 10), entry("b", 10, 11, 20)],
            1_000,
        );
        let table = append(&table, vec![entry("c", 10, 21, 30)], 2_000);
        let removed = live_entry(&table, "a.parquet");
        let b_origin = origin_of(&table, "b.parquet");

        let replaced = table
            .commit(&[removed], vec![entry("d", 10, 1, 10)], 3_000)
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
        let writer = append(&created, vec![entry("a", 1, 1, 1)], 1_000);

        let refreshed = reader.refresh().unwrap().unwrap();
        let unchanged = refreshed.refresh().unwrap();

        assert_eq!(version(&refreshed), version(&writer));
        assert_eq!(paths(&refreshed), ["a.parquet"]);
        assert!(unchanged.is_none());
    }

    #[test]
    fn snapshots_past_the_window_are_dropped_by_the_next_commit() {
        // Every commit is dated by the wall clock, so only the clock the
        // expiry reads moves.
        let created_at = now_unix_ms();
        let (_fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)], created_at);
        let table = append(&table, vec![entry("b", 1, 2, 2)], created_at + HOUR_MS);

        let table = append(&table, vec![entry("c", 1, 3, 3)], created_at + 6 * HOUR_MS);

        let sequence_numbers: Vec<i64> = table
            .metadata()
            .snapshots()
            .map(|snapshot| snapshot.sequence_number())
            .collect();
        assert_eq!(table.metadata().snapshots().len(), 2);
        assert!(!sequence_numbers.contains(&1));
        assert_eq!(paths(&table), ["a.parquet", "b.parquet", "c.parquet"]);
    }

    #[test]
    fn expiry_alone_commits_a_metadata_only_version() {
        // Every commit is dated by the wall clock, so only the sweep's clock
        // moves.
        let (_fixture, created) = create_table(&[]);
        let table = append(&created, vec![entry("a", 1, 1, 1)], now_unix_ms());
        let table = append(&table, vec![entry("b", 1, 2, 2)], now_unix_ms());

        let nothing_yet = table.expire_snapshots(now_unix_ms() + 2 * HOUR_MS).unwrap();
        let expired = table
            .expire_snapshots(now_unix_ms() + 6 * HOUR_MS)
            .unwrap()
            .unwrap();

        assert!(nothing_yet.is_none());
        assert_eq!(version(&expired), version(&table) + 1);
        assert_eq!(expired.metadata().snapshots().len(), 1);
        assert_eq!(
            expired.metadata().current_snapshot_id(),
            table.metadata().current_snapshot_id()
        );
        assert_eq!(paths(&expired), paths(&table));
    }

    #[test]
    fn the_hint_follows_every_version() {
        let (fixture, mut table) = create_table(&[]);

        for version in 2..=11u64 {
            table = append(
                &table,
                vec![entry(&format!("f{version}"), 1, 1, 1)],
                i64::try_from(version).unwrap(),
            );
            assert_eq!(hint(&fixture.location), version, "after version {version}");
        }
    }

    #[test]
    fn a_partitioned_table_records_each_files_tuple() {
        let (fixture, created) = create_table(&["v".to_string()]);
        let mut partitioned = entry("a", 3, 7, 7);
        partitioned.partition = Some(PartitionValues::from([("v".to_string(), int_scalar(7))]));

        let table = append(&created, vec![partitioned], 1_000);
        let loaded = load(&fixture);

        assert_eq!(table.partition_by(), ["v"]);
        let tuple = live_entry(&loaded, "a.parquet").partition.unwrap();
        assert_eq!(scalar_equal(&tuple["v"], &int_scalar(7)), Some(true));
    }
}
