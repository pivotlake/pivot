//! One table's versions as its catalog keeps them, and one commit to the
//! table: the version the commit is built on, and the one `updateTable` that
//! asks the catalog to make its successor current.
//!
//! The shared table crate derives the committed table from the version the
//! commit was built on, so a commit must succeed only as that version's
//! successor:
//!
//! - The transaction is answered the held version as the table's current
//!   one, so it never builds the commit on any other. No error it is handed
//!   is retryable, so it asks once.
//! - The update is the one call a commit makes to the catalog. It carries
//!   the requirements the transaction built from the held version: the
//!   table's UUID, and `main` still naming the held snapshot. The catalog
//!   checks them as it applies the update, so another writer's snapshot
//!   ends the commit as a lost race.
//! - A change of another kind by another engine since the held version, to
//!   the schema, the properties, the partition spec or the sort order,
//!   leaves a legal table: the appended files are recorded by field id
//!   under the spec and the order they were written with. The committed
//!   table is built from the metadata the catalog answers, and reuses what
//!   the held version learned only when the schema, spec and sort order ids
//!   are the held ones.
//! - The version the catalog answers must hold the held snapshot or one made
//!   on top of it.
//!
//! A catalog that refuses the update did not apply it. One that answers
//! without saying whether it applied it (a server error, a gateway timeout, a
//! connection that broke) may have, and the commit's error says so.

use std::sync::Arc;

use async_trait::async_trait;
use iceberg::table::Table;
use iceberg::{Catalog, ErrorKind, TableCommit};
use iceberg_tables::{CurrentMetadata, MetadataStore, block_on};
use planner::catalog::SchemaQualifiedTableName;

use crate::write::table_ident;

/// One table's versions as its catalog keeps them: what the shared crate's
/// table loads and refreshes from, and commits its next version through.
pub(crate) struct RestMetadataStore {
    catalog: Arc<dyn Catalog>,
    name: SchemaQualifiedTableName,
}

impl RestMetadataStore {
    pub(crate) fn new(catalog: Arc<dyn Catalog>, name: &SchemaQualifiedTableName) -> Self {
        Self {
            catalog,
            name: name.clone(),
        }
    }
}

#[async_trait]
impl MetadataStore for RestMetadataStore {
    /// One load-table call; the catalog answers the same location for an
    /// unchanged table.
    fn load_latest(&self) -> iceberg_tables::Result<CurrentMetadata> {
        let table = block_on(self.catalog.load_table(&table_ident(&self.name)))
            .map_err(|error| iceberg_tables::Error::Catalog(Box::new(error)))?;
        CurrentMetadata::from_catalog_table(&table)
    }

    /// Ask the catalog to apply the commit as the successor of the base, in
    /// one call. A commit conflict means another writer moved `main` off
    /// the base's snapshot. A table that is gone is a refusal. Any other
    /// error of the update leaves the outcome unknown, which covers every
    /// answer the catalog gives without saying whether it applied the
    /// commit and every failure after it did: the shared crate's error
    /// rides along as the source, which is how the table tells an unknown
    /// outcome from a failure. The catalog writes the table's metadata
    /// files and keeps their log itself, so `expired_metadata_files` is
    /// ignored.
    async fn commit_successor(
        &self,
        base: &Table,
        commit: TableCommit,
        _expired_metadata_files: &[String],
    ) -> iceberg::Result<Table> {
        let base_snapshot_id = base.metadata().current_snapshot_id();
        let (message, error) = match self.catalog.update_table(commit).await {
            Ok(committed) => {
                // What a commit built on the base leaves current: the
                // base's snapshot, or one made on top of it.
                let is_built_on_base = match committed.metadata().current_snapshot() {
                    None => base_snapshot_id.is_none(),
                    Some(snapshot) => {
                        Some(snapshot.snapshot_id()) == base_snapshot_id
                            || snapshot.parent_snapshot_id() == base_snapshot_id
                    }
                };
                if is_built_on_base {
                    return Ok(committed);
                }
                (
                    "the catalog applied the commit, but the version it answered is not built on the one the commit was"
                        .to_string(),
                    iceberg::Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "`main` names snapshot {:?} where a successor of snapshot {:?} was committed",
                            committed.metadata().current_snapshot_id(),
                            base_snapshot_id,
                        ),
                    ),
                )
            }
            Err(error) if error.kind() == ErrorKind::CatalogCommitConflicts => {
                return Err(iceberg::Error::new(
                    ErrorKind::CatalogCommitConflicts,
                    format!("the catalog refused the commit's requirements: {error}"),
                ));
            }
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::TableNotFound | ErrorKind::NamespaceNotFound
                ) =>
            {
                return Err(error.with_retryable(false));
            }
            Err(error) => (
                format!("the catalog did not say whether it applied the commit: {error}"),
                error,
            ),
        };
        Err(
            iceberg::Error::new(ErrorKind::Unexpected, message).with_source(
                iceberg_tables::Error::CommitOutcomeUnknown {
                    source: Box::new(iceberg_tables::Error::Catalog(Box::new(error))),
                },
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    use arrow_array::{ArrayRef, Int64Array};
    use dispatch::{DataFlowDispatcher, Dispatch};
    use iceberg::Runtime;
    use iceberg::io::FileIO;
    use iceberg::spec::{TableMetadata, TableMetadataBuilder};
    use iceberg::{Namespace, NamespaceIdent, TableCreation, TableIdent};
    use iceberg_tables::commit::TableCreation as TableShape;
    use iceberg_tables::files::data_files::DataFileEntry;
    use iceberg_tables::{Table as LoadedTable, TableLocation};
    use object_storage::{FileRef, LocalStore, ObjectPath, ObjectStore};
    use parquet_engine::FileStats;
    use planner::catalog::{Column, SchemaQualifiedTableName};
    use planner::types::Type;
    use uuid::Uuid;

    use super::*;

    /// A catalog holding one table's current version in memory, which
    /// applies a commit the way a catalog service does: under the commit's
    /// requirements, as the successor of the version it holds.
    struct OneTableCatalog {
        current: Mutex<CurrentMetadata>,
        /// The kind of error every update is answered with, unapplied.
        update_failure: Mutex<Option<ErrorKind>>,
        /// A change another engine lands just before the next update is
        /// applied.
        racing_change: Mutex<Option<MetadataChange>>,
        loads_asked: AtomicUsize,
        updates_asked: AtomicUsize,
    }

    type MetadataChange = Box<dyn FnOnce(TableMetadataBuilder) -> TableMetadataBuilder + Send>;

    impl fmt::Debug for OneTableCatalog {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("OneTableCatalog").finish_non_exhaustive()
        }
    }

    impl OneTableCatalog {
        fn current(&self) -> CurrentMetadata {
            self.current.lock().unwrap().clone()
        }

        /// Make `change` to the table's metadata its next version, the way
        /// another engine alters a table.
        fn commit_change(&self, change: impl FnOnce(TableMetadataBuilder) -> TableMetadataBuilder) {
            let mut current = self.current.lock().unwrap();
            let builder = TableMetadata::clone(&current.metadata)
                .into_builder(Some(current.metadata_location.clone()));
            let metadata = change(builder).build().unwrap().metadata;
            let (metadata_dir, _) = current.metadata_location.rsplit_once('/').unwrap();
            *current = CurrentMetadata {
                metadata_location: format!(
                    "{metadata_dir}/{}",
                    metadata_file_name(metadata.metadata_log().len())
                ),
                metadata: Arc::new(metadata),
            };
        }

        /// The table `table` names at the version the catalog holds.
        fn build_current_table(&self, table: &TableIdent) -> iceberg::Result<Table> {
            let current = self.current();
            Table::builder()
                .metadata(current.metadata)
                .metadata_location(current.metadata_location)
                .identifier(table.clone())
                .file_io(FileIO::new_with_memory())
                .runtime(Runtime::try_current()?)
                .build()
        }
    }

    fn metadata_file_name(version: usize) -> String {
        format!("{version:05}-{}.metadata.json", Uuid::new_v4())
    }

    #[async_trait]
    impl Catalog for OneTableCatalog {
        async fn load_table(&self, table: &TableIdent) -> iceberg::Result<Table> {
            self.loads_asked.fetch_add(1, Ordering::SeqCst);
            self.build_current_table(table)
        }

        async fn update_table(&self, commit: TableCommit) -> iceberg::Result<Table> {
            self.updates_asked.fetch_add(1, Ordering::SeqCst);
            if let Some(change) = self.racing_change.lock().unwrap().take() {
                self.commit_change(change);
            }
            if let Some(kind) = *self.update_failure.lock().unwrap() {
                return Err(iceberg::Error::new(kind, "the update failed"));
            }
            let held = self.build_current_table(commit.identifier())?;
            let committed = commit.apply(held)?;
            *self.current.lock().unwrap() = CurrentMetadata {
                metadata_location: committed.metadata_location_result()?.to_string(),
                metadata: committed.metadata_ref(),
            };
            Ok(committed)
        }

        async fn list_namespaces(
            &self,
            _parent: Option<&NamespaceIdent>,
        ) -> iceberg::Result<Vec<NamespaceIdent>> {
            unimplemented!()
        }

        async fn create_namespace(
            &self,
            _namespace: &NamespaceIdent,
            _properties: HashMap<String, String>,
        ) -> iceberg::Result<Namespace> {
            unimplemented!()
        }

        async fn get_namespace(&self, _namespace: &NamespaceIdent) -> iceberg::Result<Namespace> {
            unimplemented!()
        }

        async fn namespace_exists(&self, _namespace: &NamespaceIdent) -> iceberg::Result<bool> {
            unimplemented!()
        }

        async fn update_namespace(
            &self,
            _namespace: &NamespaceIdent,
            _properties: HashMap<String, String>,
        ) -> iceberg::Result<()> {
            unimplemented!()
        }

        async fn drop_namespace(&self, _namespace: &NamespaceIdent) -> iceberg::Result<()> {
            unimplemented!()
        }

        async fn list_tables(
            &self,
            _namespace: &NamespaceIdent,
        ) -> iceberg::Result<Vec<TableIdent>> {
            unimplemented!()
        }

        async fn create_table(
            &self,
            _namespace: &NamespaceIdent,
            _creation: TableCreation,
        ) -> iceberg::Result<Table> {
            unimplemented!()
        }

        async fn drop_table(&self, _table: &TableIdent) -> iceberg::Result<()> {
            unimplemented!()
        }

        async fn purge_table(&self, _table: &TableIdent) -> iceberg::Result<()> {
            unimplemented!()
        }

        async fn table_exists(&self, _table: &TableIdent) -> iceberg::Result<bool> {
            unimplemented!()
        }

        async fn rename_table(&self, _src: &TableIdent, _dest: &TableIdent) -> iceberg::Result<()> {
            unimplemented!()
        }

        async fn register_table(
            &self,
            _table: &TableIdent,
            _metadata_location: String,
        ) -> iceberg::Result<Table> {
            unimplemented!()
        }
    }

    fn test_pool() -> DataFlowDispatcher {
        static POOL: OnceLock<Dispatch> = OnceLock::new();
        POOL.get_or_init(|| Dispatch::spin_up(2, 64, None))
            .dispatcher()
            .clone()
    }

    struct Fixture {
        _directory: tempfile::TempDir,
        catalog: Arc<OneTableCatalog>,
        location: TableLocation,
    }

    /// A table of one column `v` with no snapshot, in a catalog, and the
    /// table as a writer loaded it from there.
    fn open_table() -> (Fixture, LoadedTable) {
        let directory = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(directory.path()).unwrap());
        let location =
            TableLocation::in_datastore_store("main.t", Arc::clone(&store), ObjectPath::new("t"))
                .unwrap();
        let name = SchemaQualifiedTableName::new("db", "t");
        let shape = TableShape {
            table_uuid: Uuid::new_v4(),
            columns: &[Column {
                name: "v".to_string(),
                col_type: Type::Int64,
            }],
            partition_by: &[],
            sort_by: &[],
            adopted: Vec::new(),
        };
        let created = LoadedTable::create(&test_pool(), location.clone(), &name, shape).unwrap();
        let catalog = Arc::new(OneTableCatalog {
            current: Mutex::new(CurrentMetadata {
                metadata_location: location.metadata_uri(&metadata_file_name(0)),
                metadata: Arc::clone(created.metadata()),
            }),
            update_failure: Mutex::new(None),
            racing_change: Mutex::new(None),
            loads_asked: AtomicUsize::new(0),
            updates_asked: AtomicUsize::new(0),
        });
        let table = LoadedTable::load(
            &test_pool(),
            Arc::new(RestMetadataStore::new(catalog.clone(), &name)),
            location.clone(),
            &name,
        )
        .unwrap();
        let fixture = Fixture {
            _directory: directory,
            catalog,
            location,
        };
        (fixture, table)
    }

    /// An entry for the file `name`, holding one row whose `v` is 1.
    fn entry(name: &str) -> DataFileEntry {
        let bound: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        DataFileEntry {
            file: FileRef {
                path: ObjectPath::new(format!("{name}.parquet")),
                size: 10,
            },
            partition: None,
            stats: Some(Arc::new(FileStats {
                num_records: Some(1),
                min_values: HashMap::from([("v".to_string(), bound.clone())]),
                max_values: HashMap::from([("v".to_string(), bound)]),
                null_counts: HashMap::from([("v".to_string(), 0)]),
            })),
            sort_order_id: None,
            first_row_id: None,
        }
    }

    /// How many manifests and manifest lists the table's directory holds.
    fn count_avro_objects(location: &TableLocation) -> usize {
        location
            .store()
            .list(&location.metadata_dir())
            .unwrap()
            .objects
            .iter()
            .filter(|object| object.file.path.as_str().ends_with(".avro"))
            .count()
    }

    #[test]
    fn a_commit_built_on_the_catalogs_current_version_is_its_successor() {
        let (fixture, table) = open_table();

        let committed = table
            .commit(&[], vec![entry("a")])
            .unwrap()
            .expect("no other writer");

        let current = fixture.catalog.current();
        assert_eq!(committed.metadata_location(), current.metadata_location);
        assert_eq!(committed.entries().unwrap().len(), 1);
        assert_eq!(
            current
                .metadata
                .metadata_log()
                .last()
                .unwrap()
                .metadata_file,
            table.metadata_location()
        );
    }

    #[test]
    fn a_commit_asks_the_catalog_for_one_update_and_no_load() {
        let (fixture, table) = open_table();
        let loads_before = fixture.catalog.loads_asked.load(Ordering::SeqCst);

        table
            .commit(&[], vec![entry("a")])
            .unwrap()
            .expect("no other writer");

        let loads = fixture.catalog.loads_asked.load(Ordering::SeqCst) - loads_before;
        assert_eq!(loads, 0);
        assert_eq!(fixture.catalog.updates_asked.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_commit_built_before_another_writers_snapshot_is_a_lost_race() {
        let (fixture, table) = open_table();
        table
            .commit(&[], vec![entry("a")])
            .unwrap()
            .expect("no other writer");
        let winner = fixture.catalog.current();
        let objects_of_winner = count_avro_objects(&fixture.location);

        let committed = table.commit(&[], vec![entry("b")]).unwrap();

        assert!(committed.is_none());
        let current = fixture.catalog.current();
        assert_eq!(current.metadata_location, winner.metadata_location);
        assert_eq!(count_avro_objects(&fixture.location), objects_of_winner);
    }

    #[test]
    fn a_commit_whose_requirements_the_catalog_refuses_is_a_lost_race_asked_once() {
        let (fixture, table) = open_table();
        let recreate: MetadataChange = Box::new(|builder| builder.assign_uuid(Uuid::new_v4()));
        *fixture.catalog.racing_change.lock().unwrap() = Some(recreate);

        let committed = table.commit(&[], vec![entry("a")]).unwrap();

        assert!(committed.is_none());
        assert_eq!(fixture.catalog.updates_asked.load(Ordering::SeqCst), 1);
        assert_eq!(count_avro_objects(&fixture.location), 0);
    }

    #[test]
    fn a_property_set_by_another_writer_before_the_update_is_in_the_committed_table() {
        let (fixture, table) = open_table();
        let set_owner: MetadataChange = Box::new(|builder| {
            builder
                .set_properties(HashMap::from([("owner".to_string(), "spark".to_string())]))
                .unwrap()
        });
        *fixture.catalog.racing_change.lock().unwrap() = Some(set_owner);

        let committed = table
            .commit(&[], vec![entry("a")])
            .unwrap()
            .expect("no writer moved `main`");

        assert_eq!(committed.metadata().properties()["owner"], "spark");
        assert_eq!(committed.entries().unwrap().len(), 1);
    }

    #[test]
    fn a_commit_that_lost_the_race_lands_once_built_on_the_winner() {
        let (fixture, table) = open_table();
        let winner = table
            .commit(&[], vec![entry("a")])
            .unwrap()
            .expect("no other writer");

        let lost = table.commit(&[], vec![entry("b")]).unwrap();
        let refreshed = table.refresh().unwrap().expect("the winner's version");
        let committed = refreshed
            .commit(&[], vec![entry("b")])
            .unwrap()
            .expect("no other writer");

        assert!(lost.is_none());
        assert_eq!(refreshed.metadata_location(), winner.metadata_location());
        assert_eq!(committed.entries().unwrap().len(), 2);
        let current = fixture.catalog.current();
        assert_eq!(current.metadata.snapshots().count(), 2);
    }

    #[test]
    fn a_commit_the_catalog_did_not_answer_for_keeps_its_objects_and_says_so() {
        let (fixture, table) = open_table();
        *fixture.catalog.update_failure.lock().unwrap() = Some(ErrorKind::Unexpected);

        let Err(error) = table.commit(&[], vec![entry("a")]) else {
            panic!("the catalog answered a server error");
        };

        assert!(error.may_have_committed(), "{error}");
        assert_eq!(count_avro_objects(&fixture.location), 2);
    }

    #[test]
    fn a_commit_the_catalog_refuses_for_a_dropped_table_fails_and_leaves_no_objects() {
        let (fixture, table) = open_table();
        *fixture.catalog.update_failure.lock().unwrap() = Some(ErrorKind::TableNotFound);

        let Err(error) = table.commit(&[], vec![entry("a")]) else {
            panic!("the catalog answered that the table is gone");
        };

        assert!(!error.may_have_committed(), "{error}");
        assert_eq!(count_avro_objects(&fixture.location), 0);
    }
}
