//! What this crate's integration tests share: one worker pool, a table of one
//! `Int64` column `v` in a directory of its own whose metadata versions are
//! kept in memory, the entries and the Parquet files it is filled with, and
//! which of its files a scan read.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use catalog::datastore::DatastoreTableMetadata;
use dispatch::io::{LocalFile, OpenFile};
use dispatch::{DataFlowDispatcher, Dispatch};
use iceberg::spec::{TableMetadata, TableMetadataBuilder, Transform, UnboundPartitionSpec};
use iceberg_tables::commit::{SnapshotExpiry, TableCreation};
use iceberg_tables::files::data_files::DataFileEntry;
use iceberg_tables::maintenance::retention::SNAPSHOT_RETENTION_PROPERTY;
use iceberg_tables::test_support::{Faults, FaultyStore, MemoryMetadataStore};
use iceberg_tables::{Error, IndexedFile, MetadataFile, Table, TableLocation};
use object_storage::{DataFileLocation, FileRef, LocalStore, ObjectPath, ObjectStore};
use parquet::arrow::ArrowWriter;
use parquet_engine::{FileStats, PartitionValues};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::Type;
use table_file_index::IndexedRecord;
use uuid::Uuid;

pub const HOUR_MS: i64 = 60 * 60 * 1000;

/// One worker pool shared by every test in the binary, for the reads a
/// load makes over it.
pub fn test_pool() -> DataFlowDispatcher {
    static POOL: OnceLock<Dispatch> = OnceLock::new();
    POOL.get_or_init(|| Dispatch::spin_up(2, 64, None))
        .dispatcher()
        .clone()
}

pub fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

/// The wall clock at a moment no commit is dated at: every commit before
/// the call is dated earlier and every commit after it later.
pub fn pause_between_commits() -> i64 {
    std::thread::sleep(std::time::Duration::from_millis(5));
    let paused_at = now_unix_ms();
    std::thread::sleep(std::time::Duration::from_millis(5));
    paused_at
}

/// The name every fixture's table is known by.
pub fn table_name() -> SchemaQualifiedTableName {
    SchemaQualifiedTableName::new("main", "t1")
}

pub fn column(name: &str, col_type: Type) -> Column {
    Column {
        name: name.to_string(),
        col_type,
    }
}

pub fn int_scalar(value: i64) -> Scalar<ArrayRef> {
    Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef)
}

/// The filter `v = value`.
pub fn v_equals(value: i64) -> Vec<Expression> {
    vec![Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(int_scalar(value))),
        compare_type: CompareType::Equal,
        return_type: Type::Boolean,
    })]
}

/// A table in a directory of its own, its metadata versions kept in
/// memory.
pub struct Fixture {
    _directory: tempfile::TempDir,
    pub location: TableLocation,
    pub metadata_store: Arc<MemoryMetadataStore>,
}

impl Fixture {
    /// A directory no table was created in yet.
    pub fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(directory.path()).unwrap());
        let location =
            TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t1")).unwrap();
        Self {
            _directory: directory,
            metadata_store: MemoryMetadataStore::new(location.clone()),
            location,
        }
    }

    /// The table `metadata` describes, in this directory, for a test of
    /// what a table answers from its metadata document: the document is
    /// published nowhere, and of the table's objects only the manifest
    /// list of its current snapshot, when it has one, is read.
    pub fn build_table_at(&self, metadata: TableMetadata) -> Table {
        self.try_build_table_at(metadata).unwrap()
    }

    /// [`build_table_at`](Self::build_table_at), for metadata a table may
    /// be refused over.
    pub fn try_build_table_at(&self, metadata: TableMetadata) -> iceberg_tables::Result<Table> {
        Table::at(
            &test_pool(),
            self.metadata_store.clone(),
            self.location.clone(),
            &table_name(),
            MetadataFile {
                location: String::new(),
                metadata: Arc::new(metadata),
            },
            None,
        )
    }

    /// A table partitioned by `v`, created in this directory under a UUID
    /// and a metadata store of its own: what a table created after a drop
    /// of the one the directory held is.
    pub fn create_table_again(&self) -> Table {
        Table::create(
            &test_pool(),
            MemoryMetadataStore::new(self.location.clone()),
            self.location.clone(),
            &table_name(),
            TableCreation {
                table_uuid: Uuid::new_v4(),
                columns: &columns(),
                partition_by: &["v".to_string()],
                sort_by: &[],
                adopted: Vec::new(),
            },
        )
        .unwrap()
    }

    /// Delete the data manifests of `table` from the store.
    pub fn delete_data_manifests(&self, table: &Table) {
        for manifest in table.manifest_list().data_manifests() {
            self.delete_manifest(&manifest.manifest_path);
        }
    }

    pub fn delete_manifest(&self, manifest_path: &str) {
        let location = &self.location;
        let key = location.data_file_key(&location.file_path_for_uri(manifest_path).unwrap());
        location.store().delete(&key).unwrap();
    }

    /// Write a Parquet file of ten rows `v = value` where the table names
    /// the file of `entry`, and hand the entry back with the size of what
    /// was written.
    pub fn write_data_file(&self, mut entry: DataFileEntry, value: i64) -> DataFileEntry {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));
        let rows: ArrayRef = Arc::new(Int64Array::from(vec![value; 10]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![rows]).unwrap();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        entry.file.size = bytes.len() as u64;
        let key = self.location.data_file_key(&entry.file.path);
        self.location.store().put(&key, &bytes).unwrap();
        entry
    }

    /// Write `name`, a Parquet file of ten rows `v = value`, into the
    /// table's directory, and hand back its entry in the partition
    /// `v = value`.
    pub fn write_file(&self, name: &str, value: i64) -> DataFileEntry {
        self.write_data_file(partition_entry(name, value, 0), value)
    }

    /// Put in the place of the data file `name` another file of its size
    /// that holds no footer, so whatever reads the footer of `name` from
    /// now on fails.
    pub fn replace_with_garbage(&self, name: &str) {
        let path = ObjectPath::new(format!("{name}.parquet"));
        let key = self.location.data_file_key(&path);
        let store = self.location.store();
        let size = store.get(&key).unwrap().unwrap().len();
        store.delete(&key).unwrap();
        store.put(&key, &vec![0; size]).unwrap();
    }
}

/// The columns of the table every fixture creates.
pub fn columns() -> Vec<Column> {
    vec![column("v", Type::Int64)]
}

pub fn create_table(partition_by: &[String]) -> (Fixture, Table) {
    create_table_laid_out_by(partition_by, &[])
}

pub fn create_table_laid_out_by(partition_by: &[String], sort_by: &[String]) -> (Fixture, Table) {
    create_table_of(&columns(), partition_by, sort_by)
}

/// A table of `columns`, partitioned by `partition_by` and sorted by
/// `sort_by`, created in a directory of its own.
pub fn create_table_of(
    columns: &[Column],
    partition_by: &[String],
    sort_by: &[String],
) -> (Fixture, Table) {
    let fixture = Fixture::new();
    let table = Table::create(
        &test_pool(),
        fixture.metadata_store.clone(),
        fixture.location.clone(),
        &table_name(),
        TableCreation {
            table_uuid: Uuid::new_v4(),
            columns,
            partition_by,
            sort_by,
            adopted: Vec::new(),
        },
    )
    .unwrap();
    (fixture, table)
}

/// A partition field: its source column, its name, and its transform.
pub type SpecField = (&'static str, &'static str, Transform);

/// Make `fields` the spec the table of `fixture` is written under from now
/// on, and load the table at that version.
pub fn repartition(fixture: &Fixture, fields: &[SpecField]) -> Table {
    let current = load(fixture);
    let schema = current.metadata().current_schema();
    publish_changed(fixture, |builder| {
        let mut spec = UnboundPartitionSpec::builder();
        for (source, name, transform) in fields {
            let source_id = schema.field_id_by_name(source).unwrap();
            spec = spec
                .add_partition_field(source_id, *name, *transform)
                .unwrap();
        }
        builder
            .add_partition_spec(spec.build())
            .unwrap()
            .set_default_partition_spec(-1)
            .unwrap()
    })
}

/// `table` as the catalog describes it.
pub fn describe(table: &Table) -> DatastoreTableMetadata {
    let name_file = |file: &IndexedFile| file.facts().file().path.as_str().to_string();
    table
        .describe(table_name(), "t1".to_string(), &name_file)
        .unwrap()
}

/// The table partitioned by its column `v`.
pub fn create_partitioned_table() -> (Fixture, Table) {
    create_table(&["v".to_string()])
}

/// The table as another copy loads it.
pub fn load(fixture: &Fixture) -> Table {
    Table::load(
        &test_pool(),
        fixture.metadata_store.clone(),
        fixture.location.clone(),
        &table_name(),
    )
    .unwrap()
}

/// The fixture's table location over a store that fails the way `faults`
/// says.
pub fn open_faulty_location(fixture: &Fixture, faults: &Arc<Faults>) -> TableLocation {
    let store: Arc<dyn ObjectStore> = Arc::new(FaultyStore {
        inner: Arc::clone(fixture.location.store()),
        faults: Arc::clone(faults),
    });
    TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t1")).unwrap()
}

/// The fixture's table as a writer loads it whose store fails the way
/// `faults` says, and counts the whole objects read from it.
pub fn load_over_faulty_store(fixture: &Fixture, faults: &Arc<Faults>) -> Table {
    let location = open_faulty_location(fixture, faults);
    Table::load(
        &test_pool(),
        fixture.metadata_store.clone(),
        location,
        &table_name(),
    )
    .unwrap()
}

/// The table after `change` to its current metadata is published as
/// its next version, the way another tool alters a table.
pub fn publish_changed(
    fixture: &Fixture,
    change: impl FnOnce(TableMetadataBuilder) -> TableMetadataBuilder,
) -> Table {
    fixture.metadata_store.publish_changed(change);
    load(fixture)
}

pub fn with_retention_of_one_millisecond(fixture: &Fixture) -> Table {
    publish_changed(fixture, |builder| {
        builder
            .set_properties(HashMap::from([(
                SNAPSHOT_RETENTION_PROPERTY.to_string(),
                "1".to_string(),
            )]))
            .unwrap()
    })
}

/// `table` with the tag `release` on its current snapshot.
pub fn with_tag_on_current_snapshot(fixture: &Fixture, table: &Table) -> Table {
    use iceberg::spec::{SnapshotReference, SnapshotRetention};
    let tagged_snapshot_id = table.metadata().current_snapshot_id().unwrap();
    publish_changed(fixture, |builder| {
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

/// `table` with a second branch, `audit`, on its current snapshot.
pub fn with_branch_on_current_snapshot(fixture: &Fixture, table: &Table) -> Table {
    use iceberg::spec::{SnapshotReference, SnapshotRetention};
    let branched_snapshot_id = table.metadata().current_snapshot_id().unwrap();
    publish_changed(fixture, |builder| {
        builder
            .set_ref(
                "audit",
                SnapshotReference::new(
                    branched_snapshot_id,
                    SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                ),
            )
            .unwrap()
    })
}

/// `table` with one more snapshot that no reference points at: a child of
/// its current snapshot that was written and never made current, naming
/// the current snapshot's manifest list.
pub fn with_staged_snapshot(fixture: &Fixture, table: &Table) -> Table {
    use iceberg::spec::{Operation, Snapshot, Summary};
    let metadata = table.metadata();
    let current = metadata.current_snapshot().unwrap();
    let staged = Snapshot::builder()
        .with_snapshot_id(current.snapshot_id().wrapping_add(1))
        .with_parent_snapshot_id(Some(current.snapshot_id()))
        .with_sequence_number(metadata.last_sequence_number() + 1)
        .with_timestamp_ms(now_unix_ms())
        .with_manifest_list(current.manifest_list().to_string())
        .with_summary(Summary {
            operation: Operation::Append,
            additional_properties: HashMap::new(),
        })
        .with_schema_id(metadata.current_schema_id())
        .with_row_range(metadata.next_row_id(), 0)
        .build();
    publish_changed(fixture, |builder| builder.add_snapshot(staged).unwrap())
}

/// The fixture's table set to merge its data manifests into one as soon
/// as a commit lists two of them.
pub fn lower_manifest_merge_threshold(fixture: &Fixture) -> Table {
    publish_changed(fixture, |builder| {
        builder
            .set_properties(HashMap::from([(
                "commit.manifest.min-count-to-merge".to_string(),
                "2".to_string(),
            )]))
            .unwrap()
    })
}

/// Whether `error` refuses a table because its history is not linear,
/// naming the issue that tracks branches and tags.
pub fn is_non_linear_history_error(error: &Error) -> bool {
    matches!(error, Error::NonLinearHistory { .. })
        && error
            .to_string()
            .contains("https://github.com/pivotlake/pivot/issues/101")
}

/// An entry for `name` holding `rows` rows of `v` in `low..=high`.
pub fn entry(name: &str, rows: i64, low: i64, high: i64) -> DataFileEntry {
    let bound = |value: i64| -> ArrayRef { Arc::new(Int64Array::from(vec![value])) };
    DataFileEntry {
        file: FileRef {
            path: ObjectPath::new(format!("{name}.parquet")),
            size: 10,
        },
        partition: None,
        partition_spec_id: None,
        stats: Some(Arc::new(FileStats {
            num_records: Some(rows),
            min_values: HashMap::from([("v".to_string(), bound(low))]),
            max_values: HashMap::from([("v".to_string(), bound(high))]),
            null_counts: HashMap::from([("v".to_string(), 0)]),
        })),
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    }
}

/// An entry for `name`, a file of ten rows in the partition `v = value`, of
/// which `nulls` hold no value.
pub fn partition_entry(name: &str, value: i64, nulls: i64) -> DataFileEntry {
    let mut entry = entry(name, 10, value, value);
    entry.partition = Some(PartitionValues::from([(
        "v".to_string(),
        int_scalar(value),
    )]));
    Arc::get_mut(entry.stats.as_mut().unwrap())
        .unwrap()
        .null_counts
        .insert("v".to_string(), nulls);
    entry
}

pub fn append(table: &Table, entries: Vec<DataFileEntry>) -> Table {
    table
        .commit(&[], entries)
        .unwrap()
        .expect("no other writer")
}

/// `table` after a commit that replaces its live files at `removed` with
/// one file named `added`, as a compaction does.
pub fn replace(table: &Table, removed: &[&str], added: &str) -> Table {
    let removed: Vec<DataFileEntry> = removed.iter().map(|path| live_entry(table, path)).collect();
    table
        .commit(&removed, vec![entry(added, 1, 1, 1)])
        .unwrap()
        .expect("no other writer")
}

/// The table's names for the objects `expiry` leaves to delete, sorted.
pub fn list_expired_objects(expiry: &SnapshotExpiry) -> Vec<String> {
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

/// Which version of the fixture's table, counted from 1, `table` is at.
pub fn version(fixture: &Fixture, table: &Table) -> u64 {
    fixture
        .metadata_store
        .find_version_number(table.metadata_location())
        .expect("the table is at a version of its store")
}

pub fn avro_objects(location: &TableLocation) -> usize {
    location
        .store()
        .list(&location.metadata_dir())
        .unwrap()
        .objects
        .iter()
        .filter(|object| object.file.path.as_str().ends_with(".avro"))
        .count()
}

/// The live files' names, sorted: a table lists new manifests before
/// carried ones, so the order is not what a test asserts on.
pub fn paths(table: &Table) -> Vec<String> {
    let mut paths = paths_in_order(table);
    paths.sort_unstable();
    paths
}

/// The live files' names in the order the table holds them.
pub fn paths_in_order(table: &Table) -> Vec<String> {
    table
        .entries()
        .unwrap()
        .iter()
        .map(|entry| entry.file.path.to_string())
        .collect()
}

pub fn live_entry(table: &Table, path: &str) -> DataFileEntry {
    table
        .entries()
        .unwrap()
        .into_iter()
        .find(|entry| entry.file.path.as_str() == path)
        .unwrap_or_else(|| panic!("{path} is live"))
}

/// How many manifests, data and delete ones, `table` lists.
pub fn count_manifests(table: &Table) -> usize {
    let list = table.manifest_list();
    list.data_manifests().chain(list.delete_manifests()).count()
}

/// The versions `table`'s metadata log names, in the log's order.
pub fn logged_versions(fixture: &Fixture, table: &Table) -> Vec<u64> {
    table
        .metadata()
        .metadata_log()
        .iter()
        .map(|entry| {
            fixture
                .metadata_store
                .find_version_number(&entry.metadata_file)
                .expect("the log names versions of the table")
        })
        .collect()
}

/// The names of the files a scan of `table` over `filters` reads, in scan
/// order. Every file the scan selects is read for its footer, so each is one
/// a test wrote into the table's directory, and is told by the file its row
/// groups read from.
pub fn selected(table: &Table, filters: &[Expression]) -> Vec<String> {
    let location = table.location();
    let store = location.store();
    let written: Vec<(String, OpenFile)> = store
        .list(location.directory())
        .unwrap()
        .objects
        .iter()
        .map(|object| object.file.path.to_string())
        .filter(|name| name.ends_with(".parquet"))
        .map(|name| {
            let key = location.data_file_key(&ObjectPath::new(name.as_str()));
            let DataFileLocation::Local(path) = store.source(&key).unwrap() else {
                panic!("a local store reads from a path");
            };
            let file = LocalFile::new(std::fs::File::open(path).unwrap()).unwrap();
            (name, OpenFile::Local(file))
        })
        .collect();
    let selection = table.select_scan(filters).unwrap();
    selection
        .row_groups
        .row_groups()
        .iter()
        .map(|row_group| {
            let (name, _) = written
                .iter()
                .find(|(_, file)| *file == row_group.open_file)
                .expect("a scan reads files of the table's directory");
            name.clone()
        })
        .collect()
}
