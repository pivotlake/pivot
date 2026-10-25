//! This datastore's background maintenance settings, and what it hands the
//! shared compacter: itself as the host that names its tables and lands a
//! swap, and each [`CatalogTable`] as the copy a round works on.

use std::time::Duration;

use object_storage::{FileRef, ObjectPath};
use planner::catalog::SchemaQualifiedTableName;
use table_compaction::CompactionHost;
use uuid::Uuid;

use crate::{CatalogTable, PivotlakeDatastore, Result};

pub use table_compaction::{
    CompacterHandle, CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_PARALLELISM,
    DEFAULT_COMPACT_POLL, DEFAULT_LAYOUT_CLIQUE_SIZE, DEFAULT_MIN_FILES_TO_MERGE, TableFile,
    default_merge_target_bytes,
};

/// Default cadence for reloading the tables from the store. This bounds how
/// stale a query's view of externally committed data in a shared remote store
/// is, so it trades freshness against the store's listing traffic.
pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

/// What background maintenance a [`PivotlakeDatastore`] runs for itself once opened.
/// The datastore spawns its own tasks from this on the ambient tokio runtime;
/// they stop when its dispatch pool begins shutting down.
#[derive(Clone)]
pub struct MaintenanceConfig {
    /// How often the datastore reloads its tables from the store (new
    /// versions and new files' footers), so externally committed data in a
    /// shared remote store becomes visible to queries.
    pub refresh_interval: Duration,
    /// Compaction settings, or `None` to run no compacter (a read-only server
    /// or a remote-store deployment where another process owns compaction).
    pub compaction: Option<CompactionConfig>,
    /// Vacuum settings, or `None` to run no vacuumer (a read-only server, or a
    /// remote-store deployment where another process owns physical cleanup).
    /// Deletes tombstoned data files and superseded commit JSONs past their
    /// retention.
    pub vacuum: Option<crate::vacuum::VacuumConfig>,
}

impl CompactionHost for PivotlakeDatastore {
    type Table = CatalogTable;
    type Error = crate::Error;

    fn tables(&self) -> Vec<(SchemaQualifiedTableName, CatalogTable)> {
        PivotlakeDatastore::tables(self)
    }

    fn table_by_id(&self, id: &Uuid) -> Option<CatalogTable> {
        self.table_handle_by_id(id)
    }

    /// The swap goes through the datastore's one in-process write path, under
    /// the table's commit lock.
    fn commit_swap(&self, id: Uuid, removed: &[ObjectPath], added: Vec<TableFile>) -> Result<()> {
        self.commit_to_table(id, removed, added, false)
    }
}

/// Merge `inputs` of the table with identity `id` into fresh target-sized
/// files and swap them in for the inputs, in one commit labelled a
/// rearrangement: [`table_compaction::compact_table_files`] over this
/// datastore. Blocking store I/O, so callers run it on the blocking pool.
pub fn compact_table_files(
    datastore: &PivotlakeDatastore,
    id: Uuid,
    inputs: &[FileRef],
    target_rows_per_group: usize,
    max_output_file_size: u64,
) -> Result<Vec<TableFile>> {
    table_compaction::compact_table_files(
        datastore,
        id,
        inputs,
        target_rows_per_group,
        max_output_file_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::file_entry::scalar_values_equal;
    use crate::{CatalogTable, PivotlakeDatastore};
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
    use object_storage::ObjectPath;
    use parquet::arrow::ArrowWriter;
    use parquet_engine::ParquetTable;
    use planner::catalog::SchemaQualifiedTableName;
    use std::path::Path;
    use table_compaction::{
        CompactionTable, DEFAULT_MERGE_TARGET_MULTIPLIER, ROW_GROUP_ROWS, is_small_file,
        merge_files,
    };

    const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

    /// Write `values` as a small SNAPPY Parquet file (single Int64 `Timestamp`
    /// column) into `dir` under `file_name`. A table created over `dir` picks it
    /// up as a small input for the compacter to merge.
    fn write_parquet_file(dir: &Path, file_name: &str, values: Vec<i64>) {
        let rows = values.len().max(1);
        write_parquet_file_with_row_group_size(dir, file_name, values, rows);
    }

    /// A file whose rows are cut into row groups of `max_row_group_size`, so a
    /// test file of a few rows still carries the many row groups that layout
    /// selection counts.
    fn write_parquet_file_with_row_group_size(
        dir: &Path,
        file_name: &str,
        values: Vec<i64>,
        max_row_group_size: usize,
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "Timestamp",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(values)) as _],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        // SNAPPY, like every pivot-written file -- the decompressor expects it.
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .set_max_row_group_row_count(Some(max_row_group_size))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// A sorted range with incompressible non-key payloads. Two seeds write
    /// the same key range at almost the same byte size while retaining distinct
    /// rows, which models two 80%-full overlapping files without footer bytes
    /// dominating the test.
    fn write_wide_parquet_file(dir: &Path, file_name: &str, rows: usize, seed: u64) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            Field::new("Payload0", DataType::Int64, false),
            Field::new("Payload1", DataType::Int64, false),
            Field::new("Payload2", DataType::Int64, false),
            Field::new("Payload3", DataType::Int64, false),
        ]));
        let keys: Vec<i64> = (1..=rows as i64).collect();
        let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(keys))];
        for column in 0..4 {
            let payload: Vec<i64> = (0..rows)
                .map(|row| {
                    let mut value = seed ^ row as u64 ^ ((column as u64) << 48);
                    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    value ^= value >> 31;
                    value as i64
                })
                .collect();
            columns.push(Arc::new(Int64Array::from(payload)));
        }
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn write_partitioned_parquet_file(
        dir: &Path,
        file_name: &str,
        timestamps: Vec<i64>,
        partition: i64,
    ) {
        let row_count = timestamps.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            Field::new("Partition", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps)) as _,
                Arc::new(Int64Array::from(vec![partition; row_count])) as _,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// `CREATE TABLE <name> (Timestamp Int64) WITH (with_pre_existing_parquets = dir)`
    /// against `datastore`, the way the server would run it -- or, when `dir` is
    /// `None`, an empty table adopting nothing.
    fn create_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: Option<&Path>,
        sorted: bool,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let mut options = match dir {
            Some(dir) => std::collections::HashMap::from([(
                "with_pre_existing_parquets".to_string(),
                dir.to_str().unwrap().to_string(),
            )]),
            None => std::collections::HashMap::new(),
        };
        if sorted {
            options.insert("sort_by".to_string(), "Timestamp".to_string());
        }
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![Column {
                name: "Timestamp".to_string(),
                col_type: planner::types::Type::Int64,
            }],
            options,
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    fn create_wide_sorted_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: &Path,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![
                Column {
                    name: "Timestamp".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload0".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload1".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload2".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload3".to_string(),
                    col_type: planner::types::Type::Int64,
                },
            ],
            options: std::collections::HashMap::from([
                (
                    "with_pre_existing_parquets".to_string(),
                    dir.to_str().unwrap().to_string(),
                ),
                ("sort_by".to_string(), "Timestamp".to_string()),
            ]),
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    /// Create an empty table partitioned by its `Partition` column.
    fn create_partitioned_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![
                Column {
                    name: "Timestamp".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Partition".to_string(),
                    col_type: planner::types::Type::Int64,
                },
            ],
            options: std::collections::HashMap::from([(
                "partition_by".to_string(),
                "Partition".to_string(),
            )]),
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    fn partition_value(value: i64) -> crate::PartitionValues {
        std::collections::HashMap::from([(
            "Partition".to_string(),
            Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef),
        )])
    }

    /// Refresh `name` to the latest committed manifest (a writer evolves a
    /// cloned-out handle, so the datastore's own copy lags until a refresh), then
    /// return its current row groups.
    fn fresh_parquet(datastore: &PivotlakeDatastore, name: &str) -> Arc<ParquetTable> {
        let mut table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
            .expect("table exists");
        table.refresh().expect("manifest reload");
        table.build_scan_view(&[], &[]).expect("build scan view")
    }

    /// Drive one full compaction sweep to completion on a temporary runtime.
    fn run_one_sweep(target_bytes: u64, datastore: Arc<PivotlakeDatastore>) {
        run_one_sweep_with(
            target_bytes,
            DEFAULT_COMPACT_PARALLELISM,
            DEFAULT_LAYOUT_CLIQUE_SIZE,
            datastore,
        )
    }

    /// [`run_one_sweep`] with layout cliques kept to pairs, for scenarios
    /// where two files are the full clique.
    fn run_one_sweep_with_pair_layout(target_bytes: u64, datastore: Arc<PivotlakeDatastore>) {
        run_one_sweep_with(target_bytes, DEFAULT_COMPACT_PARALLELISM, 2, datastore)
    }

    /// [`run_one_sweep`] with the merges in flight and the layout merge width
    /// spelled out.
    fn run_one_sweep_with(
        target_bytes: u64,
        max_concurrent_merges: usize,
        layout_clique_size: usize,
        datastore: Arc<PivotlakeDatastore>,
    ) {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let tables = datastore.tables();
                let (compacter, actor) = CompacterHandle::new(
                    target_bytes,
                    default_merge_target_bytes(target_bytes),
                    DEFAULT_MIN_FILES_TO_MERGE,
                    max_concurrent_merges,
                    layout_clique_size,
                    None,
                    datastore,
                );
                let actor = tokio::spawn(actor.run());
                for (name, table) in tables {
                    compacter.compact(name, table, false).await.unwrap();
                }
                actor.abort();
            });
    }

    /// The size knobs one compaction command runs under.
    struct CompacterSizes {
        target_bytes: u64,
        merge_target_bytes: u64,
        min_files: usize,
        layout_clique_size: usize,
    }

    impl CompacterSizes {
        /// The defaults for `target_bytes`.
        fn defaults(target_bytes: u64) -> Self {
            Self {
                target_bytes,
                merge_target_bytes: default_merge_target_bytes(target_bytes),
                min_files: DEFAULT_MIN_FILES_TO_MERGE,
                layout_clique_size: DEFAULT_LAYOUT_CLIQUE_SIZE,
            }
        }

        /// The defaults with the small-file strategy switched off, so a
        /// scenario exercises layout optimization alone.
        fn without_small_files(target_bytes: u64) -> Self {
            Self {
                merge_target_bytes: u64::MAX,
                min_files: usize::MAX,
                ..Self::defaults(target_bytes)
            }
        }
    }

    fn run_command(
        sizes: CompacterSizes,
        datastore: Arc<PivotlakeDatastore>,
        name: SchemaQualifiedTableName,
        table: CatalogTable,
        final_sweep: bool,
    ) -> u64 {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let (compacter, actor) = CompacterHandle::new(
                    sizes.target_bytes,
                    sizes.merge_target_bytes,
                    sizes.min_files,
                    DEFAULT_COMPACT_PARALLELISM,
                    sizes.layout_clique_size,
                    None,
                    datastore,
                );
                let actor = tokio::spawn(actor.run());
                let sweeps = compacter.compact(name, table, final_sweep).await.unwrap();
                actor.abort();
                sweeps
            })
    }

    /// The command's snapshot is refreshed before selecting, so a file
    /// committed after it was captured still joins the merge.
    #[test]
    fn command_refreshes_the_supplied_table_snapshot() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file(&adopted, "a.parquet", vec![1]);
        write_parquet_file(&adopted, "b.parquet", vec![2]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            false,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let captured = datastore.table_handle(&name).unwrap();
        // Each file is small (under half the target), two of them stay under
        // the merge target (1.3 times it), three of them reach it: the merge
        // only happens once the refresh brings `c.parquet` in.
        let target = captured.file_refs()[0].size * 22 / 10;
        write_parquet_file(&adopted, "c.parquet", vec![3]);
        let mut latest = captured.clone();
        latest
            .append_data_file(
                ObjectPath::new("c.parquet"),
                &std::fs::read(adopted.join("c.parquet")).unwrap(),
                None,
            )
            .unwrap();
        datastore.refresh_from_store().unwrap();

        let sweeps = run_command(
            CompacterSizes::defaults(target),
            datastore.clone(),
            name.clone(),
            captured,
            false,
        );

        assert_eq!(sweeps, 2, "one merging pass and one that finds nothing");
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 1);
        dispatch.exit();
    }

    /// A round does not stop at one pair: it keeps merging, folding each
    /// committed merge into its view, until no large pair overlaps any more.
    #[test]
    fn a_round_keeps_merging_until_no_pair_overlaps() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![1_000, 1_100]);
        write_parquet_file(&adopted_dir, "d.parquet", vec![1_000, 1_100]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();

        run_one_sweep_with_pair_layout(target, datastore.clone());

        let files = datastore.table_files(&name).unwrap();
        assert!(
            files.iter().all(|file| !matches!(
                file.path.name(),
                "a.parquet" | "b.parquet" | "c.parquet" | "d.parquet"
            )),
            "both overlapping pairs were rewritten in one round: {files:?}"
        );
        dispatch.exit();
    }

    /// Three overlapping pairs are rewritten with three merges in flight. The
    /// reservations keep every merge off the others' inputs, so all swaps
    /// commit and every row survives.
    #[test]
    fn concurrent_merges_never_share_an_input() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        for (file, rows) in [
            ("a.parquet", vec![0, 100]),
            ("b.parquet", vec![0, 100]),
            ("c.parquet", vec![1_000, 1_100]),
            ("d.parquet", vec![1_000, 1_100]),
            ("e.parquet", vec![2_000, 2_100]),
            ("f.parquet", vec![2_000, 2_100]),
        ] {
            write_parquet_file(&adopted_dir, file, rows);
        }
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();

        run_one_sweep_with(target, 3, 2, datastore.clone());

        let files = datastore.table_files(&name).unwrap();
        assert!(
            files
                .iter()
                .all(|file| file.path.name().starts_with("pivot-")),
            "all three pairs were rewritten: {files:?}"
        );
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 12, "no input was merged twice or dropped");
        dispatch.exit();
    }

    #[test]
    fn final_command_bypasses_small_file_guards() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let encoded = tempfile::tempdir().unwrap();
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_partitioned_table(&datastore, dispatch.dispatcher(), "events");
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        for (file, partition) in [("a", 1), ("b", 1), ("c", 2), ("d", 2)] {
            let file = format!("{file}.parquet");
            write_partitioned_parquet_file(encoded.path(), &file, vec![partition], partition);
            table
                .append_data_file(
                    ObjectPath::new(&file),
                    &std::fs::read(encoded.path().join(&file)).unwrap(),
                    Some(partition_value(partition)),
                )
                .unwrap();
        }
        datastore.refresh_from_store().unwrap();
        let table = datastore.table_handle(&name).unwrap();
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);

        let routine_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            table.clone(),
            false,
        );
        let routine_files = datastore.table_handle(&name).unwrap().file_refs().len();
        let final_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            table,
            true,
        );

        assert_eq!(routine_sweeps, 1);
        assert_eq!(routine_files, 4);
        assert_eq!(final_sweeps, 3);
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 2);
        dispatch.exit();
    }

    /// The row-group rule is the whole layout bar: a pair merges when more
    /// than a fifth of each file's row groups overlaps the other file, and
    /// files that only reach each other's boundary value never merge, not even
    /// under the final command.
    #[test]
    fn layout_merges_need_overlapping_row_groups_on_both_sides() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        // One row per row group. The first two files share seven of their
        // twenty groups; the last two meet at 119 and share nothing.
        write_parquet_file_with_row_group_size(&adopted, "a.parquet", (0..20).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "b.parquet", (12..32).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "c.parquet", (100..120).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "d.parquet", (119..139).collect(), 1);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        // Keep every input out of the small-file path.
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap()
            .saturating_mul(2);

        run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::without_small_files(target)
            },
            datastore.clone(),
            name.clone(),
            table,
            false,
        );
        let after_routine = datastore.table_handle(&name).unwrap();
        let routine_files: Vec<String> = after_routine
            .file_refs()
            .iter()
            .map(|file| file.path.name().to_string())
            .collect();
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::without_small_files(target)
            },
            datastore.clone(),
            name.clone(),
            after_routine,
            true,
        );

        assert_eq!(routine_files.len(), 3);
        assert!(routine_files.contains(&"c.parquet".to_string()));
        assert!(routine_files.contains(&"d.parquet".to_string()));
        assert_eq!(
            final_sweeps, 1,
            "the final command finds nothing the routine round left behind"
        );
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 3);
        dispatch.exit();
    }

    /// End-to-end compaction: several registered small files merge into one, the
    /// datastore swaps to the merged file in one log commit, and it decodes back to
    /// all the original rows.
    #[test]
    fn compaction_merges_registered_files_and_swaps_datastore() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![1, 2, 3]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![4, 5]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![6, 7, 8, 9]);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            false,
        );
        assert_eq!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .len(),
            3
        );

        let total: u64 = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let target = (total as f64 / DEFAULT_MERGE_TARGET_MULTIPLIER).floor() as u64;
        run_one_sweep(target, datastore.clone());

        assert_eq!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .len(),
            1
        );
        let parquet = fresh_parquet(&datastore, "events");
        assert_eq!(parquet.row_groups().len(), 1);
        assert_eq!(parquet.row_groups()[0].num_rows, 9);
        assert!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .iter()
                .all(|f| f.path.as_str().starts_with("pivot-"))
        );

        dispatch.exit();
    }

    /// Small-file compaction uses the same output target as layout work. A
    /// three-file batch whose encoded row groups need two target-sized files is
    /// split rather than becoming one oversized replacement.
    #[test]
    fn small_file_compaction_splits_outputs_at_target() {
        const ROWS_PER_INPUT: usize = 64_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "a.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "b.parquet", ROWS_PER_INPUT, 29);
        write_wide_parquet_file(&adopted, "c.parquet", ROWS_PER_INPUT, 47);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs();
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        assert!(inputs.iter().all(|file| is_small_file(file.size, target)));
        let before_version = table.version();

        run_one_sweep(target, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(compacted.version(), before_version + 1);
        let outputs = compacted.file_refs();
        assert_eq!(outputs.len(), 2, "target={target}, outputs={outputs:?}");
        assert!(
            outputs
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let row_count: i64 = compacted
            .build_scan_view(&[], &[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (3 * ROWS_PER_INPUT) as i64);
        dispatch.exit();
    }

    #[test]
    fn layout_optimization_merges_the_highest_overlap_large_pair() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![1_000, 1_100]);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();
        run_one_sweep_with_pair_layout(target, datastore.clone());

        let files = datastore.table_files(&name).unwrap();
        assert_eq!(files.len(), 2);
        assert!(
            files.iter().any(|file| file.path.name() == "c.parquet"),
            "the disjoint file is not part of the chosen pair"
        );
        assert!(
            files
                .iter()
                .all(|file| file.path.name() != "a.parquet" && file.path.name() != "b.parquet")
        );

        dispatch.exit();
    }

    /// Two 80%-full files are layout work, not small-file work. Their globally
    /// sorted rewrite is too large for one target file, so it becomes two
    /// balanced, disjoint ranges which neither strategy selects again.
    #[test]
    fn overlapping_eighty_percent_files_become_two_stable_range_files() {
        // Exactly four normal compaction row groups across the two inputs, so
        // the encoded result has a natural balanced boundary for two files.
        const ROWS_PER_INPUT: usize = 256_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "left.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "right.parquet", ROWS_PER_INPUT, 29);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs();
        assert_eq!(inputs.len(), 2);
        let largest_input = inputs.iter().map(|file| file.size).max().unwrap();
        let target = largest_input.saturating_mul(5).div_ceil(4);
        for input in &inputs {
            let fullness = input.size as f64 / target as f64;
            assert!(
                (0.75..=0.81).contains(&fullness),
                "input {} is {:.1}% of target",
                input.path,
                fullness * 100.0
            );
        }

        // Verify that an effectively unbounded encoding, not merely the sum of the
        // input sizes, requires two files under this target.
        let unbounded = merge_files(&table, &inputs, ROW_GROUP_ROWS, u64::MAX).unwrap();
        assert_eq!(unbounded.len(), 1);
        let unbounded_size = unbounded[0].file_ref().size;
        for file in unbounded {
            table.delete_data_file(&file.file_ref().path).unwrap();
        }
        assert!(unbounded_size > target);
        assert!(unbounded_size <= target * 2);

        let before_version = table.version();
        run_one_sweep_with_pair_layout(target, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(compacted.version(), before_version + 1);
        let outputs = compacted.file_refs();
        assert_eq!(
            outputs.len(),
            2,
            "target={target}, unbounded={unbounded_size}, outputs={outputs:?}"
        );
        assert!(outputs.iter().all(|file| file.size <= target));
        assert!(outputs.iter().all(|file| !is_small_file(file.size, target)));
        assert!(
            outputs
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );

        let mut ranges: Vec<(i64, i64)> = compacted
            .entries()
            .iter()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        assert_eq!(ranges.first().unwrap().0, 1);
        assert_eq!(ranges.last().unwrap().1, ROWS_PER_INPUT as i64);
        assert!(ranges[0].1 < ranges[1].0, "ranges are disjoint: {ranges:?}");

        let row_count: i64 = compacted
            .build_scan_view(&[], &[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (2 * ROWS_PER_INPUT) as i64);

        let log = db
            .path()
            .join(compacted.location())
            .join("_delta_log")
            .join(format!("{:020}.json", compacted.version()));
        let actions: Vec<serde_json::Value> = std::fs::read_to_string(log)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("remove").is_some())
                .count(),
            2
        );
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("add").is_some())
                .count(),
            2
        );

        let stable_version = compacted.version();
        let stable_paths: std::collections::HashSet<_> =
            outputs.iter().map(|file| file.path.clone()).collect();
        run_one_sweep_with_pair_layout(target, datastore.clone());
        let after_normal = datastore.table_handle(&name).unwrap();
        assert_eq!(after_normal.version(), stable_version);
        assert_eq!(
            after_normal
                .file_refs()
                .into_iter()
                .map(|file| file.path)
                .collect::<std::collections::HashSet<_>>(),
            stable_paths
        );
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name,
            after_normal,
            true,
        );
        assert_eq!(final_sweeps, 1);
        assert_eq!(
            datastore.table_handle_by_id(&table.id()).unwrap().version(),
            stable_version
        );

        dispatch.exit();
    }

    /// Three mutually overlapping 80%-full files are exactly the configured
    /// clique, so a single merge and one commit rewrite all of them together
    /// into sorted, range-disjoint files that no later pass selects again.
    #[test]
    fn layout_optimization_merges_a_group_of_three_in_one_pass() {
        const ROWS_PER_INPUT: usize = 256_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "first.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "second.parquet", ROWS_PER_INPUT, 29);
        write_wide_parquet_file(&adopted, "third.parquet", ROWS_PER_INPUT, 47);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs();
        assert_eq!(inputs.len(), 3);
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(5)
            .div_ceil(4);
        assert!(inputs.iter().all(|file| !is_small_file(file.size, target)));
        let before_version = table.version();

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(
            compacted.version(),
            before_version + 1,
            "one merge rewrites all three inputs"
        );
        let outputs = compacted.file_refs();
        assert_eq!(outputs.len(), 2, "target={target}, outputs={outputs:?}");
        assert!(outputs.iter().all(|file| file.size <= target));
        assert!(outputs.iter().all(|file| !is_small_file(file.size, target)));
        let mut ranges: Vec<(i64, i64)> = compacted
            .entries()
            .iter()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        assert_eq!(ranges.first().unwrap().0, 1);
        assert_eq!(ranges.last().unwrap().1, ROWS_PER_INPUT as i64);
        for window in ranges.windows(2) {
            assert!(window[0].1 < window[1].0, "ranges are disjoint: {ranges:?}");
        }
        let row_count: i64 = compacted
            .build_scan_view(&[], &[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (3 * ROWS_PER_INPUT) as i64);
        let log = db
            .path()
            .join(compacted.location())
            .join("_delta_log")
            .join(format!("{:020}.json", compacted.version()));
        let actions: Vec<serde_json::Value> = std::fs::read_to_string(log)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("remove").is_some())
                .count(),
            3
        );

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());
        assert_eq!(
            datastore.table_handle(&name).unwrap().version(),
            compacted.version()
        );

        dispatch.exit();
    }

    /// Two overlapping large files are less than the clique of three, so a
    /// routine round stands them deliberately; the final sweep settles them.
    #[test]
    fn routine_round_stands_a_partial_clique_and_the_final_sweep_settles_it() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap();
        let before_version = table.version();

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());
        let after_routine = datastore.table_handle(&name).unwrap();
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 3,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            after_routine.clone(),
            true,
        );

        assert_eq!(after_routine.version(), before_version);
        assert!(final_sweeps <= 3);
        let settled = datastore.table_handle(&name).unwrap();
        assert_eq!(settled.version(), before_version + 1);
        assert!(
            settled
                .file_refs()
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 4);
        dispatch.exit();
    }

    /// A wide file covers several mutually disjoint narrow files, none of
    /// which contests more than a fifth of its row groups: no pair clears
    /// the bar, so routine rounds stand still, and the final sweep's relaxed
    /// connected group settles the whole neighborhood into files that share
    /// no stretch of the key.
    #[test]
    fn final_sweep_settles_a_hub_no_routine_round_can_see() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file_with_row_group_size(&adopted, "hub.parquet", (0..20).collect(), 1);
        write_parquet_file(&adopted, "s1.parquet", vec![2, 3]);
        write_parquet_file(&adopted, "s2.parquet", vec![8, 9]);
        write_parquet_file(&adopted, "s3.parquet", vec![14, 15]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        // Keep every input, satellites included, out of the small-file path.
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap()
            .saturating_mul(2);
        let before_version = table.version();

        run_one_sweep(target, datastore.clone());
        let after_routine = datastore.table_handle(&name).unwrap();
        let final_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            after_routine.clone(),
            true,
        );

        assert_eq!(after_routine.version(), before_version);
        assert!(final_sweeps <= 4, "relaxed merging converges quickly");
        let settled = datastore.table_handle(&name).unwrap();
        assert!(settled.version() > before_version);
        assert!(
            settled
                .file_refs()
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let mut ranges: Vec<(i64, i64)> = settled
            .entries()
            .iter()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        for window in ranges.windows(2) {
            assert!(
                window[0].1 <= window[1].0,
                "no two files share a stretch of the key: {ranges:?}"
            );
        }
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 26);
        dispatch.exit();
    }

    /// The byte target is applied only between independently encoded row
    /// groups. A group that crosses it is emitted alone without changing the
    /// normal compaction row-group size or re-encoding the input.
    #[test]
    fn one_oversized_row_group_remains_one_output_file() {
        const ROWS_PER_INPUT: usize = 8_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "left.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "right.parquet", ROWS_PER_INPUT, 29);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();
        let inputs = table.file_refs();
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(5)
            .div_ceil(4);

        let outputs = merge_files(&table, &inputs, ROW_GROUP_ROWS, target).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].file_ref().size > target);
        for output in outputs {
            table.delete_data_file(&output.file_ref().path).unwrap();
        }
        dispatch.exit();
    }

    /// Partitioned compaction never mixes partitions, and both the replacement
    /// Adds and input tombstones retain the partition values needed to reload a
    /// valid Delta snapshot.
    #[test]
    fn compaction_preserves_partitioned_table_metadata_and_rows() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let encoded = tempfile::tempdir().unwrap();
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_partitioned_table(&datastore, dispatch.dispatcher(), "events");

        // Two inputs in each partition. If batching crossed the partition
        // boundary, the merged file could not truthfully carry one constant.
        let inputs = [
            ("one-a.parquet", 1),
            ("one-b.parquet", 1),
            ("two-a.parquet", 2),
            ("two-b.parquet", 2),
        ];
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        for (file_name, value) in inputs {
            write_partitioned_parquet_file(
                encoded.path(),
                file_name,
                vec![value * 10, value * 10 + 1],
                value,
            );
            table
                .append_data_file(
                    ObjectPath::new(file_name),
                    &std::fs::read(encoded.path().join(file_name)).unwrap(),
                    Some(partition_value(value)),
                )
                .unwrap();
        }
        let before_version = table.version();
        assert_eq!(table.file_refs().len(), 4);
        assert!(datastore.refresh_from_store().unwrap());

        let target = table
            .file_refs()
            .into_iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        run_command(
            CompacterSizes {
                merge_target_bytes: 0,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            table,
            false,
        );

        // One round merges each partition's small files in turn; a second round
        // then finds nothing left to do.
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 2);
        run_command(
            CompacterSizes {
                merge_target_bytes: 0,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            datastore.table_handle(&name).unwrap(),
            false,
        );

        // Reload from Delta rather than trusting the in-memory swap. Exactly one
        // output per partition remains, and every original row is still present.
        let mut reloaded = datastore.table_handle(&name).unwrap();
        reloaded.refresh().unwrap();
        assert_eq!(reloaded.file_refs().len(), 2);
        let partitions = reloaded.file_partitions();
        assert_eq!(partitions.len(), 2);
        for expected in [partition_value(1), partition_value(2)] {
            assert!(partitions.iter().any(|(_, actual)| {
                actual
                    .as_ref()
                    .is_some_and(|actual| scalar_values_equal(actual, &expected))
            }));
        }
        let parquet = reloaded.build_scan_view(&[], &[]).unwrap();
        assert_eq!(
            parquet
                .row_groups()
                .iter()
                .map(|group| group.num_rows)
                .sum::<i64>(),
            8
        );

        // Inspect every compaction commit. Each removes and adds files from one
        // partition only, and Kernel serialized that value into both actions.
        let log_dir = db.path().join(reloaded.location()).join("_delta_log");
        let mut compacted_partitions = std::collections::BTreeSet::new();
        for version in before_version + 1..=reloaded.version() {
            let commit =
                std::fs::read_to_string(log_dir.join(format!("{version:020}.json"))).unwrap();
            let actions: Vec<serde_json::Value> = commit
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let remove_values: std::collections::BTreeSet<_> = actions
                .iter()
                .filter_map(|action| action.get("remove"))
                .map(|remove| remove["partitionValues"]["Partition"].as_str().unwrap())
                .collect();
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| action.get("remove").is_some())
                    .count(),
                2
            );
            let add_values: std::collections::BTreeSet<_> = actions
                .iter()
                .filter_map(|action| action.get("add"))
                .map(|add| add["partitionValues"]["Partition"].as_str().unwrap())
                .collect();
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| action.get("add").is_some())
                    .count(),
                1
            );
            assert_eq!(remove_values.len(), 1, "one partition per compacted batch");
            assert_eq!(
                add_values, remove_values,
                "replacement stays in its partition"
            );
            compacted_partitions.extend(remove_values.into_iter().map(str::to_string));
        }
        assert_eq!(
            compacted_partitions,
            std::collections::BTreeSet::from(["1".to_string(), "2".to_string()])
        );

        dispatch.exit();
    }

    /// Compaction is location-agnostic: a table whose adopted files were listed
    /// **store-relative** (a directory under the database root, resolved through
    /// the store -- the same path a remote `s3://` root takes) compacts through
    /// the exact same code, with writes and deletes going through the table's
    /// store handle.
    #[test]
    fn compaction_works_on_store_relative_tables() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();

        // Two small files under a prefix inside the database root.
        write_parquet_file(&adopted_dir, "a.parquet", vec![1, 2]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![3]);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        // A relative adoption path: the files are read at `events` under the
        // store root, rather than at an absolute path of their own.
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            false,
        );
        let table_dir = db.path().join(
            datastore
                .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .location(),
        );

        let target = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .iter()
            .map(|f| f.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        run_command(
            CompacterSizes::defaults(target),
            datastore.clone(),
            name.clone(),
            datastore.table_handle(&name).unwrap(),
            true,
        );

        // One merged object replaced the two inputs, in the store and in the
        // datastore's table view.
        let files = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.as_str().starts_with("pivot-"));
        let on_disk: Vec<_> = std::fs::read_dir(&table_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".parquet"))
            .collect();
        // The merged object is written to the table's own directory; the two
        // adopted inputs are left where they were.
        assert!(on_disk.contains(&files[0].path.as_str().to_string()));
        assert!(adopted_dir.join("a.parquet").exists());
        assert!(adopted_dir.join("b.parquet").exists());

        let rows = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows)
            .sum::<i64>();
        assert_eq!(rows, 3);

        dispatch.exit();
    }
}
