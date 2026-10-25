//! Compaction of a managed table through `COMPACT`: small files are merged
//! into target-sized files in the table's own directory, overlapping sorted
//! files are rewritten into disjoint ranges, and every rewrite is one Replace
//! snapshot that retires its inputs with the numbers they were committed
//! with.

mod iceberg_support;

use std::time::Duration;

use datastore_pivotlake::CompactionConfig;
use datastore_pivotlake::test_support::{live_data_file_names, retained_snapshots};
use iceberg_support::{
    Database, collect_i64s, names_in, parquet_files_in, single_i64, write_int_parquet,
};
use iceberg_tables::iceberg::spec::{
    FormatVersion, Manifest, ManifestList, ManifestStatus, Operation, TableMetadata,
};
use parquet::file::reader::{FileReader, SerializedFileReader};

/// The current metadata of the table at `table_path`.
fn current_metadata(table_path: &std::path::Path) -> TableMetadata {
    let metadata_dir = table_path.join("metadata");
    let latest = names_in(&metadata_dir)
        .into_iter()
        .filter(|name| name.ends_with(".metadata.json"))
        .max()
        .unwrap();
    serde_json::from_slice(&std::fs::read(metadata_dir.join(latest)).unwrap()).unwrap()
}

/// Every entry of the current snapshot's manifests, with its status.
fn current_entries(table_path: &std::path::Path) -> Vec<(ManifestStatus, String, Option<i64>)> {
    let metadata_dir = table_path.join("metadata");
    let metadata = current_metadata(table_path);
    let list_name = metadata
        .current_snapshot()
        .unwrap()
        .manifest_list()
        .rsplit('/')
        .next()
        .unwrap()
        .to_string();
    let list = ManifestList::parse_with_version(
        &std::fs::read(metadata_dir.join(list_name)).unwrap(),
        FormatVersion::V3,
    )
    .unwrap();
    let mut entries = Vec::new();
    for manifest in list.entries() {
        let name = manifest.manifest_path.rsplit('/').next().unwrap();
        let parsed =
            Manifest::parse_avro(&std::fs::read(metadata_dir.join(name)).unwrap()).unwrap();
        for entry in parsed.entries() {
            let file_name = entry.file_path().rsplit('/').next().unwrap().to_string();
            entries.push((entry.status(), file_name, entry.sequence_number()));
        }
    }
    entries
}

#[test]
fn a_final_compaction_merges_small_files_and_retires_them_in_a_replace_snapshot() {
    let mut db = Database::new();
    let adopted = db.path().join("events");
    std::fs::create_dir_all(&adopted).unwrap();
    write_int_parquet(&adopted.join("a.parquet"), &[1, 2, 3]);
    write_int_parquet(&adopted.join("b.parquet"), &[4, 5]);
    write_int_parquet(&adopted.join("c.parquet"), &[6, 7, 8, 9]);
    db.execute("CREATE TABLE events (value BIGINT) WITH (with_pre_existing_parquets = 'events')");
    let directory = db.table_directory("events");

    db.compact("events", true);

    let live = live_data_file_names(&db.root_uri(), &directory).unwrap();
    assert_eq!(live.len(), 1, "{live:?}");
    assert!(live[0].starts_with("pivot-"), "{live:?}");
    assert_eq!(parquet_files_in(&db.table_path("events")), live);
    for name in ["a.parquet", "b.parquet", "c.parquet"] {
        assert!(
            adopted.join(name).exists(),
            "an adopted file is never deleted"
        );
    }
    let rows = db.execute("SELECT CAST(sum(value) AS BIGINT), count(*) FROM events");
    assert_eq!(collect_i64s(&rows, 0), [45]);
    assert_eq!(collect_i64s(&rows, 1), [9]);
    let metadata = current_metadata(&db.table_path("events"));
    assert_eq!(
        metadata.current_snapshot().unwrap().summary().operation,
        Operation::Replace
    );
    let entries = current_entries(&db.table_path("events"));
    let deleted: Vec<_> = entries
        .iter()
        .filter(|(status, _, _)| *status == ManifestStatus::Deleted)
        .collect();
    assert_eq!(deleted.len(), 3, "{entries:?}");
    assert!(
        deleted.iter().all(|(_, _, sequence)| *sequence == Some(1)),
        "retired inputs keep the sequence number they were committed with: {entries:?}"
    );
    assert_eq!(
        entries
            .iter()
            .filter(|(status, _, _)| *status == ManifestStatus::Added)
            .count(),
        1
    );
}

#[test]
fn compaction_keeps_each_partitions_files_apart() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (ts BIGINT, p BIGINT) WITH (partition_by = 'p')");
    db.execute("INSERT INTO t VALUES (10, 1), (11, 1)");
    db.execute("INSERT INTO t VALUES (12, 1)");
    db.execute("INSERT INTO t VALUES (20, 2)");
    db.execute("INSERT INTO t VALUES (21, 2), (22, 2)");
    let directory = db.table_directory("t");
    assert_eq!(
        live_data_file_names(&db.root_uri(), &directory)
            .unwrap()
            .len(),
        4
    );

    db.compact("t", true);

    assert_eq!(
        live_data_file_names(&db.root_uri(), &directory)
            .unwrap()
            .len(),
        2
    );
    let rows = db.execute("SELECT p, count(*) FROM t GROUP BY p ORDER BY p");
    assert_eq!(collect_i64s(&rows, 0), [1, 2]);
    assert_eq!(collect_i64s(&rows, 1), [3, 3]);
    let pruned = db.execute("SELECT CAST(sum(ts) AS BIGINT) FROM t WHERE p = 2");
    assert_eq!(single_i64(&pruned), 63);
}

#[test]
fn a_routine_round_rewrites_overlapping_sorted_files_into_disjoint_ranges() {
    // Every file counts as large and layout work pairs them, so the round
    // runs layout optimization alone.
    let mut db = Database::with_compaction(CompactionConfig {
        target_bytes: 2,
        merge_target_bytes: u64::MAX,
        min_files: usize::MAX,
        max_concurrent_merges: 1,
        layout_clique_size: 2,
        poll_interval: Duration::from_secs(3600),
    });
    db.execute("CREATE TABLE t (v BIGINT) WITH (sort_by = 'v')");
    db.execute("INSERT INTO t VALUES (0), (100)");
    db.execute("INSERT INTO t VALUES (0), (100)");
    db.execute("INSERT INTO t VALUES (1000), (1100)");
    let directory = db.table_directory("t");
    let before = live_data_file_names(&db.root_uri(), &directory).unwrap();

    db.compact("t", false);

    let after = live_data_file_names(&db.root_uri(), &directory).unwrap();
    let kept: Vec<_> = after.iter().filter(|name| before.contains(name)).collect();
    assert_eq!(
        kept.len(),
        1,
        "the disjoint file is not part of the pair: {after:?}"
    );
    assert_eq!(after.len(), 2, "{after:?}");
    let rows = db.execute("SELECT v FROM t ORDER BY v");
    assert_eq!(collect_i64s(&rows, 0), [0, 0, 100, 100, 1000, 1100]);
    assert_eq!(retained_snapshots(&db.root_uri(), &directory).unwrap(), 4);
}

#[test]
fn compaction_outputs_carry_the_tables_field_ids() {
    let mut db = Database::new();
    let adopted = db.path().join("events");
    std::fs::create_dir_all(&adopted).unwrap();
    write_int_parquet(&adopted.join("a.parquet"), &[1]);
    write_int_parquet(&adopted.join("b.parquet"), &[2]);
    db.execute("CREATE TABLE events (value BIGINT) WITH (with_pre_existing_parquets = 'events')");

    db.compact("events", true);

    let table_path = db.table_path("events");
    let output = parquet_files_in(&table_path).remove(0);
    let reader =
        SerializedFileReader::new(std::fs::File::open(table_path.join(output)).unwrap()).unwrap();
    let fields = reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema()
        .get_fields()
        .to_vec();
    assert_eq!(fields[0].get_basic_info().id(), 1);
}
