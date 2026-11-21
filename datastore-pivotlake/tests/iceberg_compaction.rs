//! Compaction of a managed table through `COMPACT`: small files are merged
//! into target-sized files in the table's own directory, overlapping sorted
//! files are rewritten into disjoint ranges, and every rewrite is one Replace
//! snapshot that retires its inputs with the numbers they were committed
//! with.

mod iceberg_support;

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use datastore_pivotlake::test_support::{live_data_file_names, retained_snapshots};
use datastore_pivotlake::{CompactionConfig, FileEntry, FileRef};
use iceberg_support::{
    Database, collect_i64s, collect_strings, names_in, parquet_files_in, single_i64,
    write_int_parquet,
};
use iceberg_tables::iceberg::spec::{
    FormatVersion, Manifest, ManifestList, ManifestStatus, Operation, TableMetadata,
};
use object_storage::ObjectPath;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet_engine::FileStats;
use planner::catalog::SchemaQualifiedTableName;

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
    db.execute("CREATE TABLE t (ts BIGINT, p BIGINT) PARTITIONED BY (p)");
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
    db.execute("CREATE TABLE t (v BIGINT) SORTED BY (v)");
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
fn a_routine_round_merges_the_small_files_a_reopened_database_holds() {
    // Any two small files are a merge.
    let compaction = CompactionConfig {
        target_bytes: 64 * 1024 * 1024,
        merge_target_bytes: 1,
        min_files: 2,
        max_concurrent_merges: 1,
        layout_clique_size: 2,
        poll_interval: Duration::from_secs(3600),
    };
    let mut db = Database::with_compaction(compaction.clone());
    db.execute("CREATE TABLE t (v BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    db.execute("INSERT INTO t VALUES (2)");
    db.execute("INSERT INTO t VALUES (3)");
    let directory = db.table_directory("t");
    let mut reopened = Database::reopen_with_compaction(db.close(), compaction);

    reopened.compact("t", false);

    let live = live_data_file_names(&reopened.root_uri(), &directory).unwrap();
    assert_eq!(live.len(), 1, "{live:?}");
    let rows = reopened.execute("SELECT v FROM t ORDER BY v");
    assert_eq!(collect_i64s(&rows, 0), [1, 2, 3]);
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

#[test]
fn a_final_compaction_rewrites_the_documents_of_a_variant_column_into_one_file() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (id BIGINT, doc VARIANT)");
    db.execute(r#"INSERT INTO events VALUES (1, '{"a": 1}'), (2, NULL)"#);
    db.execute(r#"INSERT INTO events VALUES (3, '{"b": "x"}'), (4, '{"a": "y"}')"#);
    let directory = db.table_directory("events");

    db.compact("events", true);

    let live = live_data_file_names(&db.root_uri(), &directory).unwrap();
    assert_eq!(live.len(), 1, "{live:?}");
    let documents =
        db.execute("SELECT id, CAST(doc AS VARCHAR) FROM events WHERE doc IS NOT NULL ORDER BY id");
    assert_eq!(collect_i64s(&documents, 0), [1, 3, 4]);
    assert_eq!(
        collect_strings(&documents, 1),
        [r#"{"a":1}"#, r#"{"b":"x"}"#, r#"{"a":"y"}"#]
    );
    let without_a_document = db.execute("SELECT id FROM events WHERE doc IS NULL");
    assert_eq!(collect_i64s(&without_a_document, 0), [2]);
}

/// Sizes under which every file counts as large and two files are a full
/// layout clique, so a round runs layout optimization alone and a merged
/// file holds one row group.
fn build_pair_layout_config() -> CompactionConfig {
    CompactionConfig {
        target_bytes: 2,
        merge_target_bytes: u64::MAX,
        min_files: usize::MAX,
        max_concurrent_merges: 1,
        layout_clique_size: 2,
        poll_interval: Duration::from_secs(3600),
    }
}

/// One-column (`url VARCHAR`) batches holding the URL of each of `pages`:
/// strings that tell the pages apart only after their first 26 code points.
fn build_page_url_batches(pages: Range<usize>) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "url",
        DataType::Utf8View,
        true,
    )]));
    let urls: Vec<String> = pages
        .map(|page| format!("https://example.com/pages/{page:06}"))
        .collect();
    urls.chunks(8192)
        .map(|urls| {
            let urls: ArrayRef = Arc::new(StringViewArray::from_iter_values(urls));
            RecordBatch::try_new(Arc::clone(&schema), vec![urls]).unwrap()
        })
        .collect()
}

/// Commit the live files of the table `name` again, as copies whose entries
/// bound the page URLs of `column` the way a writer that cuts string bounds
/// to 16 code points records them: below by the prefix every URL shares,
/// above by that prefix with its last code point incremented.
fn record_with_cut_url_bounds(db: &Database, name: &str, column: &str) {
    let bound = |value: &str| -> ArrayRef { Arc::new(StringViewArray::from(vec![value])) };
    let mut table = db
        .datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
        .unwrap();
    table.refresh().unwrap();
    let live = table.live_entries().unwrap();
    let table_path = db.table_path(name);
    let cut: Vec<FileEntry> = live
        .iter()
        .map(|entry| {
            let copy = format!("cut-{}", entry.file.path);
            std::fs::copy(
                table_path.join(entry.file.path.as_str()),
                table_path.join(&copy),
            )
            .unwrap();
            let stats = entry.stats.as_ref().unwrap();
            FileEntry {
                file: FileRef {
                    path: ObjectPath::new(copy),
                    size: entry.file.size,
                },
                partition: None,
                stats: Some(Arc::new(FileStats {
                    num_records: stats.num_records,
                    min_values: HashMap::from([(column.to_string(), bound("https://example."))]),
                    max_values: HashMap::from([(column.to_string(), bound("https://example/"))]),
                    null_counts: HashMap::from([(column.to_string(), 0)]),
                })),
                sorted_by: Arc::clone(&entry.sorted_by),
            }
        })
        .collect();
    let removed: Vec<ObjectPath> = live.into_iter().map(|entry| entry.file.path).collect();
    table.replace_data_files(&removed, &cut).unwrap();
    db.datastore.refresh_from_store().unwrap();
}

#[test]
fn sorted_files_whose_strings_share_a_long_prefix_and_do_not_overlap_are_left_alone() {
    let mut db = Database::with_compaction(build_pair_layout_config());
    db.execute("CREATE TABLE pages (url VARCHAR) SORTED BY (url)");
    db.insert_rows("pages", build_page_url_batches(0..10));
    db.insert_rows("pages", build_page_url_batches(10..20));
    let directory = db.table_directory("pages");
    let before = live_data_file_names(&db.root_uri(), &directory).unwrap();

    let passes = [db.compact("pages", false), db.compact("pages", true)];

    assert_eq!(passes, [1, 1], "a pass that finds no merge ends its round");
    assert_eq!(
        live_data_file_names(&db.root_uri(), &directory).unwrap(),
        before
    );
}

#[test]
fn overlapping_files_of_long_sort_strings_are_merged_into_files_no_later_round_rewrites() {
    let mut db = Database::with_compaction(build_pair_layout_config());
    db.execute("CREATE TABLE pages (url VARCHAR) SORTED BY (url)");
    db.insert_rows("pages", build_page_url_batches(0..100_000));
    db.insert_rows("pages", build_page_url_batches(0..100_000));
    let directory = db.table_directory("pages");

    let merging_passes = db.compact("pages", false);
    let merged = live_data_file_names(&db.root_uri(), &directory).unwrap();
    let later_passes = [
        db.compact("pages", false),
        db.compact("pages", true),
        db.compact("pages", true),
    ];

    assert_eq!(merging_passes, 2, "one merge, then a pass that finds none");
    assert_eq!(merged.len(), 2, "one file a row group: {merged:?}");
    assert_eq!(later_passes, [1, 1, 1]);
    assert_eq!(
        live_data_file_names(&db.root_uri(), &directory).unwrap(),
        merged
    );
    let rows = db.execute("SELECT count(*) FROM pages");
    assert_eq!(single_i64(&rows), 200_000);
}

#[test]
fn files_recorded_with_cut_sort_bounds_are_rewritten_once_into_files_no_later_round_rewrites() {
    let mut db = Database::with_compaction(build_pair_layout_config());
    db.execute("CREATE TABLE pages (url VARCHAR) SORTED BY (url)");
    db.insert_rows("pages", build_page_url_batches(0..100_000));
    db.insert_rows("pages", build_page_url_batches(100_000..200_000));
    record_with_cut_url_bounds(&db, "pages", "url");
    let directory = db.table_directory("pages");

    let rewriting_passes = db.compact("pages", false);
    let rewritten = live_data_file_names(&db.root_uri(), &directory).unwrap();
    let later_passes = [
        db.compact("pages", false),
        db.compact("pages", true),
        db.compact("pages", true),
    ];

    assert_eq!(
        rewriting_passes, 2,
        "one merge, then a pass that finds none"
    );
    assert_eq!(rewritten.len(), 2, "one file a row group: {rewritten:?}");
    assert!(
        rewritten.iter().all(|name| name.starts_with("pivot-")),
        "{rewritten:?}"
    );
    assert_eq!(later_passes, [1, 1, 1]);
    assert_eq!(
        live_data_file_names(&db.root_uri(), &directory).unwrap(),
        rewritten
    );
    let rows = db.execute("SELECT count(*) FROM pages");
    assert_eq!(single_i64(&rows), 200_000);
}
