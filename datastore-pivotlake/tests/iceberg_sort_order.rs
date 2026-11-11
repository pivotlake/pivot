//! The sort order a managed table's manifests record for its data files: the
//! table's own for a file whose rows a writer put in that order, the unsorted
//! order for a file written in none, and nothing for a file adopted as it lay.

mod iceberg_support;

use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use iceberg_support::{Database, names_in, write_int_parquet};
use iceberg_tables::iceberg::spec::{
    FormatVersion, Manifest, ManifestList, ManifestStatus, TableMetadata,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// The current metadata of the table at `table_path`.
fn read_current_metadata(table_path: &Path) -> TableMetadata {
    let metadata_dir = table_path.join("metadata");
    let latest = names_in(&metadata_dir)
        .into_iter()
        .filter(|name| name.ends_with(".metadata.json"))
        .max()
        .unwrap();
    serde_json::from_slice(&std::fs::read(metadata_dir.join(latest)).unwrap()).unwrap()
}

/// The file path and the sort order id of every live entry of the current
/// snapshot's manifests.
fn read_live_sort_order_ids(table_path: &Path) -> Vec<(String, Option<i32>)> {
    let metadata_dir = table_path.join("metadata");
    let metadata = read_current_metadata(table_path);
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
    let mut live = Vec::new();
    for manifest in list.entries() {
        let name = manifest.manifest_path.rsplit('/').next().unwrap();
        let parsed =
            Manifest::parse_avro(&std::fs::read(metadata_dir.join(name)).unwrap()).unwrap();
        for entry in parsed.entries() {
            if entry.status() != ManifestStatus::Deleted {
                live.push((
                    entry.file_path().to_string(),
                    entry.data_file().sort_order_id(),
                ));
            }
        }
    }
    live
}

/// The values of the one `BIGINT` column of the Parquet file a manifest names
/// by `file_uri`, in the order the file holds its rows.
fn read_values_in_file_order(file_uri: &str) -> Vec<Option<i64>> {
    let path = file_uri.strip_prefix("file://").unwrap_or(file_uri);
    ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .flat_map(|batch| {
            batch
                .unwrap()
                .column(0)
                .as_primitive::<Int64Type>()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn an_inserted_file_of_a_sorted_table_records_the_tables_sort_order() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT) WITH (sort_by = 'value')");

    db.execute("INSERT INTO t VALUES (3), (NULL), (1), (2)");

    let table_path = db.table_path("t");
    let live = read_live_sort_order_ids(&table_path);
    let sort_order_id = read_current_metadata(&table_path).default_sort_order_id();
    assert_eq!(sort_order_id, 1);
    assert_eq!(live.len(), 1, "{live:?}");
    assert_eq!(live[0].1, Some(1));
    assert_eq!(
        read_values_in_file_order(&live[0].0),
        [None, Some(1), Some(2), Some(3)]
    );
}

#[test]
fn a_compacted_file_of_a_sorted_table_records_the_tables_sort_order() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT) WITH (sort_by = 'value')");
    db.execute("INSERT INTO t VALUES (5), (1)");
    db.execute("INSERT INTO t VALUES (4), (NULL), (2)");

    db.compact("t", true);

    let live = read_live_sort_order_ids(&db.table_path("t"));
    assert_eq!(live.len(), 1, "{live:?}");
    assert_eq!(live[0].1, Some(1));
    assert_eq!(
        read_values_in_file_order(&live[0].0),
        [None, Some(1), Some(2), Some(4), Some(5)]
    );
}

#[test]
fn a_file_of_an_unsorted_table_records_the_unsorted_order() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");

    db.execute("INSERT INTO t VALUES (3), (1), (2)");

    let live = read_live_sort_order_ids(&db.table_path("t"));
    assert_eq!(live.len(), 1, "{live:?}");
    assert_eq!(live[0].1, Some(0));
}

#[test]
fn an_adopted_file_records_no_sort_order() {
    let mut db = Database::new();
    let adopted = db.path().join("events");
    std::fs::create_dir_all(&adopted).unwrap();
    write_int_parquet(&adopted.join("a.parquet"), &[3, 1, 2]);

    db.execute(
        "CREATE TABLE events (value BIGINT) \
         WITH (sort_by = 'value', with_pre_existing_parquets = 'events')",
    );

    let live = read_live_sort_order_ids(&db.table_path("events"));
    assert_eq!(live.len(), 1, "{live:?}");
    assert_eq!(live[0].1, None);
}

#[test]
fn a_file_compacted_from_an_adopted_file_records_the_unsorted_order() {
    let mut db = Database::new();
    let adopted = db.path().join("events");
    std::fs::create_dir_all(&adopted).unwrap();
    write_int_parquet(&adopted.join("a.parquet"), &[3, 1, 2]);
    db.execute(
        "CREATE TABLE events (value BIGINT) \
         WITH (sort_by = 'value', with_pre_existing_parquets = 'events')",
    );
    db.execute("INSERT INTO events VALUES (5), (4)");

    db.compact("events", true);

    let live = read_live_sort_order_ids(&db.table_path("events"));
    assert_eq!(live.len(), 1, "{live:?}");
    assert_eq!(live[0].1, Some(0));
}
