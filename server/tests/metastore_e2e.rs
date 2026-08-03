//! End-to-end blackbox test of a multi-datastore server: a config file's
//! `metastore` section defines two local datastores (`default` and `warm`);
//! each is attached to DuckDB as its own database, so a table created in one is
//! queryable by its datastore name, and unqualified names resolve against
//! `default`.

mod common;

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use catalog::{DEFAULT_DATASTORE_NAME, PivotCatalog};
use common::{CatalogFixture, connect_client, select_rows, start_server};
use metastore::Metastore;
use metastore_yaml::YamlMetastore;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use server::Config;
use tempfile::TempDir;

/// (id BIGINT, name VARCHAR): alice, bob, carol.
fn people_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let id: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
    let name: ArrayRef = Arc::new(StringArray::from(vec!["alice", "bob", "carol"]));
    RecordBatch::try_new(schema, vec![id, name]).unwrap()
}

/// (id BIGINT, kind VARCHAR): click, view.
fn events_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("kind", DataType::Utf8, false),
    ]));
    let id: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2]));
    let kind: ArrayRef = Arc::new(StringArray::from(vec!["click", "view"]));
    RecordBatch::try_new(schema, vec![id, kind]).unwrap()
}

/// Write `batch` as a snappy parquet file at `<datastore>/<table>/data.parquet`,
/// so `CREATE TABLE <table> WITH (path = '<table>')` finds it under the
/// datastore's root.
fn write_table(datastore: &Path, table: &str, batch: &RecordBatch) {
    let dir = datastore.join(table);
    std::fs::create_dir_all(&dir).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.join("data.parquet")).unwrap(),
        batch.schema(),
        Some(props),
    )
    .unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn queries_bind_tables_by_datastore_name() {
    // Two local datastores, each holding one table's parquet files. The dirs must
    // outlive the server (which opens them by path), so keep the guards.
    let default_dir = TempDir::new().unwrap();
    let warm_dir = TempDir::new().unwrap();
    write_table(default_dir.path(), "people", &people_batch());
    write_table(warm_dir.path(), "events", &events_batch());

    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("pivot.yaml");
    // A short refresh interval: an INSERT commits to the log and the refresh
    // brings the new rows into the live set, so the test observes them quickly.
    std::fs::write(
        &config_path,
        format!(
            "metastore:\n  refresh_interval: 100ms\n  datastores:\n    default:\n      \
             kind: delta\n      location: \"{}\"\n      default: true\n    warm:\n      \
             kind: delta\n      location: \"{}\"\n",
            default_dir.path().display(),
            warm_dir.path().display(),
        ),
    )
    .unwrap();

    let port = start_server(64, move |dispatch| {
        let config = Config::open(&config_path).unwrap();
        let metastore = YamlMetastore::from_config(config.metastore).unwrap();
        let datastores = metastore.open_datastores(dispatch.dispatcher()).unwrap();
        CatalogFixture::new(Arc::new(
            PivotCatalog::new(datastores, DEFAULT_DATASTORE_NAME.to_string()).unwrap(),
        ))
    });
    let client = connect_client(port).await;

    // A table in the default datastore (unqualified) and one in `warm`.
    client
        .simple_query("CREATE TABLE people (id BIGINT, name VARCHAR) WITH (path = 'people')")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE warm.main.events (id BIGINT, kind VARCHAR) WITH (path = 'events')",
        )
        .await
        .unwrap();

    // Unqualified resolves against the default datastore.
    let default_rows = select_rows(&client, "SELECT name FROM people WHERE id = 2").await;
    assert_eq!(default_rows, vec![vec![Some("bob".into())]]);

    // The `warm` datastore's table resolves by its database name.
    let warm_rows = select_rows(&client, "SELECT kind FROM warm.main.events WHERE id = 1").await;
    assert_eq!(warm_rows, vec![vec![Some("click".into())]]);

    // The default datastore is also reachable by its (quoted, since `default` is
    // a keyword) name, the same tables as the unqualified form.
    let default_count = select_rows(&client, "SELECT COUNT(id) FROM \"default\".main.people").await;
    assert_eq!(default_count, vec![vec![Some("3".into())]]);

    // One statement can read from one datastore and write to another. The
    // source and target bindings must compile against different child snapshots,
    // and commit must finalize the upload in `warm`'s child transaction.
    client
        .simple_query(
            "INSERT INTO warm.main.events \
             SELECT id, name FROM people WHERE id = 3",
        )
        .await
        .unwrap();

    // The INSERT is durable in `warm`'s Delta log on commit, but only visible to a
    // later query once the background refresh advances the live set, so poll for it.
    let mut copied = Vec::new();
    for _ in 0..100 {
        copied = select_rows(&client, "SELECT kind FROM warm.main.events WHERE id = 3").await;
        if !copied.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(copied, vec![vec![Some("carol".into())]]);
}
