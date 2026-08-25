//! End-to-end blackbox test of a multi-datastore server: two local datastores,
//! `default` from the config file's `metastore` section and `warm` from the
//! separate metastore file merged into it. Each is attached to DuckDB as its own
//! database, so a table created in one is queryable by its datastore name, and
//! unqualified names resolve against `default`.

mod common;

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bin::server::Config;
use catalog::metastore::Metastore;
use catalog::{DEFAULT_DATASTORE_NAME, PivotCatalog};
use common::{CatalogFixture, connect_client, select_rows, start_server};
use metastore_disk::DiskMetastore;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
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
/// so `CREATE TABLE <table> WITH (with_pre_existing_parquets = '<table>')` finds it
/// under the datastore's root.
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

fn manifest_table_id(datastore: &Path, schema: &str, table: &str) -> String {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(datastore.join("_pivot_manifest.json")).unwrap())
            .unwrap();
    manifest["schemas"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"].as_str() == Some(schema))
        .unwrap()["table_ids"][table]
        .as_str()
        .unwrap()
        .to_string()
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
    let metastore_path = config_dir.path().join("metastore.yaml");
    // A short refresh interval: an INSERT commits to the log and the refresh
    // brings the new rows into the live set, so the test observes them quickly.
    std::fs::write(
        &config_path,
        format!(
            "server:\n  refresh_interval: 100ms\nmetastore:\n  datastores:\n    default:\n      \
             kind: delta\n      location: \"{}\"\n      default: true\n",
            default_dir.path().display(),
        ),
    )
    .unwrap();
    std::fs::write(
        &metastore_path,
        format!(
            "datastores:\n  warm:\n    kind: delta\n    location: \"{}\"\n",
            warm_dir.path().display(),
        ),
    )
    .unwrap();

    let port = start_server(64, move |dispatch| {
        let config = Config::open(&config_path).unwrap();
        let metastore = Arc::new(
            DiskMetastore::open(
                config.metastore,
                Some(&metastore_path),
                config.server.refresh_interval.as_duration(),
            )
            .unwrap(),
        );
        let datastores = metastore.open_datastores(dispatch.dispatcher()).unwrap();
        CatalogFixture::new(Arc::new(
            PivotCatalog::new(
                datastores,
                DEFAULT_DATASTORE_NAME.to_string(),
                metastore.clone(),
            )
            .unwrap(),
        ))
    });
    let client = connect_client(port).await;

    // A table in the default datastore (unqualified) and one in `warm`.
    client
        .simple_query(
            "CREATE TABLE people (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = 'people')",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE warm.main.events (id BIGINT, kind VARCHAR) WITH (with_pre_existing_parquets = 'events')",
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

    // `system` is a global schema: resolving it through either datastore returns
    // the same deterministic inventory, including the durable IDs stored in
    // each datastore's manifest.
    let default_id = manifest_table_id(default_dir.path(), "main", "people");
    let warm_id = manifest_table_id(warm_dir.path(), "main", "events");
    let expected_inventory = vec![
        vec![
            Some("default".into()),
            Some("main".into()),
            Some("people".into()),
            Some(default_id),
        ],
        // The system datastore describes itself, so its own relations are part
        // of the inventory. They are not stored, so their ids are minted with
        // the datastore rather than by a manifest, and are constant.
        vec![
            Some("system".into()),
            Some("main".into()),
            Some("columns".into()),
            Some("f1e1d500-da7a-4ce5-bead-e4c77ab1e50f".into()),
        ],
        vec![
            Some("system".into()),
            Some("main".into()),
            Some("datastores".into()),
            Some("da7aba5e-5e75-4a11-ab1e-5e1ec7edda7a".into()),
        ],
        vec![
            Some("system".into()),
            Some("main".into()),
            Some("memory_blocks".into()),
            Some("a110ca7e-b10c-4bed-ba5e-b10c54110ca7".into()),
        ],
        vec![
            Some("system".into()),
            Some("main".into()),
            Some("table_files".into()),
            Some("7ab1ef11-e500-4ded-b10b-de1e7edf11e5".into()),
        ],
        vec![
            Some("system".into()),
            Some("main".into()),
            Some("tables".into()),
            Some("007ab1e5-1157-4c1d-8055-f1e1d50fda7a".into()),
        ],
        vec![
            Some("warm".into()),
            Some("main".into()),
            Some("events".into()),
            Some(warm_id),
        ],
    ];
    // The inventory spans both datastores whichever way it is named: `system`
    // is a datastore of its own, so the bare spelling and the schema-qualified
    // one are the same relation.
    let inventory_sql = "SELECT datastore, schema, name, id \
                         FROM system.tables \
                         ORDER BY datastore, schema, name";
    assert_eq!(
        select_rows(&client, inventory_sql).await,
        expected_inventory
    );

    let qualified_inventory_sql = "SELECT datastore, schema, name, id \
                                   FROM system.main.tables \
                                   ORDER BY datastore, schema, name";
    assert_eq!(
        select_rows(&client, qualified_inventory_sql).await,
        expected_inventory,
    );

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
