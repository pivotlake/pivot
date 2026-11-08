//! Wire-level `COPY <table> FROM STDIN WITH (FORMAT arrow)` into an Iceberg
//! catalog table: the rows land as one append snapshot of the table, and a
//! copy that leaves a required column out lands nothing. The tests return
//! early when Docker is unreachable, so the suite stays green offline.

mod common;

use std::sync::{Arc, Mutex, OnceLock};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bin::server::Config;
use bin::server::config::MetastoreStore;
use catalog::metastore::Metastore;
use catalog::{DEFAULT_DATASTORE_NAME, PivotCatalog};
use common::iceberg_fixture::{
    Fixture, create_namespace, create_table, load_table_metadata, start_fixture,
};
use common::{CatalogFixture, connect_client, select_rows, start_server};
use futures::SinkExt;
use metastore_disk::DiskMetastore;
use object_storage::{ObjectPath, open_store};
use tempfile::TempDir;
use tokio_postgres::Client;

const NAMESPACE: &str = "copied";

/// The catalog and the server over it, started once for the binary.
struct Harness {
    _fixture: Mutex<Fixture>,
    _config_dir: TempDir,
    port: u16,
    uri: String,
}

/// The harness, or `None` (with a note) when Docker is not available.
/// Starting the containers blocks, so a test calls this off its runtime.
fn harness() -> Option<&'static Harness> {
    static HARNESS: OnceLock<Option<Harness>> = OnceLock::new();
    HARNESS
        .get_or_init(|| {
            let fixture = start_fixture("copy-iceberg")?;
            create_namespace(&fixture.uri, NAMESPACE);
            let config_dir = TempDir::new().unwrap();
            let config_path = config_dir.path().join("pivot.yaml");
            let metastore_path = config_dir.path().join("metastore.yaml");
            std::fs::write(
                &config_path,
                format!(
                    "datastores:\n  lake:\n    kind: iceberg\n    uri: \"{}\"\n    default: true\n\
                     metastore:\n  kind: file\n  path: \"{}\"\n",
                    fixture.uri,
                    metastore_path.display(),
                ),
            )
            .unwrap();
            std::fs::write(&metastore_path, "datastores: {}\n").unwrap();
            let port = start_server(64, move |dispatch| {
                let config = Config::open(&config_path).unwrap();
                let Some(MetastoreStore::File { path }) = config.metastore else {
                    panic!("the config names its metastore file");
                };
                let metastore = Arc::new(
                    DiskMetastore::open(
                        config.entries,
                        Some(&path),
                        config.datastore_refresh_interval.as_duration(),
                    )
                    .unwrap(),
                );
                let datastores = metastore.open_datastores(dispatch.dispatcher()).unwrap();
                CatalogFixture::new(Arc::new(
                    PivotCatalog::new(datastores, DEFAULT_DATASTORE_NAME.to_string(), metastore)
                        .unwrap(),
                ))
            });
            let uri = fixture.uri.clone();
            Some(Harness {
                _fixture: Mutex::new(fixture),
                _config_dir: config_dir,
                port,
                uri,
            })
        })
        .as_ref()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_copy_into_a_catalog_table_lands_as_one_append_snapshot() {
    let Some(harness) = tokio::task::spawn_blocking(harness).await.unwrap() else {
        return;
    };
    create_table(&harness.uri, NAMESPACE, "events", true);
    let client = connect_client(harness.port).await;
    let bytes = arrow_stream_bytes(&events_batch(&[1, 2], &["ann", "bob"]));

    let copied = copy_in_arrow(
        &client,
        "COPY copied.events FROM STDIN WITH (FORMAT arrow)",
        bytes,
    )
    .await
    .unwrap();

    assert_eq!(copied, 2);
    let rows = select_rows(&client, "SELECT id, name FROM copied.events ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("ann".into())],
            vec![Some("2".into()), Some("bob".into())],
        ]
    );
    let metadata = load_table_metadata(&harness.uri, NAMESPACE, "events");
    let snapshots = metadata["snapshots"].as_array().unwrap();
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0]["summary"]["added-data-files"], "1");
    assert_eq!(data_files_of(&metadata).len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_copy_leaving_out_a_required_column_lands_no_snapshot_and_no_files() {
    let Some(harness) = tokio::task::spawn_blocking(harness).await.unwrap() else {
        return;
    };
    create_table(&harness.uri, NAMESPACE, "strict", true);
    let client = connect_client(harness.port).await;
    let names: ArrayRef = Arc::new(StringArray::from(vec!["ann"]));
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)])),
        vec![names],
    )
    .unwrap();

    let error = copy_in_arrow(
        &client,
        "COPY copied.strict (name) FROM STDIN WITH (FORMAT arrow)",
        arrow_stream_bytes(&batch),
    )
    .await
    .expect_err("a NULL headed for a required column fails the copy");

    assert!(error.to_string().contains("id"), "{error}");
    let metadata = load_table_metadata(&harness.uri, NAMESPACE, "strict");
    assert!(metadata["current-snapshot-id"].is_null());
    assert!(data_files_of(&metadata).is_empty());
    let rows = select_rows(&client, "SELECT id FROM copied.strict").await;
    assert!(rows.is_empty());
}

/// Stream one Arrow IPC payload through `COPY ... FROM STDIN` and return the
/// row count the server reports, or the error that ended the copy.
async fn copy_in_arrow(
    client: &Client,
    sql: &str,
    bytes: Vec<u8>,
) -> Result<u64, tokio_postgres::Error> {
    let sink = client.copy_in(sql).await?;
    let mut sink = Box::pin(sink);
    sink.send(bytes::Bytes::from(bytes)).await?;
    sink.as_mut().finish().await
}

/// `batch` serialized as one Arrow IPC stream, the bytes a client sends for
/// `COPY ... WITH (FORMAT arrow)`.
fn arrow_stream_bytes(batch: &RecordBatch) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer =
        arrow::ipc::writer::StreamWriter::try_new(&mut bytes, &batch.schema()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    drop(writer);
    bytes
}

fn events_batch(ids: &[i64], names: &[&str]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names.to_vec())),
        ],
    )
    .unwrap()
}

/// The objects under the table's `data/` directory, as the store lists them.
fn data_files_of(metadata: &serde_json::Value) -> Vec<String> {
    let location = metadata["location"].as_str().unwrap();
    let store = open_store(location).unwrap();
    store
        .list(&ObjectPath::new("data"))
        .unwrap()
        .objects
        .into_iter()
        .map(|object| object.file.path.to_string())
        .collect()
}
