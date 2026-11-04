//! End-to-end blackbox test of a datastore configured with a `default_schema`:
//! a table name that names no schema, and a `datastore.table` name, resolve in
//! that schema rather than in `main`.

mod common;

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bin::server::Config;
use catalog::PivotCatalog;
use catalog::metastore::Metastore;
use common::{CatalogFixture, connect_client, select_rows, start_server};
use metastore_disk::DiskMetastore;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

/// Write (id BIGINT, name VARCHAR) rows alice and bob as `<datastore>/<table>/data.parquet`,
/// where `CREATE TABLE <table> WITH (with_pre_existing_parquets = '<table>')` finds them.
fn write_people_table(datastore: &Path, table: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let id: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2]));
    let name: ArrayRef = Arc::new(StringArray::from(vec!["alice", "bob"]));
    let batch = RecordBatch::try_new(schema, vec![id, name]).unwrap();
    let dir = datastore.join(table);
    std::fs::create_dir_all(&dir).unwrap();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.join("data.parquet")).unwrap(),
        batch.schema(),
        None,
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn unqualified_names_resolve_in_the_configured_default_schema() {
    let datastore_dir = TempDir::new().unwrap();
    write_people_table(datastore_dir.path(), "staff");
    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("pivot.yaml");
    std::fs::write(
        &config_path,
        format!(
            "datastores:\n  hot:\n    kind: pivotlake\n    location: \"{}\"\n    default: true\n    \
             default_schema: reporting\n",
            datastore_dir.path().display(),
        ),
    )
    .unwrap();
    let port = start_server(64, move |dispatch| {
        let config = Config::open(&config_path).unwrap();
        let metastore = Arc::new(
            DiskMetastore::open(
                config.entries,
                None,
                config.datastore_refresh_interval.as_duration(),
            )
            .unwrap(),
        );
        let datastores = metastore.open_datastores(dispatch.dispatcher()).unwrap();
        CatalogFixture::new(Arc::new(
            PivotCatalog::new(
                datastores,
                metastore.default_datastore_name().to_string(),
                metastore.clone(),
            )
            .unwrap(),
        ))
    });
    let client = connect_client(port).await;
    client
        .simple_query("CREATE SCHEMA reporting")
        .await
        .unwrap();

    client
        .simple_query(
            "CREATE TABLE staff (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = 'staff')",
        )
        .await
        .unwrap();
    let unqualified = select_rows(&client, "SELECT name FROM staff WHERE id = 1").await;
    let by_datastore = select_rows(&client, "SELECT name FROM hot.staff WHERE id = 2").await;
    let fully_qualified = select_rows(&client, "SELECT count(id) FROM hot.reporting.staff").await;
    let in_main = client.simple_query("SELECT name FROM main.staff").await;

    assert_eq!(unqualified, vec![vec![Some("alice".into())]]);
    assert_eq!(by_datastore, vec![vec![Some("bob".into())]]);
    assert_eq!(fully_qualified, vec![vec![Some("2".into())]]);
    assert!(
        in_main.is_err(),
        "the table was created in `reporting`, not `main`"
    );
}
