//! End-to-end pgwire tests of a pivot-managed Iceberg datastore: a config
//! file whose `iceberg` datastore names a `location` rather than a catalog,
//! served to a Postgres client that creates, fills and reads a table, which
//! lands in that location as an Iceberg table.

mod common;

use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;

use bin::server::{Config, Server};
use catalog::PivotCatalog;
use catalog::metastore::Metastore;
use common::{connect_client, pick_free_port, select_rows, wait_until_listening};
use dispatch::Dispatch;
use metastore_disk::DiskMetastore;
use tempfile::TempDir;

/// Start a server whose entire configuration is `config_yaml`, on a dedicated
/// thread, and return the port once it is listening.
fn start_server_from_config(config_yaml: &str) -> u16 {
    let port = pick_free_port();
    let bind: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let workers = core_affinity::get_core_ids().unwrap().len().clamp(1, 4);
    let directory = TempDir::new().unwrap();
    let config_path = directory.path().join("pivot.yaml");
    std::fs::write(&config_path, config_yaml).unwrap();

    thread::spawn(move || {
        let _directory = directory;
        let dispatch = Dispatch::spin_up(workers, 32, None);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
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
            let default_name = metastore.default_datastore_name().to_string();
            let catalog = Arc::new(
                PivotCatalog::new(datastores, default_name, metastore.clone())
                    .unwrap()
                    .with_external_parquet_read_context(
                        dispatch.dispatcher(),
                        metastore.external_store_factory(),
                    ),
            );
            let server = Server::new(bind, dispatch, catalog, metastore).unwrap();
            let _ = server.serve(Box::pin(std::future::pending::<()>())).await;
        });
    });

    wait_until_listening(bind);
    port
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

#[test]
fn an_iceberg_pivotlake_datastore_serves_writes_and_reads_over_pgwire() {
    let location = TempDir::new().unwrap();
    let config = format!(
        "datastore_refresh_interval: 100ms
datastores:
  events:
    kind: pivotlake
    table_format: iceberg
    location: {}
    default: true
    vacuum: false
",
        location.path().display()
    );

    let (count, total, data_paths) = block_on(async {
        let client = connect_client(start_server_from_config(&config)).await;
        client
            .simple_query("CREATE TABLE events (name VARCHAR, value BIGINT)")
            .await
            .unwrap();
        client
            .simple_query("INSERT INTO events VALUES ('a', 1), ('b', 2), ('c', 3)")
            .await
            .unwrap();
        let count = select_rows(&client, "SELECT count(*) FROM events").await;
        let total = select_rows(
            &client,
            "SELECT CAST(sum(value) AS BIGINT) FROM events WHERE value > 1",
        )
        .await;
        let data_paths = select_rows(&client, "SELECT data_path FROM system.datastores").await;
        (count, total, data_paths)
    });

    assert_eq!(count, vec![vec![Some("3".to_string())]]);
    assert_eq!(total, vec![vec![Some("5".to_string())]]);
    let data_path = data_paths[0][0].clone().unwrap();
    assert!(
        data_path.contains(location.path().to_str().unwrap()),
        "{data_path}"
    );
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(location.path().join("_pivot_manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(index["table_format"], "iceberg");
    let table_directory = index["table_locations"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    let metadata_dir = location.path().join(&table_directory).join("metadata");
    assert!(
        metadata_dir
            .join("v00000000000000000002.metadata.json")
            .is_file()
    );
    assert!(metadata_dir.join("version-hint.text").is_file());
}
