//! End-to-end pgwire tests of the `secrets` section against real object
//! storage: a config file whose datastore lives in a bucket and holds no
//! credentials of its own, served to a Postgres client that reads Parquet out
//! of that bucket.
//!
//! Everything the server needs to reach the store comes from the file. The S3
//! case proves it: alongside the secret scoped to the datastore's own prefix,
//! the config carries a decoy scoped to every `s3://` location whose endpoint
//! and keys are wrong, so a query only answers if scope resolution picked the
//! more specific secret and signed with what it carries.
//!
//! MinIO and `fake-gcs-server` come from the shared harness, so these skip when
//! Docker is absent.

mod common;

use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;

use catalog::PivotCatalog;
use common::{
    connect_client, parquet_name_value_rows, pick_free_port, select_rows, wait_until_listening,
};
use datastore_delta::test_support::{self, Backend};
use dispatch::Dispatch;
use metastore::Metastore;
use metastore_disk::DiskMetastore;
use object_storage::ObjectPath;
use server::{Config, Server};
use tempfile::TempDir;
use tokio_postgres::Client;

/// Start a server whose entire configuration is `config_yaml`, on a dedicated
/// thread, and return the port once it is listening. The catalog is opened the
/// way the binary opens it: the config file's `metastore` section, through
/// [`DiskMetastore`], which resolves each datastore's secret.
fn start_server_from_config(config_yaml: &str) -> u16 {
    let port = pick_free_port();
    let bind: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let workers = core_affinity::get_core_ids().unwrap().len().clamp(1, 4);
    let directory = TempDir::new().unwrap();
    let config_path = directory.path().join("pivot.yaml");
    std::fs::write(&config_path, config_yaml).unwrap();

    thread::spawn(move || {
        // The config file is read on this thread, so its directory has to
        // outlive the read rather than the caller's statement.
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
                    config.metastore,
                    None,
                    config.server.refresh_interval.as_duration(),
                )
                .unwrap(),
            );
            let datastores = metastore.open_datastores(dispatch.dispatcher()).unwrap();
            let default_name = metastore.default_datastore_name().to_string();
            let catalog =
                Arc::new(PivotCatalog::new(datastores, default_name, metastore.clone()).unwrap());
            let server = Server::new(bind, dispatch, catalog, metastore);
            let _ = server.serve(Box::pin(std::future::pending::<()>())).await;
        });
    });

    wait_until_listening(bind);
    port
}

/// The `metastore.datastores` section for a backend's root: the one datastore,
/// carrying no credentials. Maintenance is off, so the only traffic the bucket
/// sees is the test's own.
fn datastores_section(root: &str) -> String {
    format!(
        "server:
  refresh_interval: 100ms
metastore:
  datastores:
    warm:
      kind: delta
      location: {root}
      default: true
      compact: false
      vacuum: false
"
    )
}

/// Upload a two-file `events` table (5 rows) into `backend`'s bucket, start a
/// server on `config_yaml`, and `CREATE TABLE` over those files.
async fn events_server(backend: &Backend, config_yaml: &str) -> Client {
    backend
        .store
        .put(
            &ObjectPath::new("events/p1.parquet"),
            &parquet_name_value_rows(&[1, 2, 3]),
        )
        .unwrap();
    backend
        .store
        .put(
            &ObjectPath::new("events/p2.parquet"),
            &parquet_name_value_rows(&[4, 5]),
        )
        .unwrap();
    let client = connect_client(start_server_from_config(config_yaml)).await;
    client
        .simple_query(
            "CREATE TABLE events (name VARCHAR, value BIGINT) \
             WITH (with_pre_existing_parquets = 'events')",
        )
        .await
        .unwrap();
    client
}

/// Drive an async body on a fresh runtime. Plain `#[test]` so the harness's
/// container startup, which blocks on a runtime of its own, runs before we
/// enter one.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

#[test]
fn an_s3_datastore_is_served_with_the_secret_scoped_to_it() {
    let Some(backend) = test_support::s3("secrets-s3") else {
        return;
    };
    // The harness publishes MinIO's coordinates in the environment for stores
    // that resolve their own credentials. Here they go into a secret instead:
    // a configured datastore never consults the environment, and the decoy
    // below covers every other `s3://` location with an endpoint nothing
    // answers on, so reaching MinIO at all means the scoped secret won.
    let config = format!(
        "{}  secrets:
    decoy:
      type: s3
      scope: s3://
      region: {region}
      access_key_id: wrong-key
      secret_access_key: wrong-secret
      endpoint: http://127.0.0.1:1
    minio:
      type: s3
      scope: {root}
      region: {region}
      access_key_id: {access_key_id}
      secret_access_key: {secret_access_key}
      endpoint: {endpoint}
",
        datastores_section(&backend.root),
        root = backend.root,
        region = std::env::var("AWS_REGION").unwrap(),
        access_key_id = std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
        secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
        endpoint = std::env::var("AWS_ENDPOINT_URL").unwrap(),
    );

    let rows = block_on(async {
        let client = events_server(&backend, &config).await;
        select_rows(&client, "SELECT COUNT(value) FROM events").await
    });

    assert_eq!(rows, vec![vec![Some("5".to_string())]]);
}

#[test]
fn a_gcs_datastore_is_served_with_the_secret_scoped_to_it() {
    let Some(backend) = test_support::gcs("secrets-gcs") else {
        return;
    };
    // The emulator serves any caller, so what this covers is the configured
    // path: the datastore resolves its secret and is opened with the key file
    // the secret names. That the file is the one tokens are minted from is
    // `metastore-disk`'s own test, since the emulator mints none.
    let key_file = TempDir::new().unwrap();
    let key_path = key_file.path().join("gcs-key.json");
    std::fs::write(&key_path, r#"{"type":"authorized_user"}"#).unwrap();
    let config = format!(
        "{}  secrets:
    google:
      type: gcs
      scope: {root}
      credentials_file: {key_path}
",
        datastores_section(&backend.root),
        root = backend.root,
        key_path = key_path.display(),
    );

    let rows = block_on(async {
        let client = events_server(&backend, &config).await;
        select_rows(&client, "SELECT COUNT(value) FROM events").await
    });

    assert_eq!(rows, vec![vec![Some("5".to_string())]]);
}

#[test]
fn a_gcs_datastore_no_secret_covers_is_served_with_ambient_credentials() {
    let Some(backend) = test_support::gcs("secrets-gcs-ambient") else {
        return;
    };
    let config = datastores_section(&backend.root);

    let rows = block_on(async {
        let client = events_server(&backend, &config).await;
        select_rows(&client, "SELECT COUNT(value) FROM events").await
    });

    assert_eq!(rows, vec![vec![Some("5".to_string())]]);
}
