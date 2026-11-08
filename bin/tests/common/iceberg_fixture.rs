//! The Iceberg REST catalog the end-to-end tests read and write through: the
//! `apache/iceberg-rest-fixture` image in Docker, storing its tables in the
//! MinIO the object-storage harness starts, with its namespaces and tables
//! defined over plain HTTP.

use std::time::{Duration, Instant};

use object_storage::test_support;
use testcontainers::core::{ContainerPort, Host, IntoContainerPort, WaitFor};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};

/// Where the fixture stores its tables: a prefix of the bucket the
/// object-storage harness creates.
pub const WAREHOUSE: &str = "s3://pivot-it/shell-iceberg-warehouse";

pub struct Fixture {
    _container: Container<GenericImage>,
    pub uri: String,
}

/// Bring the REST fixture up against the harness's MinIO, or `None` (with a
/// note naming `suite`) when Docker is not available.
pub fn start_fixture(suite: &str) -> Option<Fixture> {
    let Some(_backend) = test_support::s3(suite) else {
        eprintln!("[{suite}] skipping: MinIO unavailable");
        return None;
    };
    // The harness points the process's `AWS_*` variables at MinIO, which is
    // how Pivot reads and writes the tables' files; the fixture reaches the
    // same endpoint through the Docker host gateway.
    let s3_endpoint = std::env::var("AWS_ENDPOINT_URL").expect("the S3 harness sets its endpoint");
    let minio_port = url::Url::parse(&s3_endpoint)
        .unwrap()
        .port()
        .expect("the harness endpoint names its port");
    let image = GenericImage::new("apache/iceberg-rest-fixture", "1.9.1")
        .with_exposed_port(ContainerPort::Tcp(8181))
        .with_wait_for(WaitFor::message_on_stdout("Started"))
        .with_env_var("CATALOG_WAREHOUSE", WAREHOUSE)
        .with_env_var("CATALOG_IO__IMPL", "org.apache.iceberg.aws.s3.S3FileIO")
        .with_env_var(
            "CATALOG_S3_ENDPOINT",
            format!("http://host.docker.internal:{minio_port}"),
        )
        .with_env_var("CATALOG_S3_PATH__STYLE__ACCESS", "true")
        .with_env_var(
            "AWS_ACCESS_KEY_ID",
            std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
        )
        .with_env_var(
            "AWS_SECRET_ACCESS_KEY",
            std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
        )
        .with_env_var("AWS_REGION", std::env::var("AWS_REGION").unwrap())
        .with_host("host.docker.internal", Host::HostGateway);
    let container = match image.start() {
        Ok(container) => container,
        Err(error) => {
            eprintln!("[{suite}] skipping: REST fixture unavailable: {error}");
            return None;
        }
    };
    let port = container.get_host_port_ipv4(8181.tcp()).unwrap();
    let uri = format!("http://localhost:{port}");
    wait_for_catalog(&uri);
    Some(Fixture {
        _container: container,
        uri,
    })
}

/// Poll the catalog's config endpoint until it answers: the container's log
/// says it started a moment before it accepts requests.
fn wait_for_catalog(uri: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if ureq::get(&format!("{uri}/v1/config")).call().is_ok() {
            return;
        }
        assert!(Instant::now() < deadline, "the REST catalog never answered");
        std::thread::sleep(Duration::from_millis(250));
    }
}

pub fn create_namespace(uri: &str, namespace: &str) {
    ureq::post(&format!("{uri}/v1/namespaces"))
        .send_json(serde_json::json!({ "namespace": [namespace] }))
        .expect("create namespace");
}

/// Create `table` with two columns, `id` (long, required when `id_required`)
/// and `name` (string), and no snapshot.
pub fn create_table(uri: &str, namespace: &str, table: &str, id_required: bool) {
    ureq::post(&format!("{uri}/v1/namespaces/{namespace}/tables"))
        .send_json(serde_json::json!({
            "name": table,
            "schema": {
                "type": "struct",
                "schema-id": 0,
                "fields": [
                    { "id": 1, "name": "id", "required": id_required, "type": "long" },
                    { "id": 2, "name": "name", "required": false, "type": "string" },
                ],
            },
        }))
        .expect("create table");
}

/// The table's metadata as the catalog serves it now.
pub fn load_table_metadata(uri: &str, namespace: &str, table: &str) -> serde_json::Value {
    let response: serde_json::Value =
        ureq::get(&format!("{uri}/v1/namespaces/{namespace}/tables/{table}"))
            .call()
            .expect("load table")
            .into_json()
            .expect("table response");
    response["metadata"].clone()
}
