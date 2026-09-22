//! `pivot open --kind iceberg`: the shell opens an Iceberg REST catalog (the
//! `apache/iceberg-rest-fixture` image, storing its tables in the MinIO the
//! object-storage harness starts, both in Docker) and serves its tables
//! read-only, each namespace as a schema. The catalog test returns early when
//! Docker is unreachable, so the suite stays green offline; the
//! unreachable-catalog test needs nothing.

use std::net::TcpListener;
use std::time::{Duration, Instant};

use arrow_schema::DataType;
use bin::execution::{ExecuteOptions, StatementOutput};
use bin::shell::{ShellInstance, ShellTarget};
use datastore_iceberg::IcebergCatalogConfig;
use datastore_pivot::DEFAULT_REFRESH_INTERVAL;
use object_storage::test_support;
use testcontainers::core::{ContainerPort, Host, IntoContainerPort, WaitFor};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};

/// Where the fixture stores its tables: a prefix of the bucket the
/// object-storage harness creates.
const WAREHOUSE: &str = "s3://pivot-it/shell-iceberg-warehouse";

#[test]
fn an_unreachable_catalog_fails_the_open() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let target = iceberg_target(format!("http://127.0.0.1:{port}"));
    let runtime = build_runtime();

    let result = runtime.block_on(async {
        ShellInstance::open_with_resources(&target, 1, 32, DEFAULT_REFRESH_INTERVAL)
    });

    let error = match result {
        Ok(_) => panic!("the shell opened over a catalog nobody serves"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("Iceberg REST catalog"),
        "{error}"
    );
}

#[test]
fn the_shell_serves_a_catalog_table_read_only() {
    let Some(fixture) = start_fixture() else {
        return;
    };
    create_namespace(&fixture.uri, "demo");
    create_table(&fixture.uri, "demo", "events");
    let target = iceberg_target(fixture.uri.clone());
    let runtime = build_runtime();
    let instance = runtime
        .block_on(async {
            ShellInstance::open_with_resources(&target, 1, 32, DEFAULT_REFRESH_INTERVAL)
        })
        .unwrap();

    let select = runtime
        .block_on(instance.executor().execute(
            "SELECT id, name FROM demo.events".to_string(),
            ExecuteOptions::default(),
        ))
        .unwrap()
        .output;
    let insert = runtime.block_on(instance.executor().execute(
        "INSERT INTO demo.events VALUES (1, 'signup')".to_string(),
        ExecuteOptions::default(),
    ));

    assert_eq!(instance.location(), fixture.uri);
    let StatementOutput::Rows { columns, batches } = select else {
        panic!("SELECT did not return rows from the catalog table");
    };
    let column_types = columns
        .iter()
        .map(|column| (column.name.as_str(), &column.data_type))
        .collect::<Vec<_>>();
    assert_eq!(
        column_types,
        vec![("id", &DataType::Int64), ("name", &DataType::Utf8View)]
    );
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        0
    );
    assert!(insert.is_err(), "an INSERT into a catalog table succeeded");
}

fn iceberg_target(catalog_uri: String) -> ShellTarget {
    ShellTarget::Iceberg(IcebergCatalogConfig {
        uri: catalog_uri,
        ..Default::default()
    })
}

/// The catalog client blocks on the ambient runtime, which must be
/// multi-threaded for that.
fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

struct Fixture {
    _container: Container<GenericImage>,
    uri: String,
}

/// Bring the REST fixture up against the harness's MinIO, or `None` (with a
/// note) when Docker is not available.
fn start_fixture() -> Option<Fixture> {
    let Some(_backend) = test_support::s3("shell-iceberg") else {
        eprintln!("[shell_iceberg_session] skipping: MinIO unavailable");
        return None;
    };
    // The harness points the process's `AWS_*` variables at MinIO, which is
    // how the shell will read the tables' files; the fixture reaches the same
    // endpoint through the Docker host gateway.
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
            eprintln!("[shell_iceberg_session] skipping: REST fixture unavailable: {error}");
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

fn create_namespace(uri: &str, namespace: &str) {
    ureq::post(&format!("{uri}/v1/namespaces"))
        .send_json(serde_json::json!({ "namespace": [namespace] }))
        .expect("create namespace");
}

/// Create `table` with two columns, `id` (long) and `name` (string), and no
/// snapshot.
fn create_table(uri: &str, namespace: &str, table: &str) {
    ureq::post(&format!("{uri}/v1/namespaces/{namespace}/tables"))
        .send_json(serde_json::json!({
            "name": table,
            "schema": {
                "type": "struct",
                "schema-id": 0,
                "fields": [
                    { "id": 1, "name": "id", "required": false, "type": "long" },
                    { "id": 2, "name": "name", "required": false, "type": "string" },
                ],
            },
        }))
        .expect("create table");
}
