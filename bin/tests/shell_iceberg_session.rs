//! `pivot open --kind iceberg`: the shell opens an Iceberg REST catalog (the
//! `apache/iceberg-rest-fixture` image, storing its tables in the MinIO the
//! object-storage harness starts, both in Docker) and serves its tables,
//! each namespace as a schema, reading and writing them through the catalog.
//! The catalog test returns early when Docker is unreachable, so the suite
//! stays green offline; the unreachable-catalog test needs nothing.

mod common;

use std::net::TcpListener;

use arrow_schema::DataType;
use bin::execution::{ExecuteOptions, StatementOutput};
use bin::shell::{ShellInstance, ShellTarget};
use common::iceberg_fixture::{create_namespace, create_table, start_fixture};
use datastore_iceberg::IcebergCatalogConfig;
use datastore_pivotlake::DEFAULT_REFRESH_INTERVAL;

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
fn the_shell_reads_and_writes_a_catalog_table() {
    let Some(fixture) = start_fixture("shell_iceberg_session") else {
        return;
    };
    create_namespace(&fixture.uri, "demo");
    create_table(&fixture.uri, "demo", "events", false);
    let target = iceberg_target(fixture.uri.clone());
    let runtime = build_runtime();
    let instance = runtime
        .block_on(async {
            ShellInstance::open_with_resources(&target, 1, 32, DEFAULT_REFRESH_INTERVAL)
        })
        .unwrap();

    let empty = runtime
        .block_on(instance.executor().execute(
            "SELECT id, name FROM demo.events".to_string(),
            ExecuteOptions::default(),
        ))
        .unwrap()
        .output;
    runtime
        .block_on(instance.executor().execute(
            "INSERT INTO demo.events VALUES (1, 'signup')".to_string(),
            ExecuteOptions::default(),
        ))
        .unwrap();
    let written = runtime
        .block_on(instance.executor().execute(
            "SELECT id, name FROM demo.events".to_string(),
            ExecuteOptions::default(),
        ))
        .unwrap()
        .output;

    assert_eq!(instance.location(), fixture.uri);
    let StatementOutput::Rows { columns, batches } = empty else {
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
    let StatementOutput::Rows { batches, .. } = written else {
        panic!("SELECT did not return rows after the INSERT");
    };
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );
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
