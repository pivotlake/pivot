//! End-to-end: a pivotdb server whose query catalog is an
//! [`IcebergRestCatalog`], queried over the Postgres wire protocol. The full
//! path: SQL -> DuckDB binding against the REST catalog -> scan of the
//! warehouse's Parquet over MinIO -> wire encode. Skips (green) when Docker is
//! unavailable.

mod common;
use common::*;

use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use catalog::ParquetCatalog;
use dispatch::Dispatch;
use iceberg_catalog::{IcebergRestCatalog, IcebergRestConfig};
use server::Server;
use tokio_postgres::{NoTls, SimpleQueryMessage};

/// Start a server on a free port whose queries resolve against the harness's
/// REST catalog (unqualified names in `default_namespace`), and return the
/// port once it is listening.
fn start_server(rest_uri: &str, default_namespace: &str) -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let bind: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    let mut config = IcebergRestConfig::new(rest_uri);
    config.default_namespace = default_namespace.to_string();
    thread::spawn(move || {
        let dispatch = Dispatch::spin_up(2, 64, None);
        let iceberg = IcebergRestCatalog::connect(config, dispatch.dispatcher().clone()).unwrap();
        let parquet_catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let server = Server::new(bind, dispatch, parquet_catalog, vec![], 0, 4)
                .with_query_catalog(Arc::new(iceberg));
            let _ = server.serve(Box::pin(std::future::pending::<()>())).await;
        });
    });

    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(bind).is_ok() {
            return port;
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!("server failed to start listening on {bind}");
}

/// Run `sql` over a fresh pgwire connection and return the data rows as
/// stringified columns.
fn run_query(port: u16, sql: &str) -> Vec<Vec<String>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (client, connection) = tokio_postgres::Config::new()
            .host("127.0.0.1")
            .port(port)
            .user("test")
            .dbname("test")
            .connect(NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client
            .simple_query(sql)
            .await
            .unwrap()
            .into_iter()
            .filter_map(|message| match message {
                SimpleQueryMessage::Row(row) => Some(
                    (0..row.len())
                        .map(|i| row.get(i).unwrap_or("").to_string())
                        .collect(),
                ),
                _ => None,
            })
            .collect()
    })
}

#[test]
fn serves_iceberg_tables_over_the_postgres_wire() {
    let Some(harness) = harness() else { return };
    let oracle = Oracle::connect(harness);
    oracle.create_table("db_pgwire", "events");
    oracle.append_rows("db_pgwire", "events", &[(2, "b"), (1, "a"), (3, "c")]);
    let port = start_server(&harness.rest_uri, "db_pgwire");

    // An unqualified name resolves in the configured default namespace; a
    // quoted full name addresses a namespace explicitly.
    let rows = run_query(port, "SELECT id, name FROM events ORDER BY id");
    let count = run_query(port, r#"SELECT count(*) FROM "db_pgwire.events""#);
    let sum = run_query(port, "SELECT sum(id) FROM events WHERE name <> 'b'");

    assert_eq!(
        rows,
        vec![
            vec!["1".to_string(), "a".to_string()],
            vec!["2".to_string(), "b".to_string()],
            vec!["3".to_string(), "c".to_string()],
        ]
    );
    assert_eq!(count, vec![vec!["3".to_string()]]);
    assert_eq!(sum, vec![vec!["4".to_string()]]);
}
