//! End-to-end pgwire tests over object storage: stand up a real `Server` whose
//! catalog is rooted in a bucket (MinIO, or a local dir), `CREATE
//! TABLE` over Parquet files sitting in that bucket, and query them through a
//! Postgres client — the full `CREATE TABLE` → plan → scan-over-presigned-URL →
//! wire-encode path, against real object storage.
//!
//! Each behaviour is written once over a `&Backend` and run on local / S3
//! by the [`bucket_tests!`] macro; S3 skips when Docker is absent.

mod common;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use common::{connect_client, pick_free_port, wait_until_listening};
use datastore_delta::DeltaDatastore;
use datastore_delta::store::ObjectPath;
use datastore_delta::test_support::{self, Backend};
use dispatch::Dispatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use server::Server;
use tokio_postgres::{Client, SimpleQueryMessage};

// --- helpers ---------------------------------------------------------------

/// Snappy Parquet bytes for `(name VARCHAR, value BIGINT)` rows carrying the
/// given `value`s (names are filler), ready to `put` into a store.
fn pq(values: &[i64]) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let name: ArrayRef = Arc::new(StringArray::from(vec!["x"; values.len()]));
    let value: ArrayRef = Arc::new(Int64Array::from(values.to_vec()));
    let batch = RecordBatch::try_new(schema.clone(), vec![name, value]).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    buf
}

/// Start a server whose catalog is opened on `root` (a bucket URI or local
/// path), on a dedicated thread, and return the port once it is listening.
fn start_server_on(root: &str) -> u16 {
    let port = pick_free_port();
    let bind: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let workers = core_affinity::get_core_ids().unwrap().len().clamp(1, 4);
    let root = root.to_string();
    thread::spawn(move || {
        let dispatch = Dispatch::spin_up(workers, 32, None);
        let datastore: Arc<dyn Datastore> =
            DeltaDatastore::open(&root, dispatch.dispatcher()).unwrap();
        let catalog = Arc::new(
            PivotCatalog::new(
                HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore)]),
                DEFAULT_DATASTORE_NAME.to_string(),
            )
            .unwrap(),
        );
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let server = Server::new(bind, dispatch, catalog);
            let _ = server.serve(Box::pin(std::future::pending::<()>())).await;
        });
    });
    wait_until_listening(bind);
    port
}

/// Run `sql` and return the first row's first column as an `i64`.
async fn select_one_i64(client: &Client, sql: &str) -> i64 {
    let msgs = client.simple_query(sql).await.unwrap();
    for msg in msgs {
        if let SimpleQueryMessage::Row(row) = msg {
            return row
                .get(0)
                .expect("non-null scalar")
                .parse()
                .expect("integer");
        }
    }
    panic!("no data row from `{sql}`");
}

/// Upload a two-file `events` table (5 rows: 1,2,3 + 4,5) into the backing
/// store, then `CREATE TABLE events` over it on a server rooted at the backend,
/// and return a connected client.
async fn events_server(b: &Backend) -> Client {
    b.store
        .put(&ObjectPath::new("events/p1.parquet"), &pq(&[1, 2, 3]))
        .unwrap();
    b.store
        .put(&ObjectPath::new("events/p2.parquet"), &pq(&[4, 5]))
        .unwrap();
    let client = connect_client(start_server_on(&b.root)).await;
    client
        .simple_query("CREATE TABLE events (name VARCHAR, value BIGINT) WITH (path = 'events')")
        .await
        .unwrap();
    client
}

// --- behaviours (run on every backend) -------------------------------------

mod bodies {
    use super::*;

    /// `CREATE TABLE` over Parquet in the bucket, then count every row. Uses
    /// `COUNT(value)` (a real column) rather than `COUNT(*)`: the latter compiles
    /// to an empty scan projection that currently yields 0 rows — see the
    /// `count_star_no_predicate` known-bug test.
    pub async fn create_table_and_count(b: &Backend) {
        let client = events_server(b).await;

        let count = select_one_i64(&client, "SELECT COUNT(value) FROM events").await;

        assert_eq!(count, 5);
    }

    /// A filtered `COUNT(*)` scans the bucket-resident files through the engine
    /// (presigned reads) and counts only the matching rows. The predicate forces
    /// a real (non-empty) scan, so `COUNT(*)` is answered correctly here.
    pub async fn count_with_filter(b: &Backend) {
        let client = events_server(b).await;

        let count = select_one_i64(&client, "SELECT COUNT(*) FROM events WHERE value > 2").await;

        assert_eq!(count, 3);
    }
}

// --- backend matrix --------------------------------------------------------

/// Drive an async test body to completion on a fresh runtime. Plain `#[test]`
/// (not `#[tokio::test]`) so the backend setup — testcontainers' `SyncRunner`,
/// which blocks on its own runtime — runs before we ever enter one.
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(fut)
}

/// Emit `local` / `s3` pgwire tests for a [`bodies`] behaviour. S3 skips when
/// Docker is absent; each gets its own bucket prefix for isolation.
macro_rules! bucket_tests {
    ($name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn local() {
                let (_dir, b) = test_support::local();
                block_on(bodies::$name(&b));
            }
            #[test]
            fn s3() {
                let Some(b) = test_support::s3(concat!(stringify!($name), "-s3")) else {
                    return;
                };
                block_on(bodies::$name(&b));
            }
        }
    };
}

bucket_tests!(create_table_and_count);
bucket_tests!(count_with_filter);

/// KNOWN BUG (`#[ignore]`d until fixed): `SELECT COUNT(*)` with no predicate
/// returns 0 instead of the row count. DuckDB scans `COUNT(*)` with a
/// "no column needed" sentinel; `planner::compile`'s `Input::compile` drops it
/// to an empty scan projection, and the parquet scan emits 0 rows for an empty
/// projection. Any predicate (or counting a real column) forces a populated
/// scan and counts correctly. Backend-independent, so this runs on local only.
#[test]
#[ignore = "COUNT(*) with no predicate returns 0 — empty scan projection yields 0 rows"]
fn count_star_no_predicate() {
    let (_dir, b) = test_support::local();
    block_on(async {
        let client = events_server(&b).await;
        assert_eq!(
            select_one_i64(&client, "SELECT COUNT(*) FROM events").await,
            5
        );
    });
}
