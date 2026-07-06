//! The local-Iceberg test rig: MinIO (the warehouse's object store) and an
//! `apache/iceberg-rest-fixture` catalog service in Docker, plus the upstream
//! `iceberg` crate as the **oracle** that creates tables and commits snapshots
//! against them - so the crate under test reads metadata written by an
//! independent implementation.
//!
//! Both containers and the pivot-side environment (`AWS_*`) are brought up once
//! per test binary in a single `OnceLock` init. MinIO is addressed two ways:
//! the REST service (in Docker) reaches it at its bridge-network IP, while the
//! oracle and the crate under test (host processes) use the host-mapped port.
//! The REST service vends its internal endpoint in table configs, which is fine:
//! the oracle's own S3 properties take precedence over vended config, and the
//! pivot side reads storage settings from the environment only. When Docker is
//! unavailable the harness yields `None` and tests skip, staying green offline.

#![allow(dead_code)] // each test binary uses a subset

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use catalog::store::{ObjectPath, ObjectStore, open_store};
use dispatch::{DataFlowDispatcher, Dispatch};
use iceberg::Catalog as _;
use iceberg::CatalogBuilder as _;
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType,
    Schema as OracleSchema, Type as OracleType,
};
use iceberg::transaction::{ApplyTransactionAction as _, Transaction};
use iceberg::{NamespaceIdent, TableCreation, TableIdent};
use iceberg_catalog_rest::{
    REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalog, RestCatalogBuilder,
};
use testcontainers::core::{ContainerPort, IntoContainerPort};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};
use testcontainers_modules::minio::MinIO;

/// The one bucket the warehouse lives in.
pub const BUCKET: &str = "pivot-iceberg-it";
const S3_ACCESS_KEY: &str = "minioadmin";
const S3_SECRET_KEY: &str = "minioadmin";
const S3_REGION: &str = "us-east-1";
const REST_IMAGE: (&str, &str) = ("apache/iceberg-rest-fixture", "1.9.1");

pub struct IcebergHarness {
    /// The REST catalog endpoint, reachable from the host.
    pub rest_uri: String,
    /// `s3://<bucket>/warehouse` - where the REST catalog roots table locations.
    pub warehouse_uri: String,
    /// The S3 endpoint host processes (the oracle and the crate under test) use.
    s3_endpoint: String,
    _minio: Container<MinIO>,
    _rest: Container<GenericImage>,
}

/// The per-binary harness, or `None` when Docker could not provide it (the
/// caller `eprintln!`s and returns, so tests skip rather than fail offline).
pub fn harness() -> Option<&'static IcebergHarness> {
    static HARNESS: OnceLock<Option<IcebergHarness>> = OnceLock::new();
    HARNESS
        .get_or_init(|| match start_harness() {
            Ok(harness) => Some(harness),
            Err(reason) => {
                eprintln!("[iceberg harness] skipping: {reason}");
                None
            }
        })
        .as_ref()
}

fn start_harness() -> Result<IcebergHarness, String> {
    let minio = MinIO::default()
        .start()
        .map_err(|e| format!("MinIO unavailable: {e}"))?;
    let minio_port = minio
        .get_host_port_ipv4(9000)
        .map_err(|e| format!("MinIO port: {e}"))?;
    let minio_bridge_ip = minio
        .get_bridge_ip_address()
        .map_err(|e| format!("MinIO bridge ip: {e}"))?;
    // Host processes go through the mapped port; the REST container reaches
    // MinIO directly on the Docker bridge network.
    let s3_endpoint = format!("http://127.0.0.1:{minio_port}");
    let s3_endpoint_internal = format!("http://{minio_bridge_ip}:9000");
    catalog::test_support::create_s3_bucket(&s3_endpoint, BUCKET)?;

    // The environment the pivot side (catalog::store's S3 backend) reads. Set
    // once here, on one thread, before any test opens a store - the OnceLock
    // publishes the writes with a happens-before to every later reader.
    for (key, value) in [
        ("AWS_ENDPOINT_URL", s3_endpoint.as_str()),
        ("AWS_ACCESS_KEY_ID", S3_ACCESS_KEY),
        ("AWS_SECRET_ACCESS_KEY", S3_SECRET_KEY),
        ("AWS_REGION", S3_REGION),
    ] {
        unsafe { std::env::set_var(key, value) };
    }

    let warehouse_uri = format!("s3://{BUCKET}/warehouse");
    let rest = GenericImage::new(REST_IMAGE.0, REST_IMAGE.1)
        .with_exposed_port(ContainerPort::Tcp(8181))
        .with_env_var("AWS_ACCESS_KEY_ID", S3_ACCESS_KEY)
        .with_env_var("AWS_SECRET_ACCESS_KEY", S3_SECRET_KEY)
        .with_env_var("AWS_REGION", S3_REGION)
        .with_env_var("CATALOG_WAREHOUSE", &warehouse_uri)
        // The image's default sqlite URI is per-connection in-memory: every
        // pooled JDBC connection sees its own empty database, so concurrent
        // requests fail with "no such table". Back it with a file instead.
        .with_env_var(
            "CATALOG_URI",
            "jdbc:sqlite:file:/tmp/iceberg_rest_catalog.db",
        )
        .with_env_var("CATALOG_IO__IMPL", "org.apache.iceberg.aws.s3.S3FileIO")
        .with_env_var("CATALOG_S3_ENDPOINT", &s3_endpoint_internal)
        .with_env_var("CATALOG_S3_PATH__STYLE__ACCESS", "true")
        .start()
        .map_err(|e| format!("{}: {e}", REST_IMAGE.0))?;
    let rest_port = rest
        .get_host_port_ipv4(8181.tcp())
        .map_err(|e| format!("rest port: {e}"))?;
    let rest_uri = format!("http://127.0.0.1:{rest_port}");
    wait_for_rest_catalog(&rest_uri)?;

    Ok(IcebergHarness {
        rest_uri,
        warehouse_uri,
        s3_endpoint,
        _minio: minio,
        _rest: rest,
    })
}

/// Poll until the REST service answers (it boots a JVM, so allow a generous
/// window). `/v1/config` gates on the HTTP server; the follow-up
/// `/v1/namespaces` forces the fixture's backing catalog to initialize its
/// schema once, before parallel tests hit it concurrently.
fn wait_for_rest_catalog(rest_uri: &str) -> Result<(), String> {
    let mut last = String::new();
    for path in ["/v1/config", "/v1/namespaces"] {
        let url = format!("{rest_uri}{path}");
        let ready = (0..240).any(|_| match ureq::get(&url).call() {
            Ok(_) => true,
            Err(e) => {
                last = e.to_string();
                std::thread::sleep(std::time::Duration::from_millis(500));
                false
            }
        });
        if !ready {
            return Err(format!("rest catalog never became ready on {path}: {last}"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The oracle: the upstream iceberg implementation writing through the REST
// catalog. Tables carry the fixed schema (id BIGINT, name VARCHAR).
// ---------------------------------------------------------------------------

pub struct Oracle {
    runtime: tokio::runtime::Runtime,
    catalog: RestCatalog,
    /// Writes the Parquet data files the committed snapshots point at.
    bucket_store: Box<dyn ObjectStore>,
    file_seq: AtomicU64,
}

impl Oracle {
    pub fn connect(harness: &IcebergHarness) -> Self {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let props = HashMap::from([
            (REST_CATALOG_PROP_URI.to_string(), harness.rest_uri.clone()),
            (
                REST_CATALOG_PROP_WAREHOUSE.to_string(),
                harness.warehouse_uri.clone(),
            ),
            ("s3.endpoint".to_string(), harness.s3_endpoint.clone()),
            ("s3.access-key-id".to_string(), S3_ACCESS_KEY.to_string()),
            (
                "s3.secret-access-key".to_string(),
                S3_SECRET_KEY.to_string(),
            ),
            ("s3.region".to_string(), S3_REGION.to_string()),
        ]);
        // iceberg 0.9 requires an explicit storage backend; route the oracle's
        // warehouse I/O through opendal's S3 implementation.
        let storage = iceberg_storage_opendal::OpenDalStorageFactory::S3 {
            configured_scheme: "s3".to_string(),
            customized_credential_load: None,
        };
        let catalog = runtime
            .block_on(
                RestCatalogBuilder::default()
                    .with_storage_factory(std::sync::Arc::new(storage))
                    .load("pivot-test", props),
            )
            .expect("connect oracle rest catalog");
        Self {
            runtime,
            catalog,
            bucket_store: open_store(&format!("s3://{BUCKET}")).expect("open bucket store"),
            file_seq: AtomicU64::new(0),
        }
    }

    /// Create `namespace.table` with the fixed `(id BIGINT, name VARCHAR)`
    /// schema, creating the namespace as needed.
    pub fn create_table(&self, namespace: &str, table: &str) {
        let ns = NamespaceIdent::new(namespace.to_string());
        self.runtime.block_on(async {
            if !self.catalog.namespace_exists(&ns).await.unwrap() {
                self.catalog
                    .create_namespace(&ns, HashMap::new())
                    .await
                    .unwrap();
            }
            let schema = OracleSchema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", OracleType::Primitive(PrimitiveType::Long))
                        .into(),
                    NestedField::optional(2, "name", OracleType::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap();
            let creation = TableCreation::builder()
                .name(table.to_string())
                .schema(schema)
                .build();
            self.catalog.create_table(&ns, creation).await.unwrap();
        });
    }

    /// Commit one snapshot appending `rows` as a fresh Parquet data file:
    /// write the file into the table's location, then fast-append it through
    /// the oracle's transaction API.
    pub fn append_rows(&self, namespace: &str, table: &str, rows: &[(i64, &str)]) {
        let bytes = write_parquet(rows);
        self.runtime.block_on(async {
            let ident = TableIdent::from_strs([namespace, table]).unwrap();
            let table = self.catalog.load_table(&ident).await.unwrap();

            let sequence = self.file_seq.fetch_add(1, Ordering::Relaxed);
            let file_uri = format!(
                "{}/data/part-{sequence}-{}.parquet",
                table.metadata().location(),
                std::process::id()
            );
            let key = file_uri
                .strip_prefix(&format!("s3://{BUCKET}/"))
                .expect("table location lives in the harness bucket");
            self.bucket_store
                .put(&ObjectPath::new(key), &bytes)
                .unwrap();

            let data_file = DataFileBuilder::default()
                .content(DataContentType::Data)
                .file_path(file_uri)
                .file_format(DataFileFormat::Parquet)
                .record_count(rows.len() as u64)
                .file_size_in_bytes(bytes.len() as u64)
                .build()
                .unwrap();
            let transaction = Transaction::new(&table);
            let transaction = transaction
                .fast_append()
                .add_data_files([data_file])
                .apply(transaction)
                .unwrap();
            transaction.commit(&self.catalog).await.unwrap();
        });
    }
}

/// Snappy Parquet bytes for `(id BIGINT, name VARCHAR)` rows.
fn write_parquet(rows: &[(i64, &str)]) -> Vec<u8> {
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let ids: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    ));
    let names: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|(_, name)| *name).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, names]).unwrap();

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    buffer
}

// ---------------------------------------------------------------------------
// Dispatch plumbing and result collectors (the same shapes catalog's tests use).
// ---------------------------------------------------------------------------

/// RAII wrapper around an owned `Dispatch`: derefs to its dispatcher and shuts
/// the workers down on drop.
pub struct DispatchGuard(Option<Dispatch>);

impl Deref for DispatchGuard {
    type Target = DataFlowDispatcher;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap().dispatcher()
    }
}

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        self.0.take().unwrap().exit();
    }
}

pub fn start_dispatch(workers: usize) -> DispatchGuard {
    DispatchGuard(Some(Dispatch::spin_up(workers, 64, None)))
}

pub fn collect_i64s(batches: &[arrow_array::RecordBatch], col: usize) -> Vec<i64> {
    use arrow_array::Int64Array;
    batches
        .iter()
        .flat_map(|batch| {
            let array = batch
                .column(col)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..array.len()).map(move |i| array.value(i))
        })
        .collect()
}

pub fn collect_strings(batches: &[arrow_array::RecordBatch], col: usize) -> Vec<String> {
    use arrow_array::{Array, StringViewArray};
    batches
        .iter()
        .flat_map(|batch| {
            let array = batch
                .column(col)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            (0..array.len()).map(move |i| array.value(i).to_string())
        })
        .collect()
}
