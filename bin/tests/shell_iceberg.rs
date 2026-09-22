//! `pivot open --kind iceberg`: the shell over an Iceberg REST catalog. The
//! catalog is the `apache/iceberg-rest-fixture` image, storing its tables in
//! the MinIO the object-storage harness starts, both in Docker; a table is
//! written through the iceberg crate's own writer and read back through the
//! shell's executor, exactly as a statement typed at the prompt would be. The
//! Docker-backed test returns early when Docker is unreachable, so the suite
//! stays green offline.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, Int64Array, RecordBatch, StringArray, StringViewArray};
use bin::execution::{ExecuteOptions, StatementOutput};
use bin::shell::{ShellInstance, ShellTarget};
use datastore_iceberg::IcebergCatalogConfig;
use datastore_pivot::DEFAULT_REFRESH_INTERVAL;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_storage_opendal::OpenDalStorageFactory;
use object_storage::test_support;
use parquet::file::properties::WriterProperties;
use testcontainers::core::{ContainerPort, Host, IntoContainerPort, WaitFor};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};

/// The bucket the object-storage harness creates, which the fixture stores
/// its warehouse in.
const WAREHOUSE: &str = "s3://pivot-it/shell-iceberg-warehouse";

#[test]
fn an_unreachable_catalog_fails_the_open() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let target = ShellTarget::Iceberg(catalog_config("http://127.0.0.1:1"));

    let result = runtime.block_on(async {
        ShellInstance::open_with_resources(&target, 1, 32, DEFAULT_REFRESH_INTERVAL)
    });

    let error = match result {
        Ok(_) => panic!("the shell opened over a catalog nothing listens at"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("Iceberg REST catalog"),
        "{error}"
    );
}

#[test]
fn a_catalog_table_is_read_through_the_shell_and_cannot_be_written() {
    let Some(fixture) = start_fixture() else {
        return;
    };
    let namespace = fixture.create_table("sales", "orders");
    fixture.append(&namespace, "orders", orders(&[2, 1], &["bob", "ann"]));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let target = ShellTarget::Iceberg(catalog_config(&fixture.catalog_uri));
    let instance = runtime
        .block_on(async {
            ShellInstance::open_with_resources(&target, 1, 32, DEFAULT_REFRESH_INTERVAL)
        })
        .unwrap();

    let rows = runtime.block_on(instance.executor().execute(
        "SELECT id, customer FROM sales.orders ORDER BY id".to_string(),
        ExecuteOptions::default(),
    ));
    let insert = runtime.block_on(instance.executor().execute(
        "INSERT INTO sales.orders VALUES (3, 'cy')".to_string(),
        ExecuteOptions::default(),
    ));

    let StatementOutput::Rows { batches, .. } = rows.unwrap().output else {
        panic!("SELECT did not return rows");
    };
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let customers = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    assert_eq!((ids.value(0), customers.value(0)), (1, "ann"));
    assert_eq!((ids.value(1), customers.value(1)), (2, "bob"));
    assert!(insert.is_err(), "an INSERT into a catalog table succeeded");
    drop(instance);
}

fn catalog_config(uri: &str) -> IcebergCatalogConfig {
    IcebergCatalogConfig {
        uri: uri.to_string(),
        warehouse: None,
        auth: None,
        properties: HashMap::new(),
    }
}

/// The REST fixture over the harness's MinIO, and the writer's view of it.
struct Fixture {
    _container: Container<GenericImage>,
    catalog_uri: String,
    writer: iceberg_catalog_rest::RestCatalog,
    runtime: tokio::runtime::Runtime,
}

/// Bring the fixture up against the harness's MinIO, or `None` (with a note)
/// when Docker is not available, so the test skips rather than fails.
fn start_fixture() -> Option<Fixture> {
    let Some(_backend) = test_support::s3("shell-iceberg") else {
        eprintln!("[shell_iceberg] skipping: MinIO unavailable");
        return None;
    };
    // The harness points the process's `AWS_*` variables at MinIO; the fixture
    // reaches the same endpoint through the Docker host gateway.
    let s3_endpoint = std::env::var("AWS_ENDPOINT_URL").expect("the S3 harness sets its endpoint");
    let access_key = std::env::var("AWS_ACCESS_KEY_ID").unwrap();
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap();
    let region = std::env::var("AWS_REGION").unwrap();
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
        .with_env_var("AWS_ACCESS_KEY_ID", &access_key)
        .with_env_var("AWS_SECRET_ACCESS_KEY", &secret_key)
        .with_env_var("AWS_REGION", &region)
        .with_host("host.docker.internal", Host::HostGateway);
    let container = match image.start() {
        Ok(container) => container,
        Err(error) => {
            eprintln!("[shell_iceberg] skipping: REST fixture unavailable: {error}");
            return None;
        }
    };
    let port = container.get_host_port_ipv4(8181.tcp()).unwrap();
    let catalog_uri = format!("http://localhost:{port}");
    wait_for_catalog(&catalog_uri);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let writer = runtime
        .block_on(
            RestCatalogBuilder::default()
                .with_storage_factory(Arc::new(OpenDalStorageFactory::S3 {
                    customized_credential_load: None,
                }))
                .load(
                    "writer",
                    HashMap::from([
                        ("uri".to_string(), catalog_uri.clone()),
                        ("s3.endpoint".to_string(), s3_endpoint),
                        ("s3.access-key-id".to_string(), access_key),
                        ("s3.secret-access-key".to_string(), secret_key),
                        ("s3.region".to_string(), region),
                        ("s3.path-style-access".to_string(), "true".to_string()),
                    ]),
                ),
        )
        .expect("build the writer catalog");
    Some(Fixture {
        _container: container,
        catalog_uri,
        writer,
        runtime,
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

impl Fixture {
    /// A namespace with one table of [`orders_schema`] created in it.
    fn create_table(&self, namespace: &str, table: &str) -> NamespaceIdent {
        let namespace = NamespaceIdent::new(namespace.to_string());
        self.runtime
            .block_on(self.writer.create_namespace(&namespace, HashMap::new()))
            .expect("create namespace");
        let creation = TableCreation::builder()
            .name(table.to_string())
            .schema(orders_schema())
            .build();
        self.runtime
            .block_on(self.writer.create_table(&namespace, creation))
            .expect("create table");
        namespace
    }

    /// Write `batch` as one data file of `table` and commit it as a new
    /// snapshot, the way an engine appends.
    fn append(&self, namespace: &NamespaceIdent, table: &str, batch: RecordBatch) {
        let ident = TableIdent::new(namespace.clone(), table.to_string());
        self.runtime.block_on(async {
            let table = self.writer.load_table(&ident).await.expect("load table");
            let location_generator = DefaultLocationGenerator::new(table.metadata()).unwrap();
            let file_name_generator =
                DefaultFileNameGenerator::new("part".to_string(), None, DataFileFormat::Parquet);
            let parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::default(),
                table.metadata().current_schema().clone(),
            );
            let rolling_writer = RollingFileWriterBuilder::new_with_default_file_size(
                parquet_writer,
                table.file_io().clone(),
                location_generator,
                file_name_generator,
            );
            let mut writer = DataFileWriterBuilder::new(rolling_writer)
                .build(None)
                .await
                .unwrap();
            writer.write(batch).await.unwrap();
            let files = writer.close().await.unwrap();

            let transaction = Transaction::new(&table);
            let transaction = transaction
                .fast_append()
                .add_data_files(files)
                .apply(transaction)
                .unwrap();
            transaction.commit(&self.writer).await.unwrap();
        });
    }
}

/// `id BIGINT NOT NULL, customer VARCHAR`, with field ids 1 and 2.
fn orders_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "customer",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .unwrap()
}

/// Rows of [`orders_schema`], in its arrow form (with field ids), ready for
/// the iceberg writer.
fn orders(ids: &[i64], customers: &[&str]) -> RecordBatch {
    let schema = Arc::new(schema_to_arrow_schema(&orders_schema()).unwrap());
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(customers.to_vec())),
        ],
    )
    .unwrap()
}
