//! End-to-end tests against a real Iceberg REST catalog: the
//! `apache/iceberg-rest-fixture` image, storing its tables in the MinIO the
//! object-storage harness starts, both in Docker. Tables are written through
//! the iceberg crate's own writer and read back through a Pivot catalog and
//! planner, exactly as a query would. Every test skips when Docker is not
//! reachable, like the other object-store tests.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Float64Type, Int64Type};
use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use catalog::metastore::{DEFAULT_USER_NAME, Metastore, UserAuth};
use catalog::{Datastore, PivotCatalog};
use datastore_iceberg::{IcebergCatalogConfig, IcebergDatastore};
use dispatch::{DataFlowDispatcher, Dispatch};
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::spec::{
    DataFile, DataFileFormat, Datum, FormatVersion, Literal, ManifestList, ManifestListWriter,
    ManifestWriterBuilder, NestedField, NestedFieldRef, Operation, PartitionKey, PrimitiveLiteral,
    PrimitiveType, Schema, Snapshot, SnapshotReference, SnapshotRetention, Struct, Summary,
    Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::transform::create_transform_function;
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent, TableUpdate};
use iceberg_catalog_rest::{RestCatalog, RestCatalogBuilder};
use iceberg_storage_opendal::OpenDalStorageFactory;
use object_storage::{
    AmbientExternalStoreFactory, ExternalStoreFactory, ObjectStore, StoreError, test_support,
};
use parquet::file::properties::WriterProperties;
use planner::Planner;
use planner::catalog::SchemaQualifiedTableName;
use testcontainers::core::{ContainerPort, Host, IntoContainerPort, WaitFor};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};

/// The name the Pivot catalog serves the REST catalog's tables under.
const DATASTORE: &str = "lake";
/// The bucket the object-storage harness creates, which the fixture stores
/// its warehouse in.
const WAREHOUSE: &str = "s3://pivot-it/iceberg-warehouse";
/// How often the datastore under test asks the catalog again: short, so a
/// test that commits after opening waits little for the commit to be seen.
const REFRESH_INTERVAL: Duration = Duration::from_millis(200);
/// How long a test waits for a refresh to show a change before giving up.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(15);

/// The REST fixture, the writer's view of it, and the pool the reads run on.
/// One per test binary: the fixture takes seconds to come up, and every test
/// isolates itself in a namespace of its own.
struct Harness {
    _fixture: Container<GenericImage>,
    catalog_uri: String,
    writer: RestCatalog,
    runtime: tokio::runtime::Runtime,
    dispatch: Dispatch,
}

fn harness() -> Option<&'static Harness> {
    static HARNESS: OnceLock<Option<Harness>> = OnceLock::new();
    HARNESS.get_or_init(start).as_ref()
}

/// Bring the fixture up against the harness's MinIO, or `None` (with a note)
/// when Docker is not available, so the tests skip rather than fail.
fn start() -> Option<Harness> {
    let Some(_backend) = test_support::s3("iceberg-harness") else {
        eprintln!("[rest_catalog] skipping: MinIO unavailable");
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

    let fixture = GenericImage::new("apache/iceberg-rest-fixture", "1.9.1")
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
    let fixture = match fixture.start() {
        Ok(fixture) => fixture,
        Err(error) => {
            eprintln!("[rest_catalog] skipping: REST fixture unavailable: {error}");
            return None;
        }
    };
    let port = fixture.get_host_port_ipv4(8181.tcp()).unwrap();
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
    Some(Harness {
        _fixture: fixture,
        catalog_uri,
        writer,
        runtime,
        dispatch: Dispatch::spin_up(2, 64, None),
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

impl Harness {
    fn dispatcher(&self) -> &DataFlowDispatcher {
        self.dispatch.dispatcher()
    }

    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }

    /// A namespace of the test's own, with one table created in it.
    fn namespace(&self, name: &str, table: &str, schema: Schema) -> NamespaceIdent {
        let namespace = NamespaceIdent::new(name.to_string());
        self.block_on(self.writer.create_namespace(&namespace, HashMap::new()))
            .expect("create namespace");
        self.create_table(&namespace, table, schema);
        namespace
    }

    fn create_table(&self, namespace: &NamespaceIdent, table: &str, schema: Schema) {
        let creation = TableCreation::builder()
            .name(table.to_string())
            .schema(schema)
            .build();
        self.block_on(self.writer.create_table(namespace, creation))
            .expect("create table");
    }

    /// A namespace of the test's own with one table of Iceberg format
    /// version 3.
    fn v3_namespace(&self, name: &str, table: &str, schema: Schema) -> NamespaceIdent {
        let namespace = NamespaceIdent::new(name.to_string());
        self.block_on(self.writer.create_namespace(&namespace, HashMap::new()))
            .expect("create namespace");
        let creation = TableCreation::builder()
            .name(table.to_string())
            .schema(schema)
            .properties(HashMap::from([(
                "format-version".to_string(),
                "3".to_string(),
            )]))
            .build();
        self.block_on(self.writer.create_table(&namespace, creation))
            .expect("create table");
        namespace
    }

    /// A namespace of the test's own with one table, partitioned by
    /// `transform` of the field `partition_field_id`, the partition field
    /// named `partition_name`.
    fn partitioned_namespace(
        &self,
        name: &str,
        table: &str,
        schema: Schema,
        partition_field_id: i32,
        partition_name: &str,
        transform: Transform,
    ) -> NamespaceIdent {
        let namespace = NamespaceIdent::new(name.to_string());
        self.block_on(self.writer.create_namespace(&namespace, HashMap::new()))
            .expect("create namespace");
        let spec = UnboundPartitionSpec::builder()
            .add_partition_field(partition_field_id, partition_name, transform)
            .unwrap()
            .build();
        let creation = TableCreation::builder()
            .name(table.to_string())
            .schema(schema)
            .partition_spec(spec)
            .build();
        self.block_on(self.writer.create_table(&namespace, creation))
            .expect("create table");
        namespace
    }

    fn drop_table(&self, namespace: &NamespaceIdent, table: &str) {
        let ident = TableIdent::new(namespace.clone(), table.to_string());
        self.block_on(self.writer.drop_table(&ident))
            .expect("drop table");
    }

    fn drop_namespace(&self, namespace: &NamespaceIdent) {
        self.block_on(self.writer.drop_namespace(namespace))
            .expect("drop namespace");
    }

    fn load(&self, namespace: &NamespaceIdent, table: &str) -> Table {
        let ident = TableIdent::new(namespace.clone(), table.to_string());
        self.block_on(self.writer.load_table(&ident))
            .expect("load table")
    }

    /// Write `batch` as one data file of `table` and commit it as a new
    /// snapshot, the way an engine appends. The batch's schema is the arrow
    /// form of the table's current schema.
    fn append(&self, namespace: &NamespaceIdent, table: &str, batch: RecordBatch) {
        self.append_to_partition(namespace, table, batch, None, WriterProperties::default());
    }

    /// [`append`](Self::append), writing the file in row groups of
    /// `rows_per_group` rows, so one file holds several row groups with
    /// footer statistics of their own.
    fn append_in_row_groups(
        &self,
        namespace: &NamespaceIdent,
        table: &str,
        batch: RecordBatch,
        rows_per_group: usize,
    ) {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_group))
            .build();
        self.append_to_partition(namespace, table, batch, None, properties);
    }

    /// [`append`](Self::append) into the partition whose values are
    /// `partition`, for a partitioned table; every row of `batch` must belong
    /// to it. The files are written with `properties`.
    fn append_to_partition(
        &self,
        namespace: &NamespaceIdent,
        table: &str,
        batch: RecordBatch,
        partition: Option<Vec<Option<Literal>>>,
        properties: WriterProperties,
    ) {
        let table = self.load(namespace, table);
        let partition_key = partition.map(|values| {
            PartitionKey::new(
                table.metadata().default_partition_spec().as_ref().clone(),
                table.metadata().current_schema().clone(),
                Struct::from_iter(values),
            )
        });
        self.block_on(async {
            let files = self
                .write_data_files(&table, batch, partition_key, properties)
                .await;
            let transaction = Transaction::new(&table);
            let transaction = transaction
                .fast_append()
                .add_data_files(files)
                .apply(transaction)
                .unwrap();
            transaction.commit(&self.writer).await.unwrap();
        });
    }

    /// Write `batch` into `table`'s location as data files, uncommitted, the
    /// way the iceberg writer does: what an append and a compaction commit.
    async fn write_data_files(
        &self,
        table: &Table,
        batch: RecordBatch,
        partition_key: Option<PartitionKey>,
        properties: WriterProperties,
    ) -> Vec<DataFile> {
        let location_generator = DefaultLocationGenerator::new(table.metadata()).unwrap();
        let file_name_generator =
            DefaultFileNameGenerator::new("part".to_string(), None, DataFileFormat::Parquet);
        let parquet_writer =
            ParquetWriterBuilder::new(properties, table.metadata().current_schema().clone());
        let rolling_writer = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer,
            table.file_io().clone(),
            location_generator,
            file_name_generator,
        );
        let mut writer = DataFileWriterBuilder::new(rolling_writer)
            .build(partition_key)
            .await
            .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap()
    }

    /// Rewrite every live data file of `table` into one file holding `merged`
    /// and commit that as a `replace` snapshot, the way a compaction does: the
    /// new snapshot's manifest retires the old files and adds the new one.
    /// `merged` must hold exactly the rows of the files it replaces.
    fn compact(&self, namespace: &NamespaceIdent, table: &str, merged: RecordBatch) {
        let loaded = self.load(namespace, table);
        let metadata = loaded.metadata();
        let file_io = loaded.file_io();
        let current = metadata
            .current_snapshot()
            .expect("a table with data has a snapshot");
        let snapshot_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        let sequence_number = metadata.last_sequence_number() + 1;
        let new_snapshot = self.block_on(async {
            let list_bytes = file_io
                .new_input(current.manifest_list())
                .unwrap()
                .read()
                .await
                .unwrap();
            let list =
                ManifestList::parse_with_version(&list_bytes, metadata.format_version()).unwrap();
            let manifest_path = format!("{}/metadata/{snapshot_id}-m0.avro", metadata.location());
            let mut writer = ManifestWriterBuilder::new(
                file_io.new_output(&manifest_path).unwrap(),
                Some(snapshot_id),
                metadata.current_schema().clone(),
                metadata.default_partition_spec().as_ref().clone(),
            )
            .build_v2_data();
            for manifest_file in list.entries() {
                let manifest = manifest_file.load_manifest(file_io).await.unwrap();
                for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                    writer
                        .add_delete_file(
                            entry.data_file().clone(),
                            entry.sequence_number().unwrap(),
                            entry.file_sequence_number,
                        )
                        .unwrap();
                }
            }
            for data_file in self
                .write_data_files(&loaded, merged, None, WriterProperties::default())
                .await
            {
                writer.add_file(data_file, sequence_number).unwrap();
            }
            let manifest = writer.write_manifest_file().await.unwrap();
            let list_path = format!("{}/metadata/snap-{snapshot_id}-1.avro", metadata.location());
            let mut list_writer = ManifestListWriter::v2(
                file_io
                    .new_output(&list_path)
                    .unwrap()
                    .writer()
                    .await
                    .unwrap(),
                snapshot_id,
                Some(current.snapshot_id()),
                sequence_number,
            );
            list_writer
                .add_manifests(vec![manifest].into_iter())
                .unwrap();
            list_writer.close().await.unwrap();
            Snapshot::builder()
                .with_snapshot_id(snapshot_id)
                .with_parent_snapshot_id(Some(current.snapshot_id()))
                .with_sequence_number(sequence_number)
                .with_timestamp_ms(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as i64,
                )
                .with_manifest_list(list_path)
                .with_summary(Summary {
                    operation: Operation::Replace,
                    additional_properties: HashMap::new(),
                })
                .with_schema_id(metadata.current_schema_id())
                .build()
        });
        let updates = [
            TableUpdate::AddSnapshot {
                snapshot: new_snapshot,
            },
            TableUpdate::SetSnapshotRef {
                ref_name: "main".to_string(),
                reference: SnapshotReference::new(
                    snapshot_id,
                    SnapshotRetention::branch(None, None, None),
                ),
            },
        ];
        let body = serde_json::json!({
            "requirements": [
                {"type": "assert-ref-snapshot-id", "ref": "main", "snapshot-id": current.snapshot_id()}
            ],
            "updates": updates.iter().map(|update| serde_json::to_value(update).unwrap()).collect::<Vec<_>>(),
        });
        ureq::post(&format!(
            "{}/v1/namespaces/{}/tables/{table}",
            self.catalog_uri,
            namespace.to_url_string()
        ))
        .send_json(body)
        .expect("commit the compaction");
    }

    /// Replace `table`'s current schema through the REST protocol, as an
    /// engine's `ALTER TABLE` does: the fixture assigns the new schema its id
    /// and makes it current in one commit.
    fn set_schema(&self, namespace: &NamespaceIdent, table: &str, schema: &Schema) {
        let loaded = self.load(namespace, table);
        let body = serde_json::json!({
            "requirements": [
                {"type": "assert-table-uuid", "uuid": loaded.metadata().uuid().to_string()}
            ],
            "updates": [
                {
                    "action": "add-schema",
                    "schema": serde_json::to_value(schema).unwrap(),
                    "last-column-id": schema.highest_field_id(),
                },
                {"action": "set-current-schema", "schema-id": -1}
            ]
        });
        ureq::post(&format!(
            "{}/v1/namespaces/{}/tables/{table}",
            self.catalog_uri,
            namespace.to_url_string()
        ))
        .send_json(body)
        .expect("update the table's schema");
    }

    /// The Pivot side: a fresh datastore over the fixture, served as the
    /// default datastore of a one-datastore catalog, with a planner over it.
    fn pivot(&self) -> (Arc<PivotCatalog>, Planner) {
        self.pivot_with(HashMap::new(), Arc::new(AmbientExternalStoreFactory))
    }

    /// A Pivot catalog over the fixture, with `properties` passed to the REST
    /// client and `store_factory` opening the roots no credentials were
    /// vended for.
    fn pivot_with(
        &self,
        properties: HashMap<String, String>,
        store_factory: Arc<dyn ExternalStoreFactory>,
    ) -> (Arc<PivotCatalog>, Planner) {
        let config = IcebergCatalogConfig {
            uri: self.catalog_uri.clone(),
            properties,
            ..IcebergCatalogConfig::default()
        };
        let datastore = IcebergDatastore::open(
            DATASTORE,
            &config,
            store_factory,
            self.dispatcher(),
            REFRESH_INTERVAL,
        )
        .expect("open the iceberg datastore");
        let catalog = PivotCatalog::new(
            HashMap::from([(DATASTORE.to_string(), datastore as Arc<dyn Datastore>)]),
            DATASTORE.to_string(),
            Arc::new(TrustMetastore),
        )
        .unwrap();
        // The refresh task spawns onto the ambient runtime, as on the server.
        let _runtime = self.runtime.enter();
        catalog.start();
        let planner =
            Planner::from_datastore_names(vec![DATASTORE.to_string()], DATASTORE.to_string())
                .unwrap();
        (Arc::new(catalog), planner)
    }

    fn query(&self, catalog: &PivotCatalog, planner: &mut Planner, sql: &str) -> Vec<RecordBatch> {
        self.try_query(catalog, planner, sql).unwrap()
    }

    fn try_query(
        &self,
        catalog: &PivotCatalog,
        planner: &mut Planner,
        sql: &str,
    ) -> Result<Vec<RecordBatch>, String> {
        let transaction = catalog.begin_transaction();
        planner
            .plan(sql, transaction.clone())
            .map_err(|error| error.to_string())?
            .compile(self.dispatcher(), transaction.as_ref())
            .map_err(|error| error.to_string())?
            .collect()
            .map_err(|error| error.to_string())
    }

    /// Run `sql` until it fails to plan, and return the error; panic once
    /// [`REFRESH_TIMEOUT`] passes: how a test waits for a background refresh
    /// to notice a table or a schema is gone.
    fn plan_error_until(&self, catalog: &PivotCatalog, planner: &mut Planner, sql: &str) -> String {
        let deadline = Instant::now() + REFRESH_TIMEOUT;
        loop {
            if let Err(error) = self.try_query(catalog, planner, sql) {
                return error;
            }
            assert!(
                Instant::now() < deadline,
                "`{sql}` still planned after {REFRESH_TIMEOUT:?}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Run `sql` until its result satisfies `is_seen`, or panic once
    /// [`REFRESH_TIMEOUT`] passes: how a test waits for a background refresh
    /// to pick up a change made after the datastore opened.
    fn query_until(
        &self,
        catalog: &PivotCatalog,
        planner: &mut Planner,
        sql: &str,
        is_seen: impl Fn(&[RecordBatch]) -> bool,
    ) -> Vec<RecordBatch> {
        let deadline = Instant::now() + REFRESH_TIMEOUT;
        loop {
            let result = self.try_query(catalog, planner, sql);
            if let Ok(batches) = &result
                && is_seen(batches)
            {
                return result.unwrap();
            }
            assert!(
                Instant::now() < deadline,
                "`{sql}` did not see the change within {REFRESH_TIMEOUT:?}: {result:?}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn plan_error(&self, catalog: &PivotCatalog, planner: &mut Planner, sql: &str) -> String {
        let transaction = catalog.begin_transaction();
        match planner.plan(sql, transaction) {
            Ok(_) => panic!("`{sql}` planned, but should have failed"),
            Err(error) => error.to_string(),
        }
    }
}

/// A store factory that opens nothing: what a load falls back to when the
/// catalog vended no credentials, so a test passes only when the credentials
/// in the table's storage properties were the ones used.
#[derive(Debug)]
struct NoStoreFactory;

impl ExternalStoreFactory for NoStoreFactory {
    fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
        Err(StoreError::Config(format!(
            "no credentials were vended for `{root_uri}`"
        )))
    }
}

/// A metastore serving no datastores and only the built-in trusted user: the
/// catalog under test is built directly from an already-open datastore.
#[derive(Debug)]
struct TrustMetastore;

impl Metastore for TrustMetastore {
    fn open_datastores(
        &self,
        _dispatcher: &DataFlowDispatcher,
    ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        Ok(HashMap::new())
    }

    fn default_datastore_name(&self) -> &str {
        DATASTORE
    }

    fn user_auth(&self, username: &str) -> Option<UserAuth> {
        (username == DEFAULT_USER_NAME).then_some(UserAuth::Trust)
    }
}

/// An optional top-level field of a primitive type.
fn field(id: i32, name: &str, primitive: PrimitiveType) -> NestedFieldRef {
    Arc::new(NestedField::optional(id, name, Type::Primitive(primitive)))
}

/// A schema of `fields` under `schema_id`, its first field required (the
/// table's key).
fn schema(schema_id: i32, fields: Vec<NestedFieldRef>) -> Schema {
    let mut fields = fields;
    let key = fields.remove(0);
    fields.insert(
        0,
        Arc::new(NestedField::required(
            key.id,
            &key.name,
            *key.field_type.clone(),
        )),
    );
    Schema::builder()
        .with_schema_id(schema_id)
        .with_fields(fields)
        .build()
        .unwrap()
}

/// `id BIGINT NOT NULL, customer VARCHAR, amount DOUBLE`, with field ids 1..3.
fn orders_schema() -> Schema {
    schema(
        0,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "customer", PrimitiveType::String),
            field(3, "amount", PrimitiveType::Double),
        ],
    )
}

/// Rows of [`orders_schema`], in its arrow form (with field ids), ready for
/// the iceberg writer.
fn orders(ids: &[i64], customers: &[&str], amounts: &[f64]) -> RecordBatch {
    let schema = Arc::new(schema_to_arrow_schema(&orders_schema()).unwrap());
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(customers.to_vec())),
            Arc::new(Float64Array::from(amounts.to_vec())),
        ],
    )
    .unwrap()
}

/// `id BIGINT NOT NULL, at TIMESTAMP`, with field ids 1..2.
fn events_schema() -> Schema {
    schema(
        0,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "at", PrimitiveType::Timestamp),
        ],
    )
}

/// Rows of [`events_schema`], `at` in microseconds since the epoch, in its
/// arrow form ready for the iceberg writer.
fn events(ids: &[i64], at_micros: &[i64]) -> RecordBatch {
    let schema = Arc::new(schema_to_arrow_schema(&events_schema()).unwrap());
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(TimestampMicrosecondArray::from(at_micros.to_vec())),
        ],
    )
    .unwrap()
}

/// The bucket `transform`, a bucket transform, puts `id` in.
fn bucket_of(transform: &Transform, id: i64) -> i32 {
    let bucket = create_transform_function(transform)
        .unwrap()
        .transform_literal(&Datum::long(id))
        .unwrap()
        .expect("a long has a bucket");
    let PrimitiveLiteral::Int(bucket) = *bucket.literal() else {
        panic!("a bucket is an int");
    };
    bucket
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// Column `column` of `batches`, as one value per row.
fn column<T: ArrowPrimitiveType>(batches: &[RecordBatch], column: usize) -> Vec<Option<T::Native>> {
    batches
        .iter()
        .flat_map(|batch| batch.column(column).as_primitive::<T>().iter())
        .collect()
}

fn int64_column(batches: &[RecordBatch], column_index: usize) -> Vec<Option<i64>> {
    column::<Int64Type>(batches, column_index)
}

fn float64_column(batches: &[RecordBatch], column_index: usize) -> Vec<Option<f64>> {
    column::<Float64Type>(batches, column_index)
}

fn string_column(batches: &[RecordBatch], column: usize) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_string_view()
                .iter()
                .map(|value| value.map(str::to_string))
        })
        .collect()
}

#[test]
fn a_catalog_table_reads_through_a_pivot_query() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("sales", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2, 3], &["ann", "bob", "cy"], &[10.0, 20.0, 30.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[4, 5], &["dee", "eve"], &[40.0, 50.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), sum(amount) FROM sales.orders WHERE customer <> 'bob'",
    );

    assert_eq!(int64_column(&batches, 0), [Some(4)]);
    assert_eq!(float64_column(&batches, 1), [Some(130.0)]);
}

#[test]
fn a_commit_after_the_first_query_is_read_once_the_catalog_refreshes() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("growing", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    let (catalog, mut planner) = harness.pivot();
    let before = harness.query(&catalog, &mut planner, "SELECT id FROM growing.orders");

    harness.append(
        &namespace,
        "orders",
        orders(&[2, 3], &["bob", "cy"], &[2.0, 3.0]),
    );
    let after = harness.query_until(
        &catalog,
        &mut planner,
        "SELECT id FROM growing.orders ORDER BY id",
        |batches| total_rows(batches) == 3,
    );

    assert_eq!(total_rows(&before), 1);
    assert_eq!(int64_column(&after, 0), [Some(1), Some(2), Some(3)]);
}

#[test]
fn a_table_created_after_the_datastore_opened_is_served_once_the_catalog_refreshes() {
    let Some(harness) = harness() else { return };
    let (catalog, mut planner) = harness.pivot();
    let namespace = harness.namespace("late", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));

    let batches = harness.query_until(
        &catalog,
        &mut planner,
        "SELECT id FROM late.orders",
        |batches| total_rows(batches) == 1,
    );

    assert_eq!(int64_column(&batches, 0), [Some(1)]);
}

#[test]
fn renamed_and_added_columns_read_by_field_id() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("evolving", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    // `customer` becomes `buyer`, `amount` is dropped, and `score` is added,
    // all under the field ids the first file was written with.
    let evolved = schema(
        1,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "buyer", PrimitiveType::String),
            field(4, "score", PrimitiveType::Long),
        ],
    );
    harness.set_schema(&namespace, "orders", &evolved);
    let evolved_arrow = Arc::new(schema_to_arrow_schema(&evolved).unwrap());
    harness.append(
        &namespace,
        "orders",
        RecordBatch::try_new(
            evolved_arrow,
            vec![
                Arc::new(Int64Array::from(vec![3])),
                Arc::new(StringArray::from(vec!["cy"])),
                Arc::new(Int64Array::from(vec![7])),
            ],
        )
        .unwrap(),
    );
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, buyer, score FROM evolving.orders ORDER BY id",
    );

    assert_eq!(int64_column(&batches, 0), [Some(1), Some(2), Some(3)]);
    assert_eq!(
        string_column(&batches, 1),
        [
            Some("ann".to_string()),
            Some("bob".to_string()),
            Some("cy".to_string())
        ]
    );
    assert_eq!(int64_column(&batches, 2), [None, None, Some(7)]);
}

#[test]
fn a_v3_table_reads_with_null_for_a_column_its_older_files_predate() {
    let Some(harness) = harness() else { return };
    let namespace = harness.v3_namespace("v3", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    let evolved = schema(
        1,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "customer", PrimitiveType::String),
            field(3, "amount", PrimitiveType::Double),
            field(4, "score", PrimitiveType::Long),
        ],
    );
    harness.set_schema(&namespace, "orders", &evolved);
    let evolved_arrow = Arc::new(schema_to_arrow_schema(&evolved).unwrap());
    harness.append(
        &namespace,
        "orders",
        RecordBatch::try_new(
            evolved_arrow,
            vec![
                Arc::new(Int64Array::from(vec![3])),
                Arc::new(StringArray::from(vec!["cy"])),
                Arc::new(Float64Array::from(vec![3.0])),
                Arc::new(Int64Array::from(vec![9])),
            ],
        )
        .unwrap(),
    );
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, score FROM v3.orders ORDER BY id",
    );

    assert_eq!(
        harness
            .load(&namespace, "orders")
            .metadata()
            .format_version(),
        FormatVersion::V3
    );
    assert_eq!(int64_column(&batches, 0), [Some(1), Some(2), Some(3)]);
    assert_eq!(int64_column(&batches, 1), [None, None, Some(9)]);
}

#[test]
fn a_column_with_an_initial_default_refuses_the_table_by_name() {
    let Some(harness) = harness() else { return };
    let scored = schema(
        0,
        vec![
            field(1, "id", PrimitiveType::Long),
            Arc::new(
                NestedField::optional(2, "score", Type::Primitive(PrimitiveType::Long))
                    .with_initial_default(Literal::long(7)),
            ),
        ],
    );
    let _namespace = harness.v3_namespace("defaults", "orders", scored);
    let (catalog, mut planner) = harness.pivot();

    let error = harness.plan_error(&catalog, &mut planner, "SELECT id FROM defaults.orders");

    assert!(
        error.contains("score") && error.contains("initial default"),
        "{error}"
    );
}

#[test]
fn a_query_over_only_a_newer_column_returns_null_for_older_files() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("widened", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    let mut fields = orders_schema().as_struct().fields().to_vec();
    fields.push(field(4, "score", PrimitiveType::Long));
    let widened = schema(1, fields);
    harness.set_schema(&namespace, "orders", &widened);
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), count(score) FROM widened.orders",
    );

    assert_eq!(int64_column(&batches, 0), [Some(2)]);
    assert_eq!(int64_column(&batches, 1), [Some(0)]);
}

#[test]
fn a_predicate_on_an_added_column_keeps_the_files_that_predate_it() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("evolved", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    let mut fields = orders_schema().as_struct().fields().to_vec();
    fields.push(field(4, "score", PrimitiveType::Long));
    let widened = schema(1, fields);
    harness.set_schema(&namespace, "orders", &widened);
    let widened_arrow = Arc::new(schema_to_arrow_schema(&widened).unwrap());
    harness.append(
        &namespace,
        "orders",
        RecordBatch::try_new(
            widened_arrow,
            vec![
                Arc::new(Int64Array::from(vec![3])),
                Arc::new(StringArray::from(vec!["cy"])),
                Arc::new(Float64Array::from(vec![3.0])),
                Arc::new(Int64Array::from(vec![9])),
            ],
        )
        .unwrap(),
    );
    let (catalog, mut planner) = harness.pivot();

    // The first file's manifest knows nothing of `score`: no bound and no
    // null count. It reads as NULL there, so a comparison finds only the
    // newer file's row, IS NULL finds the older rows, and the extremes come
    // from the one file that holds the column.
    let above = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM evolved.orders WHERE score > 5",
    );
    let missing = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM evolved.orders WHERE score IS NULL ORDER BY id",
    );
    let extremes = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(score), max(score), count(score) FROM evolved.orders",
    );

    assert_eq!(int64_column(&above, 0), [Some(3)]);
    assert_eq!(int64_column(&missing, 0), [Some(1), Some(2)]);
    assert_eq!(int64_column(&extremes, 0), [Some(9)]);
    assert_eq!(int64_column(&extremes, 1), [Some(9)]);
    assert_eq!(int64_column(&extremes, 2), [Some(1)]);
}

#[test]
fn missing_tables_and_nested_namespaces_do_not_bind() {
    let Some(harness) = harness() else { return };
    let _namespace = harness.namespace("present", "orders", orders_schema());
    let nested = NamespaceIdent::from_strs(["present", "inner"]).unwrap();
    harness
        .block_on(harness.writer.create_namespace(&nested, HashMap::new()))
        .unwrap();
    harness
        .block_on(
            harness.writer.create_table(
                &nested,
                TableCreation::builder()
                    .name("orders".to_string())
                    .schema(orders_schema())
                    .build(),
            ),
        )
        .unwrap();
    let (catalog, mut planner) = harness.pivot();

    let missing_table = harness.plan_error(&catalog, &mut planner, "SELECT * FROM present.nope");
    let nested_namespace = harness.plan_error(
        &catalog,
        &mut planner,
        r#"SELECT * FROM "present.inner".orders"#,
    );
    let present = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM present.orders",
    );

    assert!(missing_table.contains("nope"), "{missing_table}");
    assert!(
        nested_namespace.contains("present.inner"),
        "{nested_namespace}"
    );
    assert_eq!(int64_column(&present, 0), [Some(0)]);
}

#[test]
fn a_column_pivot_cannot_represent_refuses_the_table_by_name() {
    let Some(harness) = harness() else { return };
    let tokens = schema(
        0,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "token", PrimitiveType::Uuid),
        ],
    );
    let _namespace = harness.namespace("odd", "tokens", tokens);
    let (catalog, mut planner) = harness.pivot();

    let error = harness.plan_error(&catalog, &mut planner, "SELECT id FROM odd.tokens");

    assert!(error.contains("token") && error.contains("uuid"), "{error}");
}

#[test]
fn s3_keys_in_the_tables_storage_properties_open_its_stores() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("vended", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    // The fixture vends nothing, so the keys enter the same merged storage
    // properties through the client's own configuration instead.
    let keys = HashMap::from([
        (
            "s3.access-key-id".to_string(),
            std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
        ),
        (
            "s3.secret-access-key".to_string(),
            std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
        ),
        (
            "s3.region".to_string(),
            std::env::var("AWS_REGION").unwrap(),
        ),
        (
            "s3.endpoint".to_string(),
            std::env::var("AWS_ENDPOINT_URL").unwrap(),
        ),
    ]);
    let (catalog, mut planner) = harness.pivot_with(keys, Arc::new(NoStoreFactory));

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM vended.orders ORDER BY id",
    );

    assert_eq!(int64_column(&batches, 0), [Some(1), Some(2)]);
}

#[test]
fn without_vended_credentials_the_store_factory_opens_the_roots() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("unvended", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(NoStoreFactory));

    let error = harness.plan_error(&catalog, &mut planner, "SELECT id FROM unvended.orders");

    assert!(error.contains("no credentials were vended"), "{error}");
}

#[test]
fn the_datastore_describes_its_tables_to_the_system_catalog() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("described", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2, 3], &["ann", "bob", "cy"], &[1.0, 2.0, 3.0]),
    );
    let (catalog, _) = harness.pivot();
    let datastore = catalog.get_datastore(DATASTORE).unwrap().clone();

    let tables = datastore.begin_transaction().tables().unwrap();

    let orders = tables
        .iter()
        .find(|table| table.name == SchemaQualifiedTableName::new("described", "orders"))
        .expect("the described namespace's table is listed");
    assert_eq!(orders.total_rows, 3);
    assert_eq!(orders.files.len(), 1);
    assert!(
        orders.files[0].path.starts_with(WAREHOUSE),
        "{}",
        orders.files[0].path
    );
    assert_eq!(
        orders
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        ["id", "customer", "amount"]
    );
    assert!(
        orders.files[0]
            .min_max_stats
            .contains(r#""id":{"min":1,"max":3}"#),
        "{}",
        orders.files[0].min_max_stats
    );
}

#[test]
fn a_dropped_table_is_gone_once_the_catalog_refreshes() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("dropping", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    let (catalog, mut planner) = harness.pivot();
    let before = harness.query(&catalog, &mut planner, "SELECT id FROM dropping.orders");

    harness.drop_table(&namespace, "orders");
    let error = harness.plan_error_until(&catalog, &mut planner, "SELECT id FROM dropping.orders");

    assert_eq!(total_rows(&before), 1);
    assert!(error.contains("orders"), "{error}");
}

#[test]
fn a_dropped_schema_is_gone_once_the_catalog_refreshes() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("vanishing", "orders", orders_schema());
    let (catalog, mut planner) = harness.pivot();
    let before = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM vanishing.orders",
    );

    harness.drop_table(&namespace, "orders");
    harness.drop_namespace(&namespace);
    let error = harness.plan_error_until(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM vanishing.orders",
    );

    assert_eq!(int64_column(&before, 0), [Some(0)]);
    assert!(error.contains("vanishing"), "{error}");
}

#[test]
fn a_table_that_could_not_be_indexed_is_left_out_of_the_listing() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("unindexed", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    // Every table's store fails to open, so none is indexed.
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(NoStoreFactory));
    let datastore = catalog.get_datastore(DATASTORE).unwrap().clone();

    let listed = datastore.begin_transaction().tables().unwrap();
    let error = harness.plan_error(&catalog, &mut planner, "SELECT id FROM unindexed.orders");

    assert!(
        !listed
            .iter()
            .any(|table| table.name == SchemaQualifiedTableName::new("unindexed", "orders")),
        "an unindexed table was listed"
    );
    assert!(error.contains("no credentials were vended"), "{error}");
}

#[test]
fn a_nan_row_survives_a_comparison_the_files_bounds_exclude() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("nan", "orders", orders_schema());
    // The file's `amount` bounds are [1.0, 1.0]: statistics leave NaN out.
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, f64::NAN]),
    );
    let (catalog, mut planner) = harness.pivot();

    // NaN is greater than every value, so the row matches and the file, and
    // its row group, must be read.
    let above = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM nan.orders WHERE amount > 2",
    );
    let below = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM nan.orders WHERE amount < 0",
    );

    assert_eq!(int64_column(&above, 0), [Some(2)]);
    assert_eq!(int64_column(&below, 0), [Some(0)]);
}

#[test]
fn a_filter_prunes_across_several_files() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("multi", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[5, 6], &["ed", "flo"], &[5.0, 6.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    let above = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), sum(id) FROM multi.orders WHERE id > 2",
    );
    let middle = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM multi.orders WHERE id BETWEEN 3 AND 4 ORDER BY id",
    );

    assert_eq!(int64_column(&above, 0), [Some(4)]);
    assert_eq!(int64_column(&above, 1), [Some(18)]);
    assert_eq!(int64_column(&middle, 0), [Some(3), Some(4)]);
}

#[test]
fn a_predicate_no_file_can_match_returns_an_empty_result_with_its_columns() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("nomatch", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    // Every file's `id` bounds sit below 100, so the manifests prune them all
    // and no footer is read; the scan still shapes the projected columns.
    let count = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM nomatch.orders WHERE id > 100",
    );
    let rows = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, customer FROM nomatch.orders WHERE id > 100",
    );

    assert_eq!(int64_column(&count, 0), [Some(0)]);
    assert_eq!(total_rows(&rows), 0);
    assert_eq!(
        rows[0]
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        ["id", "customer"]
    );
}

#[test]
fn a_predicate_keeping_one_file_reads_it_and_late_materializes_from_it() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("onefile", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[5, 6], &["ed", "flo"], &[5.0, 6.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    // Only the third file's bounds admit `id = 5`; the ordered limit then
    // materializes its row from the same pruned view the scan read.
    let one = harness.query(
        &catalog,
        &mut planner,
        "SELECT customer FROM onefile.orders WHERE id = 5",
    );
    let ranked = harness.query(
        &catalog,
        &mut planner,
        "SELECT customer FROM onefile.orders WHERE id > 2 ORDER BY id LIMIT 1",
    );

    assert_eq!(string_column(&one, 0), [Some("ed".to_string())]);
    assert_eq!(string_column(&ranked, 0), [Some("cy".to_string())]);
}

#[test]
fn min_and_max_without_a_filter_come_from_the_footers() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("bounds", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[5, 6], &["ed", "flo"], &[5.0, 6.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    // An unfiltered MIN/MAX over plain columns is answered from the row-group
    // statistics of every file, with no data page read. The manifest bounds
    // are not used: the format lets them be looser than the values.
    let extremes = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id), min(customer), max(customer), min(amount), max(amount) \
         FROM bounds.orders",
    );

    assert_eq!(int64_column(&extremes, 0), [Some(1)]);
    assert_eq!(int64_column(&extremes, 1), [Some(6)]);
    assert_eq!(string_column(&extremes, 2), [Some("ann".to_string())]);
    assert_eq!(string_column(&extremes, 3), [Some("flo".to_string())]);
    assert_eq!(float64_column(&extremes, 4), [Some(1.0)]);
    assert_eq!(float64_column(&extremes, 5), [Some(6.0)]);
}

#[test]
fn min_and_max_under_a_filter_come_from_the_pruned_scan() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("filtered", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[5, 6], &["ed", "flo"], &[5.0, 6.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    // A filter keeps the aggregate off the stats shortcut: each answer is
    // narrowed to the rows the pruned scan reads, not the whole table's bounds.
    let above = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id), count(*) FROM filtered.orders WHERE id > 2",
    );
    let one_file = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id), min(customer) FROM filtered.orders WHERE id BETWEEN 3 AND 4",
    );
    let none = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id), count(*) FROM filtered.orders WHERE id > 100",
    );

    assert_eq!(int64_column(&above, 0), [Some(3)]);
    assert_eq!(int64_column(&above, 1), [Some(6)]);
    assert_eq!(int64_column(&above, 2), [Some(4)]);
    assert_eq!(int64_column(&one_file, 0), [Some(3)]);
    assert_eq!(int64_column(&one_file, 1), [Some(4)]);
    assert_eq!(string_column(&one_file, 2), [Some("cy".to_string())]);
    assert_eq!(int64_column(&none, 0), [None]);
    assert_eq!(int64_column(&none, 1), [None]);
    assert_eq!(int64_column(&none, 2), [Some(0)]);
}

#[test]
fn row_groups_of_a_surviving_file_are_pruned_by_their_own_stats() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("grouped", "orders", orders_schema());
    // One file of three row groups, two rows each: its manifest bounds span
    // ids 1 to 6, so no filter below prunes the file itself; only its row
    // groups' footer statistics can narrow what is read.
    harness.append_in_row_groups(
        &namespace,
        "orders",
        orders(
            &[1, 2, 3, 4, 5, 6],
            &["ann", "bob", "cy", "di", "ed", "flo"],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        2,
    );
    let (catalog, mut planner) = harness.pivot();

    let all = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), min(id), max(id) FROM grouped.orders",
    );
    let middle = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, customer FROM grouped.orders WHERE id BETWEEN 3 AND 4 ORDER BY id",
    );
    let top = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), min(id), min(customer) FROM grouped.orders WHERE id > 4",
    );

    assert_eq!(int64_column(&all, 0), [Some(6)]);
    assert_eq!(int64_column(&all, 1), [Some(1)]);
    assert_eq!(int64_column(&all, 2), [Some(6)]);
    assert_eq!(int64_column(&middle, 0), [Some(3), Some(4)]);
    assert_eq!(
        string_column(&middle, 1),
        [Some("cy".to_string()), Some("di".to_string())]
    );
    assert_eq!(int64_column(&top, 0), [Some(2)]);
    assert_eq!(int64_column(&top, 1), [Some(5)]);
    assert_eq!(string_column(&top, 2), [Some("ed".to_string())]);
}

#[test]
fn an_ordered_limit_reads_the_row_it_selects_across_files() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("ranked", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 9.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, customer FROM ranked.orders ORDER BY amount DESC LIMIT 1",
    );

    assert_eq!(int64_column(&batches, 0), [Some(2)]);
    assert_eq!(string_column(&batches, 1), [Some("bob".to_string())]);
}

#[test]
fn a_partitioned_table_reports_its_partition_and_reads_by_it() {
    let Some(harness) = harness() else { return };
    let namespace = harness.partitioned_namespace(
        "parted",
        "orders",
        orders_schema(),
        2,
        "customer",
        Transform::Identity,
    );
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "ann"], &[1.0, 2.0]),
        Some(vec![Some(Literal::string("ann"))]),
        WriterProperties::default(),
    );
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[3], &["bob"], &[3.0]),
        Some(vec![Some(Literal::string("bob"))]),
        WriterProperties::default(),
    );
    let (catalog, mut planner) = harness.pivot();
    let datastore = catalog.get_datastore(DATASTORE).unwrap().clone();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM parted.orders WHERE customer = 'ann'",
    );
    let tables = datastore.begin_transaction().tables().unwrap();

    assert_eq!(int64_column(&batches, 0), [Some(2)]);
    let orders = tables
        .iter()
        .find(|table| table.name == SchemaQualifiedTableName::new("parted", "orders"))
        .expect("the partitioned table is listed");
    assert_eq!(orders.partition_by, ["customer"]);
    let mut partitions: Vec<&str> = orders
        .files
        .iter()
        .map(|file| file.partition.as_str())
        .collect();
    partitions.sort_unstable();
    assert_eq!(partitions, ["customer=ann", "customer=bob"]);
    assert!(
        orders
            .columns
            .iter()
            .any(|column| column.name == "customer" && column.is_partition_key),
        "customer is the partition key"
    );
}

#[test]
fn a_compacted_table_reads_only_the_rewritten_file() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("compacted", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    let (catalog, mut planner) = harness.pivot();
    let datastore = catalog.get_datastore(DATASTORE).unwrap().clone();
    let name = SchemaQualifiedTableName::new("compacted", "orders");
    let files_of = |datastore: &Arc<dyn Datastore>| {
        let tables = datastore.clone().begin_transaction().tables().unwrap();
        let table = tables.iter().find(|table| table.name == name).unwrap();
        table.files.len()
    };
    let before = files_of(&datastore);

    harness.compact(
        &namespace,
        "orders",
        orders(
            &[1, 2, 3, 4],
            &["ann", "bob", "cy", "di"],
            &[1.0, 2.0, 3.0, 4.0],
        ),
    );
    let deadline = Instant::now() + REFRESH_TIMEOUT;
    while files_of(&datastore) != 1 {
        assert!(Instant::now() < deadline, "the compaction was not seen");
        std::thread::sleep(Duration::from_millis(50));
    }
    let after = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), sum(id) FROM compacted.orders",
    );

    assert_eq!(before, 2);
    assert_eq!(int64_column(&after, 0), [Some(4)]);
    assert_eq!(int64_column(&after, 1), [Some(10)]);
}

#[test]
fn a_bucket_partition_prunes_files_by_the_constants_bucket() {
    let Some(harness) = harness() else { return };
    let buckets = Transform::Bucket(8);
    let namespace = harness.partitioned_namespace(
        "bucketed",
        "orders",
        orders_schema(),
        1,
        "id_bucket",
        buckets,
    );
    let bucket_of_5 = bucket_of(&buckets, 5);
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[5], &["ann"], &[5.0]),
        Some(vec![Some(Literal::int(bucket_of_5))]),
        WriterProperties::default(),
    );
    // A second file whose partition is another bucket, though its rows say
    // otherwise: its `id` bounds admit 5, so only the partition can exclude it.
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[5, 6], &["bob", "cy"], &[5.0, 6.0]),
        Some(vec![Some(Literal::int((bucket_of_5 + 1) % 8))]),
        WriterProperties::default(),
    );
    let (catalog, mut planner) = harness.pivot();

    let equal = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM bucketed.orders WHERE id = 5",
    );
    let range = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM bucketed.orders WHERE id > 4",
    );

    assert_eq!(int64_column(&equal, 0), [Some(1)]);
    // A bucket carries no range, so the range reads both files.
    assert_eq!(int64_column(&range, 0), [Some(3)]);
}

#[test]
fn a_day_partition_keeps_the_file_holding_the_boundary_row() {
    let Some(harness) = harness() else { return };
    let namespace = harness.partitioned_namespace(
        "daily",
        "events",
        events_schema(),
        2,
        "at_day",
        Transform::Day,
    );
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    const MICROS_PER_HOUR: i64 = 3_600_000_000;
    let (jan_1, jan_2) = (18_262, 18_263);
    let day = |days: i32| Some(vec![Some(Literal::Primitive(PrimitiveLiteral::Int(days)))]);
    let midnight = |days: i32| i64::from(days) * MICROS_PER_DAY;
    harness.append_to_partition(
        &namespace,
        "events",
        events(
            &[1, 2],
            &[midnight(jan_1), midnight(jan_1) + MICROS_PER_HOUR],
        ),
        day(jan_1),
        WriterProperties::default(),
    );
    harness.append_to_partition(
        &namespace,
        "events",
        events(
            &[3, 4],
            &[midnight(jan_2), midnight(jan_2) + MICROS_PER_HOUR],
        ),
        day(jan_2),
        WriterProperties::default(),
    );
    let (catalog, mut planner) = harness.pivot();
    let count_where = |planner: &mut Planner, condition: &str| {
        let batches = harness.query(
            &catalog,
            planner,
            &format!("SELECT count(*) FROM daily.events WHERE at {condition}"),
        );
        int64_column(&batches, 0)[0]
    };

    let before_jan_2 = count_where(&mut planner, "< TIMESTAMP '2020-01-02 00:00:00'");
    let up_to_jan_2 = count_where(&mut planner, "<= TIMESTAMP '2020-01-02 00:00:00'");
    let from_jan_2 = count_where(&mut planner, ">= TIMESTAMP '2020-01-02 00:00:00'");
    let after_jan_1_one_am = count_where(&mut planner, "> TIMESTAMP '2020-01-01 01:00:00'");

    assert_eq!(before_jan_2, Some(2));
    assert_eq!(up_to_jan_2, Some(3));
    assert_eq!(from_jan_2, Some(2));
    assert_eq!(after_jan_1_one_am, Some(2));
}
