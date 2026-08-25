//! Helpers and fixtures shared by the integration test binaries.
//!
//! Lives under `tests/common/mod.rs` so cargo treats it as a module of each
//! test binary that `mod common`s it in, rather than as its own test binary.

#![allow(dead_code)] // each test binary uses a subset

use std::collections::HashMap;
use std::net::{SocketAddr, TcpStream};
use std::ops::Deref;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bin::server::Server;
use catalog::delta::DeltaDatastore;
use catalog::metastore::{DEFAULT_USER_NAME, Metastore, UserAuth};
use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use dispatch::{DataFlowDispatcher, Dispatch};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rstest::fixture;
use tempfile::TempDir;
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

/// A catalog plus any temporary data directories the test server itself owns.
/// The server thread retains the guards until `serve` returns.
pub struct CatalogFixture {
    catalog: Arc<PivotCatalog>,
    data_dirs: Vec<TempDir>,
}

impl CatalogFixture {
    pub fn new(catalog: Arc<PivotCatalog>) -> Self {
        Self {
            catalog,
            data_dirs: Vec::new(),
        }
    }

    pub fn with_data_dir(catalog: Arc<PivotCatalog>, data_dir: TempDir) -> Self {
        Self {
            catalog,
            data_dirs: vec![data_dir],
        }
    }
}

/// Snappy Parquet bytes for a `(name VARCHAR, value BIGINT)` table carrying the
/// given `value`s (the names are filler), ready to `put` into a store as a
/// table's pre-existing data file.
pub fn parquet_name_value_rows(values: &[i64]) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let name: ArrayRef = Arc::new(StringArray::from(vec!["x"; values.len()]));
    let value: ArrayRef = Arc::new(Int64Array::from(values.to_vec()));
    let batch = RecordBatch::try_new(schema.clone(), vec![name, value]).unwrap();
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    bytes
}

pub fn pick_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

pub fn wait_until_listening(addr: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if TcpStream::connect(addr).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!("server failed to start listening on {addr}");
}

/// Tests share a single server (dispatch is process-global). They also touch a
/// shared catalog, so we serialise rather than relying on cargo's parallel
/// runner.
static SERIAL: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

pub fn lock_serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

/// Start a server on a fresh local port on a dedicated background thread (so the
/// server's tokio runtime and `Dispatch` stay separate from the test runtime),
/// with the datastores that `build_catalog` constructs from that thread's own
/// `Dispatch`. `ring_slots` sizes the buffer-pool ring (in 2MB slots). Returns
/// the port once listening. `Dispatch::spin_up` is process-global, so call this
/// at most once per test binary.
pub fn start_server<F>(ring_slots: usize, build_catalog: F) -> u16
where
    F: FnOnce(&Dispatch) -> CatalogFixture + Send + 'static,
{
    start_server_with_metastore(ring_slots, |dispatch| {
        (build_catalog(dispatch), pivot_metastore())
    })
}

/// Minimal metastore for tests whose catalog is built directly: it supplies the
/// same built-in trusted `pivot` user as an empty YAML user map. Its datastore
/// methods are not used because those tests pass an already-open catalog.
pub fn pivot_metastore() -> Arc<dyn Metastore> {
    #[derive(Debug)]
    struct PivotMetastore;

    impl Metastore for PivotMetastore {
        fn open_datastores(
            &self,
            _dispatcher: &DataFlowDispatcher,
        ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
            Ok(HashMap::new())
        }

        fn default_datastore_name(&self) -> &str {
            DEFAULT_DATASTORE_NAME
        }

        fn user_auth(&self, username: &str) -> Option<UserAuth> {
            (username == DEFAULT_USER_NAME).then_some(UserAuth::Trust)
        }
    }

    Arc::new(PivotMetastore)
}

/// As [`start_server`], but the builder returns the same metastore that the
/// server should consult for every login.
pub fn start_server_with_metastore<F>(ring_slots: usize, build_catalog: F) -> u16
where
    F: FnOnce(&Dispatch) -> (CatalogFixture, Arc<dyn Metastore>) + Send + 'static,
{
    start_server_with_tls(ring_slots, None, build_catalog)
}

/// As [`start_server_with_metastore`], but the server offers `tls`'s certificate
/// to connections that ask to encrypt themselves. `None` leaves it refusing
/// them, which is what the other starters do.
pub fn start_server_with_tls<F>(
    ring_slots: usize,
    tls: Option<pgwire::tokio::TlsAcceptor>,
    build_catalog: F,
) -> u16
where
    F: FnOnce(&Dispatch) -> (CatalogFixture, Arc<dyn Metastore>) + Send + 'static,
{
    let port = pick_free_port();
    let workers = core_affinity::get_core_ids().unwrap().len().clamp(1, 4);
    let bind: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    thread::spawn(move || {
        let dispatch = Dispatch::spin_up(workers, ring_slots, None);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            // Build the datastores inside the runtime: a datastore that
            // self-manages maintenance spawns its tasks onto the ambient runtime.
            let (CatalogFixture { catalog, data_dirs }, metastore) = build_catalog(&dispatch);
            let mut server = Server::new(bind, dispatch, catalog, metastore);
            if let Some(acceptor) = tls {
                server = server.with_tls(acceptor);
            }
            let result = server.serve(Box::pin(std::future::pending::<()>())).await;
            drop(data_dirs);
            let _ = result;
        });
    });

    wait_until_listening(bind);
    port
}

/// The shared server's database root, recorded when [`server_port`] starts it so
/// a test can look at what the server wrote.
static DATA_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

/// The shared server's database root. Panics if the server has not started.
pub fn data_dir() -> &'static std::path::Path {
    DATA_DIR.get().expect("the shared server has started")
}

/// Where the shared server keeps `schema.table`'s own storage: its Delta log and
/// every file written into it. A table's directory is named for its identity,
/// which the database manifest maps the table's name to, under the schema that
/// names it: the same table name lives in several schemas here, so the lookup
/// has to say which one it means.
pub fn qualified_table_dir(schema: &str, table: &str) -> std::path::PathBuf {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(data_dir().join("_pivot_manifest.json")).unwrap())
            .unwrap();
    let id = manifest["schemas"]
        .as_array()
        .expect("the manifest lists its schemas")
        .iter()
        .find(|entry| entry["name"].as_str() == Some(schema))
        .unwrap_or_else(|| panic!("no schema named `{schema}` in the manifest"))["table_ids"]
        [table]
        .as_str()
        .unwrap_or_else(|| panic!("no table named `{schema}.{table}` in the manifest"));
    data_dir().join(id)
}

/// [`qualified_table_dir`] for a table in the default schema.
pub fn table_dir(table: &str) -> std::path::PathBuf {
    qualified_table_dir(planner::DEFAULT_SCHEMA_NAME, table)
}

/// Lazily start a shared single-datastore server and return its port. A generous
/// ring absorbs the cached footer/page residue the many tables the suite creates
/// leave behind.
pub fn server_port() -> u16 {
    static PORT: OnceLock<u16> = OnceLock::new();
    *PORT.get_or_init(|| {
        let data_dir = TempDir::new().unwrap();
        DATA_DIR.set(data_dir.path().to_path_buf()).unwrap();
        start_server(256, move |dispatch| {
            let datastore: Arc<dyn Datastore> =
                DeltaDatastore::open(&data_dir.path().to_string_lossy(), dispatch.dispatcher())
                    .unwrap();
            let catalog = Arc::new(
                PivotCatalog::new(
                    HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore)]),
                    DEFAULT_DATASTORE_NAME.to_string(),
                    pivot_metastore(),
                )
                .unwrap(),
            );
            CatalogFixture::with_data_dir(catalog, data_dir)
        })
    })
}

/// Run `sql` and decode every `DataRow` in the response into
/// `Vec<Option<String>>` (text format), dropping non-row messages.
pub async fn select_rows(client: &Client, sql: &str) -> Vec<Vec<Option<String>>> {
    let msgs = client.simple_query(sql).await.unwrap();
    msgs.into_iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| r.get(i).map(|s| s.to_string()))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect()
}

pub async fn connect_client(port: u16) -> Client {
    let (client, conn) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(port)
        .user("pivot")
        .dbname("test")
        .connect(NoTls)
        .await
        .unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
}

/// Connect as `user`/`password`, returning the wire error when the login fails.
#[allow(dead_code)]
pub async fn login(port: u16, user: &str, password: &str) -> Result<Client, tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(port)
        .user(user)
        .password(password)
        .dbname("test")
        .connect(NoTls)
        .await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    Ok(client)
}

/// Connect without configuring a client password.
#[allow(dead_code)]
pub async fn login_without_password(
    port: u16,
    user: &str,
) -> Result<Client, tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(port)
        .user(user)
        .dbname("test")
        .connect(NoTls)
        .await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    Ok(client)
}

/// A test connection: a `tokio_postgres::Client` paired with the serialisation
/// guard. Holding the guard for the lifetime of the `Conn` keeps every test
/// effectively single-threaded against the shared server. Derefs to `Client`
/// so helpers that take `&Client` accept `&conn` directly via deref coercion.
pub struct Conn {
    client: Client,
    _guard: MutexGuard<'static, ()>,
}

impl Deref for Conn {
    type Target = Client;
    fn deref(&self) -> &Client {
        &self.client
    }
}

/// Default fixture: ensure the server is up, connect a fresh client, and
/// then acquire the serial lock. Connect happens before locking so we don't
/// hold the (`!Send`) `MutexGuard` across the connect `.await` — clippy's
/// `await_holding_lock` would (rightly) complain.
#[fixture]
pub async fn conn() -> Conn {
    let client = connect_client(server_port()).await;
    let _guard = lock_serial();
    Conn { client, _guard }
}

/// A minimal blocking pgwire client speaking the simple protocol, for COPY
/// FROM STDIN tests: psql runs COPY through the simple protocol, while
/// `tokio_postgres` insists on the extended protocol this server does not
/// implement. Messages are returned as `(type byte, payload)` pairs.
pub struct RawConn {
    stream: TcpStream,
}

impl RawConn {
    pub fn connect(port: u16) -> Self {
        use std::io::Write;

        let stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        let mut conn = Self { stream };

        let mut body = 196608i32.to_be_bytes().to_vec();
        for (key, value) in [("user", "pivot"), ("database", "test")] {
            body.extend_from_slice(key.as_bytes());
            body.push(0);
            body.extend_from_slice(value.as_bytes());
            body.push(0);
        }
        body.push(0);
        let mut startup = ((body.len() + 4) as i32).to_be_bytes().to_vec();
        startup.extend(body);
        conn.stream.write_all(&startup).unwrap();
        conn.read_until_ready();
        conn
    }

    pub fn read_message(&mut self) -> (u8, Vec<u8>) {
        use std::io::Read;

        let mut head = [0u8; 5];
        self.stream.read_exact(&mut head).unwrap();
        let len = i32::from_be_bytes(head[1..5].try_into().unwrap()) as usize - 4;
        let mut payload = vec![0u8; len];
        self.stream.read_exact(&mut payload).unwrap();
        (head[0], payload)
    }

    /// Read up to and including the next `ReadyForQuery`.
    pub fn read_until_ready(&mut self) -> Vec<(u8, Vec<u8>)> {
        let mut messages = Vec::new();
        loop {
            let message = self.read_message();
            let ready = message.0 == b'Z';
            messages.push(message);
            if ready {
                return messages;
            }
        }
    }

    fn send(&mut self, kind: u8, payload: &[u8]) {
        use std::io::Write;

        let mut message = vec![kind];
        message.extend(((payload.len() + 4) as i32).to_be_bytes());
        message.extend_from_slice(payload);
        self.stream.write_all(&message).unwrap();
    }

    pub fn query(&mut self, sql: &str) {
        let mut payload = sql.as_bytes().to_vec();
        payload.push(0);
        self.send(b'Q', &payload);
    }

    pub fn copy_data(&mut self, data: &[u8]) {
        self.send(b'd', data);
    }

    pub fn copy_done(&mut self) {
        self.send(b'c', &[]);
    }

    pub fn copy_fail(&mut self, message: &str) {
        let mut payload = message.as_bytes().to_vec();
        payload.push(0);
        self.send(b'f', &payload);
    }
}

/// The `CommandComplete` tag in `messages`, e.g. `COPY 2`.
pub fn command_tag(messages: &[(u8, Vec<u8>)]) -> Option<String> {
    messages
        .iter()
        .find(|(kind, _)| *kind == b'C')
        .map(|(_, payload)| {
            String::from_utf8_lossy(payload.strip_suffix(&[0]).unwrap_or(payload)).into_owned()
        })
}
