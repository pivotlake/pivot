//! Helpers and fixtures shared by the integration test binaries.
//!
//! Lives under `tests/common/mod.rs` so cargo treats it as a module of each
//! test binary that `mod common`s it in, rather than as its own test binary.

#![allow(dead_code)] // each test binary uses a subset

use std::net::{SocketAddr, TcpStream};
use std::ops::Deref;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use catalog::PivotCatalog;
use datastore_delta::ParquetCatalog;
use dispatch::Dispatch;
use rstest::fixture;
use server::Server;
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

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
/// with the datastores that `build_catalogs` constructs from that thread's own
/// `Dispatch`. `ring_slots` sizes the buffer-pool ring (in 2MB slots). Returns
/// the port once listening. `Dispatch::spin_up` is process-global, so call this
/// at most once per test binary.
pub fn start_server<F>(ring_slots: usize, build_catalogs: F) -> u16
where
    F: FnOnce(&Dispatch) -> Arc<PivotCatalog> + Send + 'static,
{
    let port = pick_free_port();
    let workers = core_affinity::get_core_ids().unwrap().len().clamp(1, 4);
    let bind: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    thread::spawn(move || {
        let dispatch = Dispatch::spin_up(workers, ring_slots, None);
        let catalogs = build_catalogs(&dispatch);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let server = Server::new(
                bind,
                dispatch,
                catalogs,
                0,
                4,
                server::DEFAULT_CATALOG_REFRESH,
            );
            let _ = server.serve(Box::pin(std::future::pending::<()>())).await;
        });
    });

    wait_until_listening(bind);
    port
}

/// Lazily start a shared single-datastore server and return its port. A generous
/// ring absorbs the cached footer/page residue the many tables the suite creates
/// leave behind.
pub fn server_port() -> u16 {
    static PORT: OnceLock<u16> = OnceLock::new();
    *PORT.get_or_init(|| {
        start_server(256, |dispatch| {
            Arc::new(PivotCatalog::single(Arc::new(ParquetCatalog::new(
                dispatch.dispatcher().clone(),
            ))))
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
        .user("test")
        .dbname("test")
        .connect(NoTls)
        .await
        .unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
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
