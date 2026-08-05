//! Boot the pivotdb server in-process for benchmarking.
//!
//! Picks a free local port, initialises a `dispatch` worker pool, and runs
//! `server::Server` on a dedicated background thread driving its own tokio
//! runtime — so the bench's blocking iteration loop on the main thread never
//! starves the accept loop. Returns a [`ServerHandle`] holding the bind port
//! plus a `oneshot::Sender` that triggers a clean shutdown when the handle is
//! dropped.

use std::collections::HashMap;
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use datastore_delta::{
    Compacter, DEFAULT_COMPACT_BYTES, DEFAULT_MIN_FILES_TO_MERGE, DeltaDatastore,
};
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use metastore::{DEFAULT_USER_NAME, Metastore, UserAuth};
use server::Server;
use tokio::sync::oneshot;

fn total_memory_bytes() -> usize {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
    .total_memory() as usize
}

/// Share of the machine's memory the ring takes, as a percentage — the same
/// `PIVOT_MEMORY_PCT` the real server reads, defaulting to the same 80, so a
/// benchmark run is sized like the server it stands in for.
///
/// Worth overriding on a machine that isn't the benchmark's alone: the ring is
/// prefaulted, so 80% of total is 80% whether or not something else is holding
/// memory, and the run is simply OOM-killed.
fn ring_buffers() -> usize {
    let pct: usize = dispatch::env::get_env_var_with_default("PIVOT_MEMORY_PCT", 80);
    total_memory_bytes() * pct / 100 / BUFFER_SIZE
}

pub struct ServerHandle {
    port: u16,
    datastore: Arc<DeltaDatastore>,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl ServerHandle {
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Compact every table synchronously. A load of many small `INSERT`s leaves
    /// as many small files; this merges them into target-sized ones before the
    /// queries read them, the way the datastore's own compacter would over time.
    ///
    /// One sweep can leave a tail: a merge changes the file list, and a later
    /// candidate in the same sweep may be missed. So sweep until one merges
    /// nothing, which a sweep that advanced no table's log version did.
    pub async fn compact(&self) {
        let compacter = Compacter::new(
            DEFAULT_COMPACT_BYTES,
            DEFAULT_MIN_FILES_TO_MERGE,
            // The poll interval is unused for a manual sweep.
            Duration::from_secs(1),
            self.datastore.clone(),
        );
        loop {
            let versions_before = self.table_versions();
            compacter.compact_all().await;
            if self.table_versions() == versions_before {
                break;
            }
        }
    }

    /// Every table's committed log version, keyed by name, so two of these taken
    /// around a sweep say whether it merged anything.
    fn table_versions(&self) -> HashMap<String, u64> {
        self.datastore
            .tables()
            .iter()
            .map(|(name, table)| (name.to_string(), table.version()))
            .collect()
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Pick a free local port by binding to `:0` and dropping the listener; the
/// kernel will not immediately reuse the port for the brief window before the
/// server thread re-binds.
fn pick_free_port() -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

fn wait_until_listening(addr: SocketAddr) -> std::io::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut last_err = None;
    while Instant::now() < deadline {
        match TcpStream::connect(addr) {
            Ok(_) => return Ok(()),
            Err(e) => last_err = Some(e),
        }
        thread::sleep(Duration::from_millis(20));
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::other("timed out waiting for server")))
}

/// User source for the in-process benchmark server. The catalog is assembled
/// directly above, so only the built-in trusted `pivot` login is used here.
fn pivot_metastore() -> Arc<dyn Metastore> {
    struct PivotMetastore;

    impl Metastore for PivotMetastore {
        fn open_datastores(
            &self,
            _dispatcher: &DataFlowDispatcher,
        ) -> metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
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

/// Start a pivotdb server with the given worker count, returning once the
/// listener is accepting connections. The catalog is empty; the runner sends
/// `CREATE TABLE` over the wire to populate it.
pub fn start(workers: usize) -> std::io::Result<ServerHandle> {
    let port = pick_free_port()?;
    let bind: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .expect("valid socket addr");

    let dispatch = Dispatch::spin_up(workers, ring_buffers(), None);
    let data_dir = tempfile::tempdir()?;
    let datastore = DeltaDatastore::open_local(data_dir.path(), dispatch.dispatcher())
        .map_err(std::io::Error::other)?;
    let catalog = Arc::new(
        PivotCatalog::new(
            HashMap::from([(
                DEFAULT_DATASTORE_NAME.to_string(),
                datastore.clone() as Arc<dyn Datastore>,
            )]),
            DEFAULT_DATASTORE_NAME.to_string(),
        )
        .map_err(std::io::Error::other)?,
    );

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let thread = thread::Builder::new()
        .name("pivot-bench-server".into())
        .spawn(move || {
            // The benchmark owns its scratch datastore explicitly for exactly
            // as long as the in-process server thread is alive.
            let _data_dir = data_dir;
            // The runtime gets few threads on purpose: its default (one per
            // core) would sit hundreds of mostly-idle threads next to the
            // pinned dispatch workers and preempt them on every wakeup.
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("build tokio runtime");
            rt.block_on(async move {
                let server = Server::new(bind, dispatch, catalog, pivot_metastore());
                let _ = server
                    .serve(Box::pin(async move {
                        let _ = shutdown_rx.await;
                    }))
                    .await;
            });
        })?;

    wait_until_listening(bind)?;

    Ok(ServerHandle {
        port,
        datastore,
        shutdown: Some(shutdown_tx),
        thread: Some(thread),
    })
}
