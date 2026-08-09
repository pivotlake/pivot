//! An object-store harness for integration tests, gated behind the
//! `test-support` feature. Used by this crate's own object-store tests and by
//! downstream crates (e.g. `server`) that want to run a catalog rooted in a
//! real bucket.
//!
//! Every backend is addressed the same way (a catalog root URI plus an
//! [`ObjectStore`] opened on it), so a test written against `(root, store)` runs
//! unchanged on all three:
//!
//! - [`local`] always works: a tempdir, no Docker.
//! - [`s3`] brings up MinIO via testcontainers and returns `Some` only when
//!   Docker is reachable; otherwise the caller `eprintln!`s and returns, so
//!   tests stay green offline.
//! - [`gcs`] always works too: MinIO cannot stand in for GCS (the two differ in
//!   exactly the places a backend has to get right: bearer auth and the
//!   conditional-create header), so an in-process
//!   [emulator](gcs_emulator::GcsEmulator) serves the GCS XML API instead.
//!
//! The backends (and the process-global environment they configure: `AWS_*`,
//! `STORAGE_EMULATOR_HOST`, `GOOGLE_OAUTH_ACCESS_TOKEN`) are brought up **once
//! per test binary** inside a single [`OnceLock`] init: all `set_var`s happen on
//! one thread before any store is opened, and the `OnceLock` publishes them with
//! a happens-before to every later reader. Tests isolate themselves under a
//! unique key prefix within the shared bucket rather than per-test backends.

mod gcs_emulator;

use std::sync::OnceLock;

use crate::store::{DataFileLocation, ObjectPath, ObjectStore, open_store};
use gcs_emulator::GcsEmulator;
use testcontainers::Container;
use testcontainers::runners::SyncRunner;
use testcontainers_modules::minio::MinIO;

/// The one bucket every emulated backend serves; tests namespace under it.
const BUCKET: &str = "pivot-it";
/// MinIO's default root credentials (the `minio/minio` image, unset env).
const S3_ACCESS_KEY: &str = "minioadmin";
const S3_SECRET_KEY: &str = "minioadmin";
const S3_REGION: &str = "us-east-1";
/// The bearer token the GCS emulator accepts, standing in for a minted OAuth2
/// access token.
const GCS_ACCESS_TOKEN: &str = "pivot-test-token";

/// A backend addressable as a catalog root: the root URI and a store opened on
/// it. Drop order is irrelevant: the store holds no container, and the backends
/// live for the whole binary in [`backends`].
pub struct Backend {
    pub root: String,
    pub store: Box<dyn ObjectStore>,
}

/// A fresh local-filesystem backend (always available). The returned `TempDir`
/// owns the root directory — keep it alive for the test's duration.
pub fn local() -> (tempfile::TempDir, Backend) {
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path().to_str().unwrap().to_string();
    let store = open_store(&root).expect("open local store");
    (dir, Backend { root, store })
}

/// An S3 backend rooted at `s3://<bucket>/<prefix>`, or `None` when MinIO could
/// not be started (no Docker). `prefix` should be unique per test.
pub fn s3(prefix: &str) -> Option<Backend> {
    backends().s3.as_ref()?;
    let root = format!("s3://{BUCKET}/{prefix}");
    let store = open_store(&root).expect("open s3 store");
    Some(Backend { root, store })
}

/// A GCS backend rooted at `gs://<bucket>/<prefix>`, served by the in-process
/// emulator. `prefix` should be unique per test.
pub fn gcs(prefix: &str) -> Backend {
    backends();
    let root = format!("gs://{BUCKET}/{prefix}");
    let store = open_store(&root).expect("open gcs store");
    Backend { root, store }
}

/// Fetch an object through [`ObjectStore::source`] exactly as the engine would:
/// a local path is read off disk, and a remote URL (a presigned S3 URL, or a GCS
/// object URL plus the bearer header its `auth` mints) is fetched over HTTP.
/// Exercises the same read source the io_uring reader is handed.
pub fn read_via_source(store: &dyn ObjectStore, key: &ObjectPath) -> Vec<u8> {
    match store.source(key).expect("source") {
        DataFileLocation::Local(path) => std::fs::read(path).expect("read local source"),
        DataFileLocation::Remote { url, auth } => {
            let mut req = ureq::get(url.as_str());
            if let Some(header) = auth.and_then(|f| f()) {
                req = req.set("Authorization", &header);
            }
            let resp = req.call().expect("GET source url");
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf)
                .expect("read source body");
            buf
        }
    }
}

// ---------------------------------------------------------------------------
// Backend lifecycle: one MinIO and one GCS emulator per test binary.
// ---------------------------------------------------------------------------

struct Backends {
    s3: Option<Container<MinIO>>,
    /// Held so the emulator's port stays described for the whole binary; the
    /// stores reach it through the environment [`start_gcs`] set.
    _gcs: GcsEmulator,
}

static BACKENDS: OnceLock<Backends> = OnceLock::new();

fn backends() -> &'static Backends {
    BACKENDS.get_or_init(|| Backends {
        s3: start_s3(),
        _gcs: start_gcs(),
    })
}

/// Bring up the in-process GCS emulator and point the `gs://` environment at
/// it: the endpoint override and the bearer token it accepts.
fn start_gcs() -> GcsEmulator {
    let emulator = GcsEmulator::start(BUCKET, GCS_ACCESS_TOKEN);
    // A key file in the developer's own environment outranks a token, and would
    // send these tests to real GCS with real credentials.
    unset_env("GOOGLE_APPLICATION_CREDENTIALS");
    set_env("STORAGE_EMULATOR_HOST", emulator.endpoint());
    set_env("GOOGLE_OAUTH_ACCESS_TOKEN", emulator.token());
    emulator
}

fn set_env(key: &str, value: &str) {
    // SAFE: only ever called from the single `OnceLock` init, on one thread,
    // before any store reads the environment.
    unsafe { std::env::set_var(key, value) };
}

fn unset_env(key: &str) {
    // SAFE: as `set_env`, from the single `OnceLock` init, before any reader.
    unsafe { std::env::remove_var(key) };
}

/// Bring up MinIO, point `AWS_*` at it, and create the shared bucket. Returns
/// `None` (with a note) on any failure so the tests skip rather than fail.
fn start_s3() -> Option<Container<MinIO>> {
    let container = match MinIO::default().start() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[test_support] skipping S3 backend — MinIO unavailable: {e}");
            return None;
        }
    };
    let host = container.get_host().ok()?.to_string();
    let port = container.get_host_port_ipv4(9000).ok()?;
    let endpoint = format!("http://{host}:{port}");
    set_env("AWS_ENDPOINT_URL", &endpoint);
    set_env("AWS_ACCESS_KEY_ID", S3_ACCESS_KEY);
    set_env("AWS_SECRET_ACCESS_KEY", S3_SECRET_KEY);
    set_env("AWS_REGION", S3_REGION);
    if let Err(e) = s3_create_bucket(&endpoint) {
        eprintln!("[test_support] skipping S3 backend — bucket create failed: {e}");
        return None;
    }
    Some(container)
}

/// `PUT /<bucket>` against MinIO, SigV4-signed (mirrors the datastore's own S3
/// signing). A 409 means the bucket already exists — fine.
fn s3_create_bucket(endpoint: &str) -> Result<(), String> {
    use aws_credential_types::Credentials;
    use aws_sigv4::http_request::{
        PayloadChecksumKind, SignableBody, SignableRequest, SigningSettings, sign,
    };
    use aws_sigv4::sign::v4;

    let url = format!("{endpoint}/{BUCKET}");
    let host = endpoint.split("://").nth(1).unwrap_or(endpoint);

    let creds = Credentials::new(S3_ACCESS_KEY, S3_SECRET_KEY, None, None, "harness");
    let identity = creds.into();
    let mut settings = SigningSettings::default();
    settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(S3_REGION)
        .name("s3")
        .time(std::time::SystemTime::now())
        .settings(settings)
        .build()
        .map_err(|e| format!("sigv4 params: {e}"))?;

    let headers = [("host", host)];
    let signable = SignableRequest::new(
        "PUT",
        &url,
        headers.iter().copied(),
        SignableBody::Bytes(b""),
    )
    .map_err(|e| format!("sigv4 signable: {e}"))?;
    let (instructions, _) = sign(signable, &params.into())
        .map_err(|e| format!("sigv4 sign: {e}"))?
        .into_parts();

    let mut req = ureq::put(&url);
    for (name, value) in instructions.headers() {
        req = req.set(name, value);
    }
    match req.send_bytes(b"") {
        Ok(_) => Ok(()),
        // Bucket already exists / already owned by us.
        Err(ureq::Error::Status(409, _)) => Ok(()),
        Err(e) => Err(format!("PUT bucket: {e}")),
    }
}

use crate::CatalogTable;
use crate::manifest::{DeltaFileEntry, SortBounds};

impl CatalogTable {
    /// Test-only: write `bytes` as a new data file at `path` (under the table's
    /// location) and commit it into the table in one manifest version — the way
    /// tests seed a table with a specific pre-encoded file. Idempotent on `path`.
    ///
    /// Production writers never take this path: INSERT and compaction upload
    /// their files over the shared ring and commit the built row groups. This
    /// lives here so it stays out of production builds.
    pub fn append_data_file(
        &mut self,
        path: ObjectPath,
        bytes: &[u8],
        partition: Option<crate::PartitionValues>,
        sort_bounds: Option<SortBounds>,
    ) -> crate::Result<()> {
        if self.file_refs().iter().any(|file| file.path == path) {
            return Ok(());
        }
        let file = self.write_data_file(path, bytes)?;
        let entry = DeltaFileEntry {
            file,
            partition,
            sort_bounds,
            stats: None,
        };
        // A plain add (data_change = true); the new file's footer is read on a
        // winning commit via `sync_files_to_manifest`, so no dispatcher is needed.
        self.commit_entries(&[], &[entry], true)?;
        Ok(())
    }
}

impl CatalogTable {
    /// Test-only: atomically swap `removed` files for `added` (bare manifest
    /// entries) in one manifest version, reading the added files' footers only if
    /// the swap wins -- so a losing swap errors before touching its output.
    /// Production compaction commits its merged files through
    /// `compact_files`, which already holds the row groups.
    pub fn replace_data_files(
        &mut self,
        removed: &[ObjectPath],
        added: &[DeltaFileEntry],
    ) -> crate::Result<()> {
        self.commit_entries(removed, added, false)
    }
}
