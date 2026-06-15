//! A MinIO / fake-gcs-server object-store harness for integration tests, gated
//! behind the `test-support` feature. Used by this crate's own object-store
//! tests and by downstream crates (e.g. `server`) that want to run a catalog
//! rooted in a real bucket.
//!
//! Every backend is addressed the same way — a catalog root URI plus an
//! [`ObjectStore`] opened on it — so a test written against `(root, store)` runs
//! unchanged on all three:
//!
//! - [`local`] always works: a tempdir, no Docker.
//! - [`s3`] and [`gcs`] bring up MinIO and `fake-gcs-server` via testcontainers
//!   and return `Some` only when Docker is reachable; otherwise the caller
//!   `eprintln!`s and returns, so tests stay green offline.
//!
//! The two containers (and the process-global environment they configure —
//! `AWS_*` for S3, `STORAGE_EMULATOR_HOST` for GCS) are brought up **once per
//! test binary** inside a single [`OnceLock`] init: all `set_var`s happen on one
//! thread before any store is opened, and the `OnceLock` publishes them with a
//! happens-before to every later reader. Tests isolate themselves under a unique
//! key prefix within the shared bucket rather than per-test containers.

use std::sync::OnceLock;

use crate::store::{DataFileSource, ObjectPath, ObjectStore, open_store};
use testcontainers::core::ContainerPort;
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};
use testcontainers_modules::minio::MinIO;

/// The one bucket both emulated backends share; tests namespace under it.
const BUCKET: &str = "pivot-it";
/// MinIO's default root credentials (the `minio/minio` image, unset env).
const S3_ACCESS_KEY: &str = "minioadmin";
const S3_SECRET_KEY: &str = "minioadmin";
const S3_REGION: &str = "us-east-1";

/// A backend addressable as a catalog root: the root URI and a store opened on
/// it. Drop order is irrelevant — the store holds no container; the containers
/// live for the whole binary in [`containers`].
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
    containers().s3.as_ref()?;
    let root = format!("s3://{BUCKET}/{prefix}");
    let store = open_store(&root).expect("open s3 store");
    Some(Backend { root, store })
}

/// A GCS backend rooted at `gs://<bucket>/<prefix>`, or `None` when
/// `fake-gcs-server` could not be started (no Docker). `prefix` should be unique
/// per test.
pub fn gcs(prefix: &str) -> Option<Backend> {
    containers().gcs.as_ref()?;
    let root = format!("gs://{BUCKET}/{prefix}");
    let store = open_store(&root).expect("open gcs store");
    Some(Backend { root, store })
}

/// Fetch an object through [`ObjectStore::source`] exactly as the engine would:
/// a local path is read off disk; a remote URL (presigned S3 / emulator GCS
/// media URL) is fetched over HTTP. Exercises the same read source the io_uring
/// reader is handed.
pub fn read_via_source(store: &dyn ObjectStore, key: &ObjectPath) -> Vec<u8> {
    match store.source(key).expect("source") {
        DataFileSource::Local(path) => std::fs::read(path).expect("read local source"),
        DataFileSource::Remote(url) => {
            let resp = ureq::get(url.as_str()).call().expect("GET source url");
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf)
                .expect("read source body");
            buf
        }
    }
}

// ---------------------------------------------------------------------------
// Container lifecycle: one MinIO + one fake-gcs-server per test binary.
// ---------------------------------------------------------------------------

struct Containers {
    s3: Option<Container<MinIO>>,
    gcs: Option<Container<GenericImage>>,
}

static CONTAINERS: OnceLock<Containers> = OnceLock::new();

fn containers() -> &'static Containers {
    CONTAINERS.get_or_init(|| Containers {
        s3: start_s3(),
        gcs: start_gcs(),
    })
}

fn set_env(key: &str, value: &str) {
    // SAFE: only ever called from the single `OnceLock` init, on one thread,
    // before any store reads the environment.
    unsafe { std::env::set_var(key, value) };
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

/// Bring up `fake-gcs-server` (plain HTTP, in-memory), point
/// `STORAGE_EMULATOR_HOST` at it, and create the shared bucket.
fn start_gcs() -> Option<Container<GenericImage>> {
    // No log-message wait: `fake-gcs-server`'s banner has shifted across
    // versions, so we treat "container running" as the gate and poll the JSON
    // API for actual readiness via the bucket-create retry below.
    let image = GenericImage::new("fsouza/fake-gcs-server", "1.52.2")
        .with_exposed_port(ContainerPort::Tcp(4443))
        .with_cmd(["-scheme", "http", "-backend", "memory", "-port", "4443"]);
    let container = match image.start() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[test_support] skipping GCS backend — fake-gcs-server unavailable: {e}");
            return None;
        }
    };
    let host = container.get_host().ok()?.to_string();
    let port = container.get_host_port_ipv4(4443).ok()?;
    let endpoint = format!("http://{host}:{port}");
    set_env("STORAGE_EMULATOR_HOST", &endpoint);
    if let Err(e) = gcs_create_bucket(&endpoint) {
        eprintln!("[test_support] skipping GCS backend — bucket create failed: {e}");
        return None;
    }
    Some(container)
}

/// `PUT /<bucket>` against MinIO, SigV4-signed (mirrors the catalog's own S3
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

/// Create the shared bucket on `fake-gcs-server` via its JSON API, retrying
/// while the server is still coming up. A 409 means it already exists — fine.
fn gcs_create_bucket(endpoint: &str) -> Result<(), String> {
    let url = format!("{endpoint}/storage/v1/b?project=pivot-test");
    let body = format!(r#"{{"name":"{BUCKET}"}}"#);
    let mut last = String::new();
    // ~12s of patience: a transport error means "not listening yet"; retry.
    for _ in 0..60 {
        match ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body)
        {
            Ok(_) => return Ok(()),
            Err(ureq::Error::Status(409, _)) => return Ok(()),
            // A non-409 status means the server answered — the request is wrong,
            // not the readiness; don't keep hammering.
            Err(e @ ureq::Error::Status(..)) => return Err(format!("create bucket: {e}")),
            Err(e) => {
                last = e.to_string();
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
    Err(format!("create bucket: server never became ready ({last})"))
}
