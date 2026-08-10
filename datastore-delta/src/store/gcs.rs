//! Google Cloud Storage backend: blocking HTTP via [`ureq`] over the GCS JSON
//! API, authenticated with an OAuth2 bearer token — for both the control-plane
//! metadata ops here *and* the io_uring range reads, which carry a fresh
//! `Authorization: Bearer` header rather than a presigned URL (a presigned URL
//! expires after an hour, stranding files cached across queries).
//!
//! The token is minted (and cached until shortly before expiry) following the
//! standard Application Default Credentials chain, in order: the file
//! [`GcsStore::with_credentials_file`] was opened with, else the file at
//! `GOOGLE_APPLICATION_CREDENTIALS`, else the well-known file that
//! `gcloud auth application-default login` writes (so a developer with gcloud
//! needs no extra config), else the **GCE metadata server** (workload identity
//! on Google compute). A credentials file is either a **service-account** key
//! (a self-signed RS256 JWT exchanged for an access token) or an
//! **authorized_user** file (refresh-token grant). Minting is a blocking `ureq`
//! call kept off the ring: it happens on the control thread (every query primes
//! the token via [`source`](GcsStore::source)), so a worker only ever *reads* a
//! current token. RS256 signing uses `ring`; everything is synchronous, no async
//! runtime.
//!
//! Reads and writes address the same object through different APIs, because the
//! ring speaks one method per direction. A read is a `GET` of the JSON API's
//! `?alt=media` URL; an upload is a `PUT`, which the JSON API's upload endpoint
//! does not accept (it takes a `POST`), so a write addresses the object through
//! the XML API instead — `PUT <endpoint>/<bucket>/<object>`, the same bytes at
//! the same object name.

use super::{
    DataFileLocation, FileRef, ListedObject, ObjectPath, ObjectStore, Result, StoreError,
    absolute_object_key, object_key, parse_iso8601_millis, percent_encode,
};
use base64::Engine;
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Re-mint a token once it's within this many seconds of expiry. Generous so a
/// token primed at query start stays valid for the whole query's worker reads.
const TOKEN_REFRESH_MARGIN: u64 = 300;

const TOKEN_SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_write";
const OAUTH_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
const METADATA_TOKEN_URI: &str =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
/// The real GCS API origin. Tests point at an emulator instead via
/// `STORAGE_EMULATOR_HOST` (the convention `fake-gcs-server` and the Google
/// client libraries share).
const DEFAULT_ENDPOINT: &str = "https://storage.googleapis.com";

#[derive(Debug)]
pub struct GcsStore {
    /// The `gs://bucket/prefix` URI this store was opened with, kept verbatim
    /// as its addressable root (see [`ObjectStore::location_uri`]).
    uri: String,
    bucket: String,
    /// In-bucket prefix under which this datastore's keys live.
    prefix: String,
    /// API origin (scheme + `host[:port]`, no trailing slash). The real service
    /// by default; an emulator when `STORAGE_EMULATOR_HOST` is set.
    endpoint: String,
    /// Shared token state. An `Arc` so [`source`](Self::source) can hand workers
    /// a closure that reads the same cell this store re-mints.
    auth: Arc<GcsAuth>,
    /// The shared async client, built on first use (see
    /// [`ObjectStore::build_delta_object_store`]).
    client: std::sync::OnceLock<Arc<DynObjectStore>>,
}

/// Shared OAuth2 token state. The token outlives any one query and is identical
/// for every file and worker, so a single cell is shared store-wide. Reads (the
/// worker hot path) take the `RwLock` for read and never block each other; only
/// a rare re-mint takes it for write — and the blocking mint itself runs
/// *outside* the lock, so workers never stall behind a network round-trip.
#[derive(Debug)]
struct GcsAuth {
    agent: ureq::Agent,
    token: RwLock<Option<CachedToken>>,
    /// An explicitly configured credentials file, which takes precedence over
    /// the rest of the Application Default Credentials chain.
    credentials_file: Option<String>,
    /// Whether the endpoint is an emulator: it ignores credentials, so we never
    /// mint a token and reads need no `Authorization` header.
    emulated: bool,
}

#[derive(Debug, Clone)]
struct CachedToken {
    /// The full header value (`Bearer <token>`), ready to send as-is.
    header: Arc<str>,
    /// Unix seconds after which the token must be re-minted.
    expires_at: u64,
}

impl GcsStore {
    /// Parse `gs://bucket/prefix` and resolve credentials from the Application
    /// Default Credentials chain. Honors `STORAGE_EMULATOR_HOST` (e.g.
    /// `http://localhost:4443`) to target a `fake-gcs-server` emulator instead
    /// of the real service — the scheme is optional and defaults to `http`.
    pub fn from_uri(uri: &str) -> Result<Self> {
        Self::build(uri, None)
    }

    /// Parse `gs://bucket/prefix` but mint tokens from the service-account (or
    /// authorized-user) JSON at `credentials_file` instead of consulting the
    /// environment, the path a metastore uses to open a datastore with its own
    /// configured credentials.
    pub fn with_credentials_file(uri: &str, credentials_file: &str) -> Result<Self> {
        Self::build(uri, Some(credentials_file.to_string()))
    }

    fn build(uri: &str, credentials_file: Option<String>) -> Result<Self> {
        let rest = uri
            .strip_prefix("gs://")
            .ok_or_else(|| StoreError::UnsupportedUri(uri.to_string()))?;
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        let (endpoint, emulated) = match std::env::var("STORAGE_EMULATOR_HOST") {
            Ok(host) if !host.is_empty() => {
                let host = host.trim_end_matches('/');
                let endpoint = if host.contains("://") {
                    host.to_string()
                } else {
                    format!("http://{host}")
                };
                (endpoint, true)
            }
            _ => (DEFAULT_ENDPOINT.to_string(), false),
        };
        Ok(Self {
            uri: uri.to_string(),
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            endpoint,
            auth: Arc::new(GcsAuth {
                agent: ureq::AgentBuilder::new().build(),
                token: RwLock::new(None),
                credentials_file,
                emulated,
            }),
            client: std::sync::OnceLock::new(),
        })
    }

    /// JSON API object name for a store key, escaped as one path segment (`/`
    /// included, since the whole key is a single `{object}` parameter).
    fn object_segment(&self, key: &ObjectPath) -> String {
        percent_encode(&object_key(&self.prefix, key))
    }

    /// The JSON API media URL a read GETs: stable (no embedded signature to
    /// expire), with auth riding in a per-request `Authorization` header.
    fn media_url(&self, key: &ObjectPath) -> String {
        format!(
            "{}/storage/v1/b/{}/o/{}?alt=media",
            self.endpoint,
            self.bucket,
            self.object_segment(key)
        )
    }

    /// The XML API object URL an upload PUTs. Each key component is escaped
    /// separately so the key's `/`s stay path separators.
    fn upload_url(&self, key: &ObjectPath) -> String {
        let object = object_key(&self.prefix, key)
            .split('/')
            .map(percent_encode)
            .collect::<Vec<_>>()
            .join("/");
        format!("{}/{}/{}", self.endpoint, self.bucket, object)
    }
}

impl GcsAuth {
    /// A current `Authorization` header, minting and caching a fresh token when
    /// the cached one is missing or near expiry. Blocking (`ureq`); call only on
    /// the control thread — workers use [`current`](Self::current) instead.
    fn header(&self) -> Result<Arc<str>> {
        // An emulator ignores `Authorization`; never mint (no credentials exist).
        if self.emulated {
            return Ok(Arc::from("Bearer emulator"));
        }
        let now = unix_now();
        if let Some(token) = self.token.read().unwrap().as_ref()
            && token.expires_at > now + TOKEN_REFRESH_MARGIN
        {
            return Ok(token.header.clone());
        }
        // Mint without holding the lock — a concurrent mint is harmless (last
        // writer wins) and this way a worker's `current` read never waits on the
        // network round-trip.
        let (value, expires_in) = self.mint_token()?;
        let header: Arc<str> = Arc::from(format!("Bearer {value}"));
        *self.token.write().unwrap() = Some(CachedToken {
            header: header.clone(),
            expires_at: now + expires_in,
        });
        Ok(header)
    }

    /// The currently cached header, or `None` if none is cached yet. Lock-free
    /// read-only fast path for workers — never mints, never blocks.
    fn current(&self) -> Option<Arc<str>> {
        self.token
            .read()
            .unwrap()
            .as_ref()
            .map(|token| token.header.clone())
    }

    /// Mint a fresh access token, returning `(token, lifetime_seconds)`, walking
    /// the Application Default Credentials chain: the configured credentials
    /// file, else an explicit `GOOGLE_APPLICATION_CREDENTIALS` file, else the
    /// gcloud-written well-known file, else the metadata server.
    fn mint_token(&self) -> Result<(String, u64)> {
        if let Some(path) = &self.credentials_file {
            return self.token_from_credentials_file(path);
        }
        if let Ok(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            return self.token_from_credentials_file(&path);
        }
        if let Some(path) = well_known_credentials_path()
            && path.is_file()
        {
            return self.token_from_credentials_file(&path.to_string_lossy());
        }
        self.metadata_token()
    }

    /// Mint a token from a credentials file (a service-account key or an
    /// authorized-user file), dispatching on its `type`.
    fn token_from_credentials_file(&self, path: &str) -> Result<(String, u64)> {
        let bytes = std::fs::read(path).map_err(|source| StoreError::Io {
            key: path.to_string(),
            source,
        })?;
        let credentials: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| StoreError::Config(format!("parsing {path}: {e}")))?;
        match credentials.get("type").and_then(|t| t.as_str()) {
            Some("service_account") => self.service_account_token(&credentials),
            Some("authorized_user") => self.authorized_user_token(&credentials),
            other => Err(StoreError::Config(format!(
                "unsupported credential type {other:?} in {path}"
            ))),
        }
    }

    /// Service-account flow: build and RS256-sign a JWT asserting our identity,
    /// then exchange it for an access token.
    fn service_account_token(&self, credentials: &serde_json::Value) -> Result<(String, u64)> {
        let client_email = field(credentials, "client_email")?;
        let private_key_pem = field(credentials, "private_key")?;
        let token_uri = credentials
            .get("token_uri")
            .and_then(|v| v.as_str())
            .unwrap_or(OAUTH_TOKEN_URI);

        let now = unix_now();
        let header = base64url(br#"{"alg":"RS256","typ":"JWT"}"#);
        let claims = format!(
            r#"{{"iss":"{client_email}","scope":"{TOKEN_SCOPE}","aud":"{token_uri}","iat":{now},"exp":{}}}"#,
            now + 3600
        );
        let claims = base64url(claims.as_bytes());
        let signing_input = format!("{header}.{claims}");
        let signature = rs256_sign(&private_key_pem, signing_input.as_bytes())?;
        let jwt = format!("{signing_input}.{}", base64url(&signature));

        let response = self
            .agent
            .post(token_uri)
            .send_form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", &jwt),
            ])
            .map_err(|e| StoreError::Http(format!("gcs token exchange: {e}")))?;
        parse_token_response(response)
    }

    /// Authorized-user flow (`gcloud auth application-default login`): exchange a
    /// refresh token for an access token.
    fn authorized_user_token(&self, credentials: &serde_json::Value) -> Result<(String, u64)> {
        let client_id = field(credentials, "client_id")?;
        let client_secret = field(credentials, "client_secret")?;
        let refresh_token = field(credentials, "refresh_token")?;
        let response = self
            .agent
            .post(OAUTH_TOKEN_URI)
            .send_form(&[
                ("client_id", &client_id),
                ("client_secret", &client_secret),
                ("refresh_token", &refresh_token),
                ("grant_type", "refresh_token"),
            ])
            .map_err(|e| StoreError::Http(format!("gcs token refresh: {e}")))?;
        parse_token_response(response)
    }

    /// GCE/GKE/Cloud Run metadata server (workload identity).
    fn metadata_token(&self) -> Result<(String, u64)> {
        let response = self
            .agent
            .get(METADATA_TOKEN_URI)
            .set("Metadata-Flavor", "Google")
            .call()
            .map_err(|e| {
                StoreError::Config(format!(
                    "no configured credentials file, no GOOGLE_APPLICATION_CREDENTIALS, no \
                     gcloud application-default credentials, and metadata server unavailable: {e}"
                ))
            })?;
        parse_token_response(response)
    }
}

impl ObjectStore for GcsStore {
    fn describe(&self) -> String {
        format!("gs://{}/{}", self.bucket, self.prefix)
    }

    fn location_uri(&self) -> String {
        self.uri.clone()
    }

    /// A no-op: GCS has no directories. A key is a flat string, and an object at
    /// `prefix/name` exists the moment it is written, with no parent to create.
    fn create_dir(&self, _prefix: &ObjectPath) -> Result<()> {
        Ok(())
    }

    fn build_delta_object_store(&self) -> Result<Arc<DynObjectStore>> {
        if let Some(client) = self.client.get() {
            return Ok(client.clone());
        }
        let mut builder = GoogleCloudStorageBuilder::from_env().with_url(self.uri.as_str());
        if let Some(path) = &self.auth.credentials_file {
            builder = builder.with_service_account_path(path);
        }
        // The emulator ignores credentials and serves plain HTTP on its own
        // origin, so point Kernel's client at it and skip signing entirely.
        if self.auth.emulated {
            builder = builder
                .with_config(GoogleConfigKey::BaseUrl, &self.endpoint)
                .with_config(GoogleConfigKey::SkipSignature, "true");
        }
        let store = builder
            .build()
            .map_err(|source| StoreError::DeltaObjectStore {
                uri: self.uri.clone(),
                source,
            })?;
        Ok(self
            .client
            .get_or_init(|| Arc::new(store) as Arc<DynObjectStore>)
            .clone())
    }

    fn client_key(&self, key: &ObjectPath) -> Result<super::ClientKey> {
        // The async client is rooted at the bucket, so a key addresses it by
        // its full in-bucket name, prefix included.
        super::ClientKey::parse(object_key(&self.prefix, key)).map_err(|e| StoreError::ClientKey {
            key: key.to_string(),
            reason: e.to_string(),
        })
    }

    fn get(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>> {
        let header = self.auth.header()?;
        match self
            .auth
            .agent
            .get(&self.media_url(key))
            .set("Authorization", &header)
            .call()
        {
            Ok(response) => {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut response.into_reader(), &mut buf).map_err(
                    |source| StoreError::Io {
                        key: key.to_string(),
                        source,
                    },
                )?;
                Ok(Some(buf))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(StoreError::Http(format!("GCS GET {key}: {e}"))),
        }
    }

    fn put(&self, key: &ObjectPath, data: &[u8]) -> Result<()> {
        let header = self.auth.header()?;
        let url = format!(
            "{}/upload/storage/v1/b/{}/o?uploadType=media&name={}",
            self.endpoint,
            self.bucket,
            self.object_segment(key)
        );
        match self
            .auth
            .agent
            .post(&url)
            .set("Authorization", &header)
            .set("Content-Type", "application/octet-stream")
            .send_bytes(data)
        {
            Ok(_) => Ok(()),
            Err(e) => Err(StoreError::Http(format!("GCS PUT {key}: {e}"))),
        }
    }

    fn put_if_absent(&self, key: &ObjectPath, data: &[u8]) -> Result<bool> {
        let header = self.auth.header()?;
        // GCS conditional create: `ifGenerationMatch=0` only succeeds when no
        // live generation of the object exists; otherwise 412.
        let url = format!(
            "{}/upload/storage/v1/b/{}/o?uploadType=media&name={}&ifGenerationMatch=0",
            self.endpoint,
            self.bucket,
            self.object_segment(key)
        );
        match self
            .auth
            .agent
            .post(&url)
            .set("Authorization", &header)
            .set("Content-Type", "application/octet-stream")
            .send_bytes(data)
        {
            Ok(_) => Ok(true),
            Err(ureq::Error::Status(412, _)) => Ok(false),
            Err(e) => Err(StoreError::Http(format!("GCS conditional PUT {key}: {e}"))),
        }
    }

    fn delete(&self, key: &ObjectPath) -> Result<()> {
        let header = self.auth.header()?;
        let url = format!(
            "{}/storage/v1/b/{}/o/{}",
            self.endpoint,
            self.bucket,
            self.object_segment(key)
        );
        match self
            .auth
            .agent
            .delete(&url)
            .set("Authorization", &header)
            .call()
        {
            // DELETE is idempotent; a missing key is the goal state.
            Ok(_) | Err(ureq::Error::Status(404, _)) => Ok(()),
            Err(e) => Err(StoreError::Http(format!("GCS DELETE {key}: {e}"))),
        }
    }

    /// One page of objects directly under `prefix`, one level deep.
    fn list(&self, prefix: &ObjectPath) -> Result<Vec<ListedObject>> {
        let header = self.auth.header()?;
        let object_prefix = object_key(&self.prefix, prefix);
        let url = format!(
            "{}/storage/v1/b/{}/o?prefix={}%2F&delimiter=%2F",
            self.endpoint,
            self.bucket,
            percent_encode(&object_prefix)
        );
        let response = self
            .auth
            .agent
            .get(&url)
            .set("Authorization", &header)
            .call()
            .map_err(|e| StoreError::Http(format!("GCS LIST {object_prefix}: {e}")))?;
        let body: ListResponse = response
            .into_json()
            .map_err(|e| StoreError::Http(format!("GCS LIST parse: {e}")))?;

        body.items
            .into_iter()
            .map(|item| {
                let size = item.size.parse().map_err(|_| {
                    StoreError::Http(format!(
                        "GCS LIST: bad size `{}` for {}",
                        item.size, item.name
                    ))
                })?;
                let modified_unix_ms = parse_iso8601_millis(&item.updated).ok_or_else(|| {
                    StoreError::Http(format!(
                        "GCS LIST object `{}` has an unparseable updated `{}`",
                        item.name, item.updated
                    ))
                })?;
                Ok(ListedObject {
                    file: FileRef {
                        path: ObjectPath::new(super::key_name(&item.name)),
                        size,
                    },
                    modified_unix_ms,
                })
            })
            .collect()
    }

    fn absolute_key(&self, key: &ObjectPath) -> Result<ObjectPath> {
        Ok(absolute_object_key(&self.prefix, key))
    }

    fn source(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        self.remote_location(self.media_url(key))
    }

    fn sink(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        self.remote_location(self.upload_url(key))
    }

    fn prepare_write(&self) -> Result<()> {
        self.auth.header()?;
        Ok(())
    }
}

impl GcsStore {
    /// Pair a stable object URL with the token closure workers read per request.
    /// Primes the token here on the control thread (a blocking mint is fine off
    /// the ring) so the query's worker reads find a fresh one and never mint.
    fn remote_location(&self, url: String) -> Result<DataFileLocation> {
        let url = url::Url::parse(&url)
            .map_err(|e| StoreError::Config(format!("building gcs url: {e}")))?;
        // An emulator ignores `Authorization`, so its requests need no header.
        if self.auth.emulated {
            return Ok(DataFileLocation::Remote { url, auth: None });
        }
        self.auth.header()?;
        let auth = self.auth.clone();
        Ok(DataFileLocation::Remote {
            url,
            auth: Some(Arc::new(move || auth.current())),
        })
    }
}

#[derive(serde::Deserialize)]
struct ListResponse {
    #[serde(default)]
    items: Vec<ObjectItem>,
}

#[derive(serde::Deserialize)]
struct ObjectItem {
    name: String,
    /// GCS reports object size as a decimal string in the JSON API.
    #[serde(default)]
    size: String,
    /// RFC 3339 last-modification time, parsed by [`parse_iso8601_millis`] for
    /// vacuum's orphan sweep.
    #[serde(default)]
    updated: String,
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default = "default_expiry")]
    expires_in: u64,
}

/// The path `gcloud auth application-default login` writes its credentials to:
/// `$CLOUDSDK_CONFIG/application_default_credentials.json` when that env var is
/// set, else the per-user gcloud config dir (`%APPDATA%\gcloud` on Windows,
/// `~/.config/gcloud` elsewhere, including macOS). `None` if the home/config dir
/// can't be resolved.
fn well_known_credentials_path() -> Option<PathBuf> {
    const FILE: &str = "application_default_credentials.json";
    if let Some(dir) = std::env::var_os("CLOUDSDK_CONFIG") {
        return Some(PathBuf::from(dir).join(FILE));
    }
    #[cfg(windows)]
    {
        std::env::var_os("APPDATA").map(|dir| PathBuf::from(dir).join("gcloud").join(FILE))
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(|home| {
            PathBuf::from(home)
                .join(".config")
                .join("gcloud")
                .join(FILE)
        })
    }
}

fn default_expiry() -> u64 {
    3600
}

fn parse_token_response(response: ureq::Response) -> Result<(String, u64)> {
    let token: TokenResponse = response
        .into_json()
        .map_err(|e| StoreError::Config(format!("parsing token response: {e}")))?;
    Ok((token.access_token, token.expires_in))
}

fn field(credentials: &serde_json::Value, name: &str) -> Result<String> {
    credentials
        .get(name)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| StoreError::Config(format!("credential file missing `{name}`")))
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

fn base64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// RS256-sign `message` with a PEM-encoded PKCS#8 RSA private key (the
/// `private_key` field of a GCS service-account JSON).
fn rs256_sign(private_key_pem: &str, message: &[u8]) -> Result<Vec<u8>> {
    let der = pem_to_der(private_key_pem)?;
    let key_pair = ring::signature::RsaKeyPair::from_pkcs8(&der)
        .map_err(|e| StoreError::Config(format!("invalid service-account private key: {e}")))?;
    let mut signature = vec![0u8; key_pair.public().modulus_len()];
    let rng = ring::rand::SystemRandom::new();
    key_pair
        .sign(
            &ring::signature::RSA_PKCS1_SHA256,
            &rng,
            message,
            &mut signature,
        )
        .map_err(|e| StoreError::Config(format!("signing service-account JWT: {e}")))?;
    Ok(signature)
}

/// Strip the PEM armor and base64-decode to DER.
fn pem_to_der(pem: &str) -> Result<Vec<u8>> {
    let body: String = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");
    base64::engine::general_purpose::STANDARD
        .decode(body.trim())
        .map_err(|e| StoreError::Config(format!("decoding PEM private key: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_response_parses_names_string_sizes_and_timestamps() {
        // GCS reports object size as a decimal string; the reader parses it to
        // the byte size it needs to locate a Parquet footer.
        let json = r#"{"items":[
            {"name":"db/events/a.parquet","size":"123","updated":"2026-01-02T03:04:05.678Z"},
            {"name":"db/events/b.parquet","size":"4096","updated":"2026-01-02T03:04:06Z"}
        ]}"#;

        let parsed: ListResponse = serde_json::from_str(json).unwrap();

        let objects: Vec<_> = parsed
            .items
            .iter()
            .map(|item| {
                (
                    item.name.clone(),
                    item.size.parse::<u64>().unwrap(),
                    parse_iso8601_millis(&item.updated).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            objects,
            vec![
                ("db/events/a.parquet".to_string(), 123, 1767323045678),
                ("db/events/b.parquet".to_string(), 4096, 1767323046000),
            ]
        );
    }

    fn auth(emulated: bool, token: Option<CachedToken>) -> GcsAuth {
        GcsAuth {
            agent: ureq::AgentBuilder::new().build(),
            token: RwLock::new(token),
            credentials_file: None,
            emulated,
        }
    }

    #[test]
    fn emulated_auth_never_mints() {
        let auth = auth(true, None);

        let header = auth.header().unwrap();

        // No credentials in a test env; an emulator ignores `Authorization`.
        assert_eq!(&*header, "Bearer emulator");
    }

    #[test]
    fn current_reads_cached_header_without_minting() {
        let cached = CachedToken {
            header: Arc::from("Bearer abc"),
            expires_at: u64::MAX,
        };
        let auth = auth(false, Some(cached));

        // Worker hot path: a plain read of the cell, no network.
        assert_eq!(auth.current().as_deref(), Some("Bearer abc"));
        // A still-valid token short-circuits the (blocking) mint too.
        assert_eq!(&*auth.header().unwrap(), "Bearer abc");
    }

    #[test]
    fn current_is_none_before_any_token_is_minted() {
        let auth = auth(false, None);

        assert!(auth.current().is_none());
    }

    #[test]
    fn read_and_write_urls_address_the_same_object() {
        let store = GcsStore::from_uri("gs://bucket/db").unwrap();
        let key = ObjectPath::new("events/a b.parquet");

        let media = store.media_url(&key);
        let upload = store.upload_url(&key);

        // A read goes through the JSON API, where the whole key is one escaped
        // `{object}` parameter.
        assert_eq!(
            media,
            "https://storage.googleapis.com/storage/v1/b/bucket/o/db%2Fevents%2Fa%20b.parquet?alt=media"
        );
        // A write goes through the XML API, where the key's separators stay
        // path separators.
        assert_eq!(
            upload,
            "https://storage.googleapis.com/bucket/db/events/a%20b.parquet"
        );
    }
}
