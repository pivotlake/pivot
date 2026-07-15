//! Google Cloud Storage backend: blocking HTTP via [`ureq`] over the GCS JSON
//! API, authenticated with an OAuth2 bearer token — for both the control-plane
//! metadata ops here *and* the io_uring range reads, which carry a fresh
//! `Authorization: Bearer` header rather than a presigned URL (presigned URLs
//! expire after an hour, stranding files cached across queries).
//!
//! The token is minted (and cached until shortly before expiry) following the
//! standard Application Default Credentials chain, in order: the file at
//! `GOOGLE_APPLICATION_CREDENTIALS`, else the well-known file that
//! `gcloud auth application-default login` writes (so a developer with gcloud
//! needs no extra config), else the **GCE metadata server** (workload identity
//! on Google compute). A credentials file is either a **service-account** key
//! (a self-signed RS256 JWT exchanged for an access token) or an
//! **authorized_user** file (refresh-token grant). Minting is a blocking `ureq` call kept off the
//! ring: it happens on the control thread (every query primes the token via
//! [`source`](GcsStore::source)), so a worker only ever *reads* a current token.
//! RS256 signing uses `ring`; everything is synchronous, no async runtime.

use super::{
    DataFileSource, FileRef, ObjectPath, ObjectStore, Result, StoreError, UploadMethod,
    UploadTarget, object_key,
};
use base64::Engine;
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
/// The real GCS JSON API origin. Tests point at an emulator instead via
/// `STORAGE_EMULATOR_HOST` (the convention `fake-gcs-server` and the Google
/// client libraries share).
const DEFAULT_ENDPOINT: &str = "https://storage.googleapis.com";

#[derive(Debug)]
pub struct GcsStore {
    bucket: String,
    prefix: String,
    /// JSON API origin (scheme + `host[:port]`, no trailing slash). The real
    /// service by default; an emulator when `STORAGE_EMULATOR_HOST` is set.
    endpoint: String,
    /// Shared token state. An `Arc` so [`source`](Self::source) can hand workers
    /// a closure that reads the same cell this store re-mints.
    auth: Arc<GcsAuth>,
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
    /// Parse `gs://bucket/prefix`. Honors `STORAGE_EMULATOR_HOST` (e.g.
    /// `http://localhost:4443`) to target a `fake-gcs-server` emulator instead
    /// of the real service — the scheme is optional and defaults to `http`.
    pub fn from_uri(uri: &str) -> Result<Self> {
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
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            endpoint,
            auth: Arc::new(GcsAuth {
                agent: ureq::AgentBuilder::new().build(),
                token: RwLock::new(None),
                emulated,
            }),
        })
    }

    /// `storage.googleapis.com` object path for a catalog-relative key.
    fn object_path(&self, key: &ObjectPath) -> String {
        percent_encode(&object_key(&self.prefix, key))
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
        if let Some(tok) = self.token.read().unwrap().as_ref()
            && tok.expires_at > now + TOKEN_REFRESH_MARGIN
        {
            return Ok(tok.header.clone());
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
            .map(|t| t.header.clone())
    }

    /// Mint a fresh access token, returning `(token, lifetime_seconds)`, walking
    /// the Application Default Credentials chain: an explicit
    /// `GOOGLE_APPLICATION_CREDENTIALS` file, else the gcloud-written well-known
    /// file, else the metadata server.
    fn mint_token(&self) -> Result<(String, u64)> {
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
        let creds: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| StoreError::Config(format!("parsing {path}: {e}")))?;
        match creds.get("type").and_then(|t| t.as_str()) {
            Some("service_account") => self.service_account_token(&creds),
            Some("authorized_user") => self.authorized_user_token(&creds),
            other => Err(StoreError::Config(format!(
                "unsupported credential type {other:?} in {path}"
            ))),
        }
    }

    /// Service-account flow: build and RS256-sign a JWT asserting our identity,
    /// then exchange it for an access token.
    fn service_account_token(&self, creds: &serde_json::Value) -> Result<(String, u64)> {
        let client_email = field(creds, "client_email")?;
        let private_key_pem = field(creds, "private_key")?;
        let token_uri = creds
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

        let resp = self
            .agent
            .post(token_uri)
            .send_form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", &jwt),
            ])
            .map_err(|e| StoreError::Http(format!("gcs token exchange: {e}")))?;
        parse_token_response(resp)
    }

    /// Authorized-user flow (`gcloud auth application-default login`): exchange a
    /// refresh token for an access token.
    fn authorized_user_token(&self, creds: &serde_json::Value) -> Result<(String, u64)> {
        let client_id = field(creds, "client_id")?;
        let client_secret = field(creds, "client_secret")?;
        let refresh_token = field(creds, "refresh_token")?;
        let resp = self
            .agent
            .post(OAUTH_TOKEN_URI)
            .send_form(&[
                ("client_id", &client_id),
                ("client_secret", &client_secret),
                ("refresh_token", &refresh_token),
                ("grant_type", "refresh_token"),
            ])
            .map_err(|e| StoreError::Http(format!("gcs token refresh: {e}")))?;
        parse_token_response(resp)
    }

    /// GCE/GKE/Cloud Run metadata server (workload identity).
    fn metadata_token(&self) -> Result<(String, u64)> {
        let resp = self
            .agent
            .get(METADATA_TOKEN_URI)
            .set("Metadata-Flavor", "Google")
            .call()
            .map_err(|e| {
                StoreError::Config(format!(
                    "no GOOGLE_APPLICATION_CREDENTIALS, no gcloud application-default \
                     credentials, and metadata server unavailable: {e}"
                ))
            })?;
        parse_token_response(resp)
    }
}

impl ObjectStore for GcsStore {
    fn describe(&self) -> String {
        format!("gs://{}/{}", self.bucket, self.prefix)
    }

    fn get(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>> {
        let header = self.auth.header()?;
        let url = format!(
            "{}/storage/v1/b/{}/o/{}?alt=media",
            self.endpoint,
            self.bucket,
            self.object_path(key)
        );
        match self
            .auth
            .agent
            .get(&url)
            .set("Authorization", &header)
            .call()
        {
            Ok(resp) => {
                let mut buf = Vec::new();
                use std::io::Read;
                resp.into_reader()
                    .read_to_end(&mut buf)
                    .map_err(|source| StoreError::Io {
                        key: key.to_string(),
                        source,
                    })?;
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
            self.object_path(key)
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
            self.object_path(key)
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
            self.object_path(key)
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

    fn list(&self, prefix: &ObjectPath) -> Result<Vec<FileRef>> {
        self.list_impl(prefix, None)
    }

    fn list_from(&self, prefix: &ObjectPath, start: &ObjectPath) -> Result<Vec<FileRef>> {
        self.list_impl(prefix, Some(start))
    }

    fn source(&self, key: &ObjectPath) -> Result<DataFileSource> {
        // The same media URL the JSON API serves for `get`, but range-read
        // straight off the ring. Unlike a presigned URL it's stable (no embedded
        // signature to expire) — auth rides in a per-request `Authorization`
        // header instead, so a compressed cached across queries never goes stale.
        let url = format!(
            "{}/storage/v1/b/{}/o/{}?alt=media",
            self.endpoint,
            self.bucket,
            self.object_path(key)
        );
        let url = url::Url::parse(&url)
            .map_err(|e| StoreError::Config(format!("building gcs media url: {e}")))?;
        Ok(DataFileSource::Remote {
            url,
            auth: self.ring_auth()?,
        })
    }

    fn upload_target(&self, key: &ObjectPath) -> Result<UploadTarget> {
        // The JSON API media-upload endpoint the blocking `put` posts to, but
        // sent straight off the ring. Auth rides in a per-request `Authorization`
        // header (a fresh bearer token), mirroring `source`.
        let url = format!(
            "{}/upload/storage/v1/b/{}/o?uploadType=media&name={}",
            self.endpoint,
            self.bucket,
            self.object_path(key)
        );
        let url = url::Url::parse(&url)
            .map_err(|e| StoreError::Config(format!("building gcs upload url: {e}")))?;
        Ok(UploadTarget::Remote {
            url,
            method: UploadMethod::Post,
            auth: self.ring_auth()?,
            content_type: Some("application/octet-stream"),
        })
    }
}

impl GcsStore {
    /// The per-request `Authorization` provider for ring IO (reads and uploads):
    /// `None` against an emulator, which ignores auth, else a fresh bearer token.
    /// The token is primed here on the control thread (a blocking mint is fine off
    /// the ring) so a query's workers only ever read a cached one and never mint.
    fn ring_auth(&self) -> Result<Option<dispatch::io::AuthHeader>> {
        if self.auth.emulated {
            return Ok(None);
        }
        self.auth.header()?;
        let auth = self.auth.clone();
        Ok(Some(Arc::new(move || auth.current())))
    }

    /// One-page list of objects directly under `prefix`, optionally beginning at
    /// `start` (`startOffset`) so the scan skips everything lexicographically
    /// below it.
    fn list_impl(&self, prefix: &ObjectPath, start: Option<&ObjectPath>) -> Result<Vec<FileRef>> {
        let header = self.auth.header()?;
        let object_prefix = object_key(&self.prefix, prefix);
        let mut url = format!(
            "{}/storage/v1/b/{}/o?prefix={}%2F&delimiter=%2F",
            self.endpoint,
            self.bucket,
            percent_encode(&object_prefix)
        );
        if let Some(start) = start {
            url.push_str("&startOffset=");
            url.push_str(&percent_encode(&object_key(&self.prefix, start)));
        }
        let resp = self
            .auth
            .agent
            .get(&url)
            .set("Authorization", &header)
            .call()
            .map_err(|e| StoreError::Http(format!("GCS LIST {object_prefix}: {e}")))?;
        let body: ListResponse = resp
            .into_json()
            .map_err(|e| StoreError::Http(format!("GCS LIST parse: {e}")))?;

        body.items
            .into_iter()
            .map(|it| {
                Ok(FileRef {
                    path: ObjectPath::new(super::key_name(&it.name)),
                    size: it.size.parse().map_err(|_| {
                        StoreError::Http(format!(
                            "GCS LIST: bad size `{}` for {}",
                            it.size, it.name
                        ))
                    })?,
                })
            })
            .collect()
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

fn parse_token_response(resp: ureq::Response) -> Result<(String, u64)> {
    let tok: TokenResponse = resp
        .into_json()
        .map_err(|e| StoreError::Config(format!("parsing token response: {e}")))?;
    Ok((tok.access_token, tok.expires_in))
}

fn field(creds: &serde_json::Value, name: &str) -> Result<String> {
    creds
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
        .filter(|l| !l.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");
    base64::engine::general_purpose::STANDARD
        .decode(body.trim())
        .map_err(|e| StoreError::Config(format!("decoding PEM private key: {e}")))
}

/// Percent-encode a GCS object name for use in a path/query (RFC 3986
/// unreserved pass through; `/` and everything else escaped).
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_response_parses_names_and_string_sizes() {
        // GCS reports object size as a decimal string; the reader parses it to
        // the byte size it needs to locate a Parquet footer.
        let json = r#"{"items":[
            {"name":"db/events/a.parquet","size":"123"},
            {"name":"db/events/b.parquet","size":"4096"}
        ]}"#;
        let parsed: ListResponse = serde_json::from_str(json).unwrap();
        let objects: Vec<_> = parsed
            .items
            .iter()
            .map(|it| (it.name.clone(), it.size.parse::<u64>().unwrap()))
            .collect();
        assert_eq!(
            objects,
            vec![
                ("db/events/a.parquet".to_string(), 123),
                ("db/events/b.parquet".to_string(), 4096),
            ]
        );
    }

    fn auth(emulated: bool, token: Option<CachedToken>) -> GcsAuth {
        GcsAuth {
            agent: ureq::AgentBuilder::new().build(),
            token: RwLock::new(token),
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
}
