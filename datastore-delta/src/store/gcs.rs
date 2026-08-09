//! Google Cloud Storage backend: blocking HTTP via [`ureq`] against the GCS XML
//! API, every request carrying an OAuth2 `Authorization: Bearer <token>` header.
//!
//! GCS has no signed-URL scheme the ring can use without a round trip to mint
//! one, so where [`S3Store`](super::S3Store) presigns a per-object URL this
//! backend hands out a **stable** object URL paired with an
//! [`AuthHeader`](dispatch::io::AuthHeader). That closure only ever reads the
//! token this store already holds; minting is never on a worker's path. The
//! store refreshes the token on the coordinator instead: every blocking
//! operation, and [`ObjectStore::prepare_write`] before an INSERT dataflow,
//! goes through [`GcsStore::access_token`], which re-mints only once the
//! cached one is close to expiring.
//!
//! Tokens come from one of three places, named by [`GcsAuth`]: a service
//! account key (signed into a JWT assertion and exchanged for an access token),
//! the instance metadata server (the service account attached to a GCE/GKE
//! node), or a bearer token handed over ready-made. [`GcsStore::from_uri`]
//! picks one from the environment; [`GcsStore::with_credentials`] takes it
//! explicitly, so a metastore can supply a datastore's own credentials rather
//! than relying on ambient environment. Either way the same credentials build
//! the async client Delta Kernel reads the `_delta_log` with
//! ([`ObjectStore::build_delta_object_store`]).

use super::list_bucket::{parse_listing, percent_encode};
use super::{
    DataFileLocation, ListedObject, ObjectPath, ObjectStore, Result, StoreError, object_key,
};
use base64::Engine;
use delta_kernel::object_store::gcp::{GcpCredential, GoogleCloudStorageBuilder};
use delta_kernel::object_store::{
    ClientConfigKey, DynObjectStore, StaticCredentialProvider, gcp::GoogleConfigKey,
};
use dispatch::io::AuthHeader;
use std::fmt::{self, Debug};
use std::io::Read;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// The public GCS endpoint, overridden only to address an emulator.
const DEFAULT_ENDPOINT: &str = "https://storage.googleapis.com";

/// The metadata server host when `GCE_METADATA_HOST` does not name another.
const DEFAULT_METADATA_HOST: &str = "metadata.google.internal";

/// Where a service account key's own `token_uri` is absent.
const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

/// The OAuth2 scope a minted token is requested for: read and write objects,
/// which is everything this backend does (it never administers buckets).
const STORAGE_SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_write";

/// How long a self-signed JWT assertion is valid for. Google caps the
/// assertion's own lifetime at an hour; it is spent immediately on the token
/// exchange, so this only has to outlive that one request.
const ASSERTION_LIFETIME: Duration = Duration::from_secs(3600);

/// Re-mint this long before a token actually expires, so a request never leaves
/// with a token that dies in flight.
const TOKEN_REFRESH_MARGIN: Duration = Duration::from_secs(300);

pub struct GcsStore {
    /// The `gs://bucket/prefix` URI this store was opened with, kept verbatim as
    /// its addressable root (see [`ObjectStore::location_uri`]).
    uri: String,
    /// In-bucket prefix under which this datastore's keys live.
    prefix: String,
    /// The endpoint override this store was opened with, if any.
    endpoint: Option<String>,
    /// Base origin including the bucket, e.g.
    /// `https://storage.googleapis.com/my-bucket`.
    base: String,
    auth: GcsAuth,
    token: Arc<TokenCache>,
    agent: ureq::Agent,
}

/// Explicit GCS credentials and connection parameters for
/// [`GcsStore::with_credentials`], as a metastore holds them per datastore.
pub struct GcsCredentials {
    pub auth: GcsAuth,
    /// A full origin (`scheme://host[:port]`) replacing the public GCS endpoint
    /// (a local emulator). `None` addresses `https://storage.googleapis.com`.
    pub endpoint: Option<String>,
}

/// Where a [`GcsStore`]'s OAuth2 access tokens come from.
pub enum GcsAuth {
    /// A service account JSON key (a key file's contents). Signed into a JWT
    /// assertion and exchanged for an access token at the key's `token_uri`.
    ServiceAccountKey(String),
    /// The instance metadata server, which mints tokens for the service account
    /// attached to a GCE instance or GKE node. No credential material is held by
    /// this process.
    MetadataServer,
    /// A bearer token supplied ready-made by whoever opened the store, used as
    /// is. This store never refreshes it, so it must outlive the process or be
    /// replaced by reopening the store.
    AccessToken(String),
}

impl GcsStore {
    /// Parse `gs://bucket/prefix` and resolve credentials from the environment,
    /// in order: `GOOGLE_APPLICATION_CREDENTIALS` (a service account key file),
    /// then `GOOGLE_OAUTH_ACCESS_TOKEN` (a ready-made bearer token), then the
    /// instance metadata server. `STORAGE_EMULATOR_HOST` overrides the endpoint.
    pub fn from_uri(uri: &str) -> Result<Self> {
        let (bucket, prefix) = parse_gs_uri(uri)?;
        let credentials = GcsCredentials {
            auth: auth_from_env()?,
            endpoint: endpoint_from_env(),
        };
        Ok(Self::build(uri, bucket, prefix, credentials))
    }

    /// Parse `gs://bucket/prefix` but take credentials and endpoint from
    /// `credentials` instead of the environment, the path a metastore uses to
    /// open a datastore with its own configured credentials.
    pub fn with_credentials(uri: &str, credentials: GcsCredentials) -> Result<Self> {
        let (bucket, prefix) = parse_gs_uri(uri)?;
        Ok(Self::build(uri, bucket, prefix, credentials))
    }

    fn build(uri: &str, bucket: &str, prefix: &str, credentials: GcsCredentials) -> Self {
        let origin = credentials
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/');
        Self {
            uri: uri.to_string(),
            prefix: prefix.to_string(),
            base: format!("{origin}/{bucket}"),
            endpoint: credentials.endpoint,
            auth: credentials.auth,
            token: Arc::new(TokenCache::default()),
            agent: ureq::AgentBuilder::new().build(),
        }
    }

    /// Full request URL for an in-bucket object name (already prefixed).
    fn url_for(&self, object: &str) -> String {
        format!("{}/{}", self.base, object)
    }

    /// The `Authorization` header value to send, minting a fresh token when the
    /// cached one is missing or close to expiring. Blocking (it may do a token
    /// round trip), so only the coordinator's own operations call it.
    fn access_token(&self) -> Result<Arc<str>> {
        if let Some(header) = self.token.fresh() {
            return Ok(header);
        }
        let (token, lifetime) = self.mint_token()?;
        let header: Arc<str> = format!("Bearer {token}").into();
        let expires_at =
            lifetime.map(|lifetime| Instant::now() + lifetime.saturating_sub(TOKEN_REFRESH_MARGIN));
        self.token.store(header.clone(), expires_at);
        Ok(header)
    }

    /// A closure the io_uring reader calls per request for its `Authorization`
    /// header. It reads this store's cached token and nothing else; see the
    /// module docs on why minting cannot happen there.
    fn auth_header(&self) -> AuthHeader {
        let token = self.token.clone();
        Arc::new(move || token.current())
    }

    /// Fetch a new access token and how long it is good for (`None` for a token
    /// supplied ready-made, which this store does not manage the lifetime of).
    fn mint_token(&self) -> Result<(String, Option<Duration>)> {
        match &self.auth {
            GcsAuth::AccessToken(token) => Ok((token.clone(), None)),
            GcsAuth::ServiceAccountKey(key) => {
                let key: ServiceAccountKey = serde_json::from_str(key).map_err(|e| {
                    StoreError::Config(format!("service account key is not valid JSON: {e}"))
                })?;
                let assertion = sign_jwt_assertion(&key)?;
                let response: TokenResponse = self
                    .agent
                    .post(&key.token_uri)
                    .send_form(&[
                        ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                        ("assertion", &assertion),
                    ])
                    .map_err(|e| StoreError::Http(format!("minting a gcs access token: {e}")))?
                    .into_json()
                    .map_err(|e| StoreError::Http(format!("gcs token response: {e}")))?;
                Ok(response.into_token())
            }
            GcsAuth::MetadataServer => {
                let host = std::env::var("GCE_METADATA_HOST")
                    .unwrap_or_else(|_| DEFAULT_METADATA_HOST.to_string());
                let url = format!(
                    "http://{host}/computeMetadata/v1/instance/service-accounts/default/token"
                );
                let response: TokenResponse = self
                    .agent
                    .get(&url)
                    .set("Metadata-Flavor", "Google")
                    .call()
                    .map_err(|e| {
                        StoreError::Http(format!("asking the metadata server for a token: {e}"))
                    })?
                    .into_json()
                    .map_err(|e| StoreError::Http(format!("metadata token response: {e}")))?;
                Ok(response.into_token())
            }
        }
    }

    /// Attach the `Authorization` header a GCS request cannot go out without.
    fn authorized(&self, request: ureq::Request) -> Result<ureq::Request> {
        Ok(request.set("Authorization", &self.access_token()?))
    }
}

impl ObjectStore for GcsStore {
    fn describe(&self) -> String {
        format!("{} (prefix `{}`)", self.base, self.prefix)
    }

    /// A no-op: GCS has no directories. A key is a flat string, and an object at
    /// `prefix/name` exists the moment it is written, with no parent to create.
    fn create_dir(&self, _prefix: &ObjectPath) -> Result<()> {
        Ok(())
    }

    fn location_uri(&self) -> String {
        self.uri.clone()
    }

    fn build_delta_object_store(&self) -> Result<Arc<DynObjectStore>> {
        let mut builder = GoogleCloudStorageBuilder::new().with_url(self.uri.as_str());
        if let Some(endpoint) = &self.endpoint {
            builder = builder
                .with_base_url(endpoint)
                .with_config(GoogleConfigKey::Client(ClientConfigKey::AllowHttp), "true");
        }
        builder = match &self.auth {
            GcsAuth::ServiceAccountKey(key) => builder.with_service_account_key(key),
            // Kernel's own client resolves the metadata server the same way.
            GcsAuth::MetadataServer => builder,
            GcsAuth::AccessToken(token) => {
                builder.with_credentials(Arc::new(StaticCredentialProvider::new(GcpCredential {
                    bearer: token.clone(),
                })))
            }
        };
        let store = builder
            .build()
            .map_err(|source| StoreError::DeltaObjectStore {
                uri: self.uri.clone(),
                source,
            })?;
        Ok(Arc::new(store))
    }

    fn get(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        match self.authorized(self.agent.get(&url))?.call() {
            Ok(response) => {
                let mut buf = Vec::new();
                response
                    .into_reader()
                    .read_to_end(&mut buf)
                    .map_err(|source| StoreError::Io {
                        key: key.to_string(),
                        source,
                    })?;
                Ok(Some(buf))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(StoreError::Http(format!("GET {object}: {e}"))),
        }
    }

    fn put(&self, key: &ObjectPath, data: &[u8]) -> Result<()> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        match self.authorized(self.agent.put(&url))?.send_bytes(data) {
            Ok(_) => Ok(()),
            Err(e) => Err(StoreError::Http(format!("PUT {object}: {e}"))),
        }
    }

    fn put_if_absent(&self, key: &ObjectPath, data: &[u8]) -> Result<bool> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        // GCS conditional write: generation 0 is "no live object", so matching it
        // fails the write with 412 when the key already exists.
        let response = self
            .authorized(self.agent.put(&url))?
            .set("x-goog-if-generation-match", "0")
            .send_bytes(data);
        match response {
            Ok(_) => Ok(true),
            Err(ureq::Error::Status(412, _)) => Ok(false),
            Err(e) => Err(StoreError::Http(format!("conditional PUT {object}: {e}"))),
        }
    }

    fn delete(&self, key: &ObjectPath) -> Result<()> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        match self.authorized(self.agent.delete(&url))?.call() {
            // DELETE is idempotent; a missing key is the goal state.
            Ok(_) | Err(ureq::Error::Status(404, _)) => Ok(()),
            Err(e) => Err(StoreError::Http(format!("DELETE {object}: {e}"))),
        }
    }

    fn list(&self, prefix: &ObjectPath) -> Result<Vec<ListedObject>> {
        let object_prefix = object_key(&self.prefix, prefix);
        // One level (delimiter=/) under the object prefix.
        let query = format!(
            "list-type=2&prefix={}%2F&delimiter=%2F",
            percent_encode(&object_prefix)
        );
        let url = format!("{}?{}", self.base, query);
        let body = match self.authorized(self.agent.get(&url))?.call() {
            Ok(response) => response
                .into_string()
                .map_err(|e| StoreError::Http(format!("LIST body: {e}")))?,
            Err(e) => return Err(StoreError::Http(format!("LIST {object_prefix}: {e}"))),
        };
        parse_listing(&body)
    }

    fn source(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        self.remote(key)
    }

    fn sink(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        self.remote(key)
    }

    fn prepare_write(&self) -> Result<()> {
        self.access_token().map(|_| ())
    }
}

impl GcsStore {
    /// The object's stable URL plus the header closure that authenticates each
    /// request against it. Reads and writes address the same URL (GCS
    /// distinguishes them by method), so both directions land here.
    ///
    /// The token is minted **now**, on the caller's thread, so the closure the
    /// ring ends up calling always finds one in the cache.
    fn remote(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        let url = self.url_for(&object_key(&self.prefix, key));
        let url = url::Url::parse(&url)
            .map_err(|e| StoreError::Http(format!("parsing object url `{url}`: {e}")))?;
        self.access_token()?;
        Ok(DataFileLocation::Remote {
            url,
            auth: Some(self.auth_header()),
        })
    }
}

/// Redacts the credentials: a service account key holds a private key, and a
/// bearer token is itself the secret.
impl Debug for GcsStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcsStore")
            .field("uri", &self.uri)
            .field("base", &self.base)
            .field("prefix", &self.prefix)
            .finish()
    }
}

/// The access token in hand, shared between the store, which refreshes it, and
/// the [`AuthHeader`] closures handed to the io_uring reader, which only read
/// it.
#[derive(Default)]
struct TokenCache {
    token: RwLock<Option<CachedToken>>,
}

struct CachedToken {
    /// The complete `Authorization` header value, `Bearer <token>`.
    header: Arc<str>,
    /// When this token stops being usable, already pulled back by
    /// [`TOKEN_REFRESH_MARGIN`]. `None` for a token supplied ready-made, whose
    /// lifetime this store does not know and does not manage.
    expires_at: Option<Instant>,
}

impl TokenCache {
    /// The cached header if it is good for a while yet, else `None`, on which
    /// the caller mints.
    fn fresh(&self) -> Option<Arc<str>> {
        let cached = self.token.read().unwrap();
        let cached = cached.as_ref()?;
        match cached.expires_at {
            None => Some(cached.header.clone()),
            Some(expires_at) => (Instant::now() < expires_at).then(|| cached.header.clone()),
        }
    }

    /// The cached header whatever its age. This is the read path's view: it
    /// cannot mint, so an aging token is the best it can send, and refreshing
    /// before the margin runs out is the coordinator's job.
    fn current(&self) -> Option<Arc<str>> {
        self.token
            .read()
            .unwrap()
            .as_ref()
            .map(|cached| cached.header.clone())
    }

    fn store(&self, header: Arc<str>, expires_at: Option<Instant>) {
        *self.token.write().unwrap() = Some(CachedToken { header, expires_at });
    }
}

/// The fields this backend reads out of a service account JSON key.
#[derive(serde::Deserialize)]
struct ServiceAccountKey {
    client_email: String,
    /// PEM-encoded PKCS#8 RSA private key.
    private_key: String,
    #[serde(default = "default_token_uri")]
    token_uri: String,
}

fn default_token_uri() -> String {
    DEFAULT_TOKEN_URI.to_string()
}

/// An OAuth2 token endpoint's response (the metadata server answers alike).
#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

impl TokenResponse {
    /// The minted token and how long it is good for, in the shape
    /// [`GcsStore::mint_token`] returns.
    fn into_token(self) -> (String, Option<Duration>) {
        (
            self.access_token,
            Some(Duration::from_secs(self.expires_in)),
        )
    }
}

/// Build the signed JWT assertion that buys an access token: a `RS256` header
/// and claims naming the service account, the scope, and the token endpoint the
/// assertion may be spent at, signed with the key's private half.
fn sign_jwt_assertion(key: &ServiceAccountKey) -> Result<String> {
    let issued_at = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| StoreError::Config(format!("system clock is before the unix epoch: {e}")))?
        .as_secs();
    let claims = serde_json::json!({
        "iss": key.client_email,
        "scope": STORAGE_SCOPE,
        "aud": key.token_uri,
        "iat": issued_at,
        "exp": issued_at + ASSERTION_LIFETIME.as_secs(),
    });
    let message = format!(
        "{}.{}",
        base64url(br#"{"alg":"RS256","typ":"JWT"}"#),
        base64url(claims.to_string().as_bytes())
    );

    let key_pair = ring::signature::RsaKeyPair::from_pkcs8(&pkcs8_der(&key.private_key)?)
        .map_err(|e| StoreError::Config(format!("service account private key: {e}")))?;
    let mut signature = vec![0u8; key_pair.public().modulus_len()];
    key_pair
        .sign(
            &ring::signature::RSA_PKCS1_SHA256,
            &ring::rand::SystemRandom::new(),
            message.as_bytes(),
            &mut signature,
        )
        .map_err(|e| StoreError::Config(format!("signing the service account assertion: {e}")))?;

    Ok(format!("{message}.{}", base64url(&signature)))
}

/// Decode a PEM private key into the PKCS#8 DER bytes the signer takes.
fn pkcs8_der(pem: &str) -> Result<Vec<u8>> {
    let body: String = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .flat_map(|line| line.chars().filter(|c| !c.is_whitespace()))
        .collect();
    base64::engine::general_purpose::STANDARD
        .decode(body)
        .map_err(|e| StoreError::Config(format!("service account private key is not PEM: {e}")))
}

/// Base64url without padding, as JWT encodes each of its three parts.
fn base64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Split a `gs://bucket/prefix` (or `gcs://…`) URI into its bucket and in-bucket
/// prefix (empty when the URI names only a bucket).
fn parse_gs_uri(uri: &str) -> Result<(&str, &str)> {
    let rest = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("gcs://"))
        .ok_or_else(|| StoreError::UnsupportedUri(uri.to_string()))?;
    Ok(rest.split_once('/').unwrap_or((rest, "")))
}

/// Resolve where tokens come from when nobody configured it explicitly. The
/// order is Google's own: an application-credentials key file, then a token
/// placed in the environment, then the instance's attached service account.
fn auth_from_env() -> Result<GcsAuth> {
    if let Ok(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        let key = std::fs::read_to_string(&path)
            .map_err(|source| StoreError::Io { key: path, source })?;
        return Ok(GcsAuth::ServiceAccountKey(key));
    }
    if let Ok(token) = std::env::var("GOOGLE_OAUTH_ACCESS_TOKEN") {
        return Ok(GcsAuth::AccessToken(token));
    }
    Ok(GcsAuth::MetadataServer)
}

/// `STORAGE_EMULATOR_HOST` as an origin. The convention allows a bare
/// `host:port`, which names a plaintext emulator.
fn endpoint_from_env() -> Option<String> {
    let host = std::env::var("STORAGE_EMULATOR_HOST").ok()?;
    Some(match host.contains("://") {
        true => host,
        false => format!("http://{host}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway service account key (its RSA key pair exists only here) in
    /// the shape Google hands one out, for the assertion-signing tests.
    const TEST_SERVICE_ACCOUNT: &str = include_str!("testdata/service_account.json");

    fn store_with(auth: GcsAuth, endpoint: Option<String>) -> GcsStore {
        GcsStore::with_credentials("gs://bucket/db", GcsCredentials { auth, endpoint }).unwrap()
    }

    #[test]
    fn gs_uri_splits_into_bucket_and_prefix() {
        assert_eq!(parse_gs_uri("gs://bucket/db").unwrap(), ("bucket", "db"));
        assert_eq!(parse_gs_uri("gcs://bucket/db").unwrap(), ("bucket", "db"));
        // A bucket-only URI has no in-bucket prefix.
        assert_eq!(parse_gs_uri("gs://bucket").unwrap(), ("bucket", ""));
        assert!(parse_gs_uri("s3://bucket/db").is_err());
    }

    #[test]
    fn objects_address_the_endpoint_the_store_was_opened_with() {
        let public = store_with(GcsAuth::MetadataServer, None);
        assert_eq!(
            public.url_for("db/events/a.parquet"),
            "https://storage.googleapis.com/bucket/db/events/a.parquet"
        );

        let emulated = store_with(
            GcsAuth::MetadataServer,
            Some("http://127.0.0.1:4443/".to_string()),
        );
        assert_eq!(
            emulated.url_for("db/events/a.parquet"),
            "http://127.0.0.1:4443/bucket/db/events/a.parquet"
        );
    }

    #[test]
    fn a_ready_made_token_is_handed_out_and_never_re_minted() {
        let store = store_with(GcsAuth::AccessToken("ya29.test".to_string()), None);

        let header = store.access_token().unwrap();

        assert_eq!(&*header, "Bearer ya29.test");
        // No expiry, so the cache keeps serving it, and the read path's own
        // view sees the same header without minting anything.
        assert_eq!(store.token.fresh().as_deref(), Some("Bearer ya29.test"));
        assert_eq!((store.auth_header())().as_deref(), Some("Bearer ya29.test"));
    }

    #[test]
    fn a_token_within_the_refresh_margin_is_stale() {
        let cache = TokenCache::default();

        cache.store(
            "Bearer expiring".into(),
            Some(Instant::now() - Duration::from_secs(1)),
        );

        // The coordinator re-mints...
        assert!(cache.fresh().is_none());
        // ...but the read path still sends what it has rather than nothing.
        assert_eq!(cache.current().as_deref(), Some("Bearer expiring"));
    }

    #[test]
    fn the_read_path_has_no_header_before_the_store_mints_one() {
        let store = store_with(GcsAuth::MetadataServer, None);

        assert!((store.auth_header())().is_none());
    }

    #[test]
    fn the_delta_client_builds_from_every_credential_source() {
        // Kernel reads the `_delta_log` through its own client, so each way of
        // authenticating has to reach it, including the metadata server, where
        // this process configures no credential at all.
        for auth in [
            GcsAuth::MetadataServer,
            GcsAuth::AccessToken("ya29.test".to_string()),
            GcsAuth::ServiceAccountKey(TEST_SERVICE_ACCOUNT.to_string()),
        ] {
            store_with(auth, None).build_delta_object_store().unwrap();
        }
        // An emulator's plaintext endpoint reaches it too.
        store_with(
            GcsAuth::AccessToken("ya29.test".to_string()),
            Some("http://127.0.0.1:4443".to_string()),
        )
        .build_delta_object_store()
        .unwrap();
    }

    #[test]
    fn a_service_account_assertion_is_signed_by_the_key_and_claims_the_storage_scope() {
        let key: ServiceAccountKey = serde_json::from_str(TEST_SERVICE_ACCOUNT).unwrap();

        let assertion = sign_jwt_assertion(&key).unwrap();

        let parts: Vec<&str> = assertion.split('.').collect();
        assert_eq!(parts.len(), 3, "header.claims.signature");
        let decode = |part: &str| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(part)
                .unwrap()
        };
        assert_eq!(decode(parts[0]), br#"{"alg":"RS256","typ":"JWT"}"#);
        let claims: serde_json::Value = serde_json::from_slice(&decode(parts[1])).unwrap();
        assert_eq!(claims["iss"], key.client_email);
        assert_eq!(claims["scope"], STORAGE_SCOPE);
        // The assertion may only be spent at the endpoint it names, and only
        // within its own window.
        assert_eq!(claims["aud"], key.token_uri);
        assert_eq!(
            claims["exp"].as_u64().unwrap() - claims["iat"].as_u64().unwrap(),
            ASSERTION_LIFETIME.as_secs()
        );

        // The signature verifies against the key's public half, which is what
        // Google checks before handing back a token.
        let key_pair =
            ring::signature::RsaKeyPair::from_pkcs8(&pkcs8_der(&key.private_key).unwrap()).unwrap();
        let public_key = ring::signature::UnparsedPublicKey::new(
            &ring::signature::RSA_PKCS1_2048_8192_SHA256,
            key_pair.public().as_ref(),
        );
        let signed_message = format!("{}.{}", parts[0], parts[1]);
        public_key
            .verify(signed_message.as_bytes(), &decode(parts[2]))
            .expect("the assertion carries a valid RS256 signature");
    }

    #[test]
    fn a_private_key_that_is_not_pem_fails_to_sign() {
        let key = ServiceAccountKey {
            client_email: "pivot@example.iam.gserviceaccount.com".to_string(),
            private_key: "not a key".to_string(),
            token_uri: DEFAULT_TOKEN_URI.to_string(),
        };

        let error = sign_jwt_assertion(&key).unwrap_err();

        assert!(error.to_string().contains("service account private key"));
    }

    #[test]
    fn debug_redacts_the_credentials() {
        let store = store_with(GcsAuth::AccessToken("ya29.secret".to_string()), None);

        assert!(!format!("{store:?}").contains("ya29.secret"));
    }
}
