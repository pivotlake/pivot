//! Google Cloud Storage backend: blocking HTTP via [`ureq`] over the GCS JSON
//! API, authenticated with an OAuth2 bearer token.
//!
//! The token is minted (and cached until shortly before expiry) from, in order:
//! `GOOGLE_APPLICATION_CREDENTIALS` pointing at a **service-account** key (a
//! self-signed RS256 JWT exchanged for an access token) or an **authorized_user**
//! file (refresh-token grant), else the **GCE metadata server** (workload
//! identity on Google compute). RS256 signing uses `ring`; everything is
//! synchronous, no async runtime.

use super::{join_prefix, ObjectStore, PutOutcome, Result, StoreError};
use base64::Engine;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const TOKEN_SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_write";
const OAUTH_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
const METADATA_TOKEN_URI: &str = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

#[derive(Debug)]
pub struct GcsStore {
    bucket: String,
    prefix: String,
    agent: ureq::Agent,
    token: Mutex<Option<CachedToken>>,
}

#[derive(Debug, Clone)]
struct CachedToken {
    value: String,
    /// Unix seconds after which the token must be re-minted.
    expires_at: u64,
}

impl GcsStore {
    /// Parse `gs://bucket/prefix`.
    pub fn from_uri(uri: &str) -> Result<Self> {
        let rest = uri
            .strip_prefix("gs://")
            .ok_or_else(|| StoreError::UnsupportedUri(uri.to_string()))?;
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        Ok(Self {
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            agent: ureq::AgentBuilder::new().build(),
            token: Mutex::new(None),
        })
    }

    /// A valid bearer token, minting and caching a fresh one when needed.
    fn bearer(&self) -> Result<String> {
        let now = unix_now();
        {
            let cache = self.token.lock().unwrap();
            if let Some(tok) = cache.as_ref() {
                // 60s safety margin so an in-flight request never uses a token
                // that expires mid-flight.
                if tok.expires_at > now + 60 {
                    return Ok(tok.value.clone());
                }
            }
        }
        let (value, expires_in) = self.mint_token()?;
        let token = CachedToken {
            value: value.clone(),
            expires_at: now + expires_in,
        };
        *self.token.lock().unwrap() = Some(token);
        Ok(value)
    }

    /// Mint a fresh access token, returning `(token, lifetime_seconds)`.
    fn mint_token(&self) -> Result<(String, u64)> {
        if let Ok(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            let bytes = std::fs::read(&path).map_err(|source| StoreError::Io {
                key: path.clone(),
                source,
            })?;
            let creds: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| StoreError::Config(format!("parsing {path}: {e}")))?;
            return match creds.get("type").and_then(|t| t.as_str()) {
                Some("service_account") => self.service_account_token(&creds),
                Some("authorized_user") => self.authorized_user_token(&creds),
                other => Err(StoreError::Config(format!(
                    "unsupported credential type {other:?} in {path}"
                ))),
            };
        }
        self.metadata_token()
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
                    "no GOOGLE_APPLICATION_CREDENTIALS and metadata server unavailable: {e}"
                ))
            })?;
        parse_token_response(resp)
    }

    /// `storage.googleapis.com` object path for a catalog-relative key.
    fn object_path(&self, key: &str) -> String {
        percent_encode(&join_prefix(&self.prefix, key))
    }
}

impl ObjectStore for GcsStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let token = self.bearer()?;
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}?alt=media",
            self.bucket,
            self.object_path(key)
        );
        match self
            .agent
            .get(&url)
            .set("Authorization", &format!("Bearer {token}"))
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

    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<PutOutcome> {
        let token = self.bearer()?;
        // ifGenerationMatch=0 => create only if the object does not exist: the
        // GCS conditional create that backs a catalog CAS commit.
        let url = format!(
            "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=media&name={}&ifGenerationMatch=0",
            self.bucket,
            self.object_path(key)
        );
        match self
            .agent
            .post(&url)
            .set("Authorization", &format!("Bearer {token}"))
            .set("Content-Type", "application/octet-stream")
            .send_bytes(data)
        {
            Ok(_) => Ok(PutOutcome::Created),
            Err(ureq::Error::Status(412, _)) => Ok(PutOutcome::AlreadyExists),
            Err(e) => Err(StoreError::Http(format!("GCS PUT {key}: {e}"))),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let token = self.bearer()?;
        let object_prefix = join_prefix(&self.prefix, prefix);
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o?prefix={}%2F&delimiter=%2F",
            self.bucket,
            percent_encode(&object_prefix)
        );
        let resp = self
            .agent
            .get(&url)
            .set("Authorization", &format!("Bearer {token}"))
            .call()
            .map_err(|e| StoreError::Http(format!("GCS LIST {object_prefix}: {e}")))?;
        let body: ListResponse = resp
            .into_json()
            .map_err(|e| StoreError::Http(format!("GCS LIST parse: {e}")))?;

        let strip = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("{}/", self.prefix.trim_matches('/'))
        };
        Ok(body
            .items
            .into_iter()
            .map(|it| it.name.strip_prefix(&strip).unwrap_or(&it.name).to_string())
            .collect())
    }

    fn describe(&self) -> String {
        format!("gs://{}/{}", self.bucket, self.prefix)
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
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default = "default_expiry")]
    expires_in: u64,
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
