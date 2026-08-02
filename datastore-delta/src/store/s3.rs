//! S3 (and S3-compatible) backend: blocking HTTP via [`ureq`], requests signed
//! with SigV4 via `aws_sigv4::http_request::sign` (a pure function — no runtime).
//!
//! Credentials come from one of two places. [`S3Store::from_uri`] reads them from
//! the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`; region from
//! `AWS_REGION`/`AWS_DEFAULT_REGION`; an optional
//! `AWS_ENDPOINT_URL` selects a path-style S3-compatible endpoint like MinIO).
//! [`S3Store::with_credentials`] takes them explicitly, so a metastore can supply
//! a datastore's own key/secret/region/endpoint rather than relying on ambient
//! environment.
//!
//! Either way the resolved parameters are kept on the store, which builds the
//! Delta Kernel client for the same bucket from them
//! ([`ObjectStore::build_delta_object_store`])
//! instead of letting it resolve its own.

use super::{DataFileLocation, FileRef, ObjectPath, ObjectStore, Result, StoreError, object_key};
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    PayloadChecksumKind, SignableBody, SignableRequest, SignatureLocation, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::object_store::aws::AmazonS3Builder;
use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

#[derive(Debug)]
pub struct S3Store {
    /// The `s3://bucket/prefix` URI this store was opened with, kept verbatim
    /// as its addressable root (see [`ObjectStore::location_uri`]).
    uri: String,
    /// In-bucket prefix under which this datastore's keys live.
    prefix: String,
    region: String,
    access_key: String,
    secret_key: String,
    /// The path-style endpoint override this store was opened with, if any.
    endpoint: Option<String>,
    /// Base origin, e.g. `https://bucket.s3.us-east-1.amazonaws.com` (virtual
    /// hosted) or `http://localhost:9000/bucket` (path-style endpoint override).
    base: String,
    /// `Host` header value for signing.
    host: String,
    agent: ureq::Agent,
}

/// Explicit S3 credentials and connection parameters for
/// [`S3Store::with_credentials`], as a metastore holds them per datastore.
pub struct S3Credentials {
    pub region: String,
    pub access_key: String,
    pub secret_key: String,
    /// A path-style S3-compatible endpoint (e.g. MinIO). `None` uses AWS
    /// virtual-hosted style.
    pub endpoint: Option<String>,
}

impl S3Store {
    /// Parse `s3://bucket/prefix` and resolve credentials/region/endpoint from
    /// the environment.
    pub fn from_uri(uri: &str) -> Result<Self> {
        let (bucket, prefix) = parse_s3_uri(uri)?;

        let region = env_any(&["AWS_REGION", "AWS_DEFAULT_REGION"])
            .unwrap_or_else(|| "us-east-1".to_string());
        let credentials = S3Credentials {
            region,
            access_key: env_req("AWS_ACCESS_KEY_ID")?,
            secret_key: env_req("AWS_SECRET_ACCESS_KEY")?,
            endpoint: std::env::var("AWS_ENDPOINT_URL").ok(),
        };
        Ok(Self::build(uri, bucket, prefix, credentials))
    }

    /// Parse `s3://bucket/prefix` but take credentials/region/endpoint from
    /// `credentials` instead of the environment, the path a metastore uses to
    /// open a datastore with its own configured keys.
    pub fn with_credentials(uri: &str, credentials: S3Credentials) -> Result<Self> {
        let (bucket, prefix) = parse_s3_uri(uri)?;
        Ok(Self::build(uri, bucket, prefix, credentials))
    }

    /// Assemble the store from a parsed bucket/prefix and resolved credentials,
    /// computing the signing origin and `Host` from the (optional) endpoint.
    fn build(uri: &str, bucket: &str, prefix: &str, credentials: S3Credentials) -> Self {
        let (base, host) = match credentials.endpoint.as_deref() {
            // Path-style against a custom endpoint (MinIO etc.).
            Some(ep) => {
                let ep = ep.trim_end_matches('/');
                let host = ep
                    .split("://")
                    .nth(1)
                    .unwrap_or(ep)
                    .split('/')
                    .next()
                    .unwrap_or(ep)
                    .to_string();
                (format!("{ep}/{bucket}"), host)
            }
            // Virtual-hosted style on AWS.
            None => {
                let host = format!("{bucket}.s3.{}.amazonaws.com", credentials.region);
                (format!("https://{host}"), host)
            }
        };

        Self {
            uri: uri.to_string(),
            prefix: prefix.to_string(),
            region: credentials.region,
            access_key: credentials.access_key,
            secret_key: credentials.secret_key,
            endpoint: credentials.endpoint,
            base,
            host,
            agent: ureq::AgentBuilder::new().build(),
        }
    }

    /// Full request URL for an in-bucket object name (already prefixed).
    fn url_for(&self, object: &str) -> String {
        format!("{}/{}", self.base, object)
    }

    /// Compute the SigV4 headers (Authorization, x-amz-date, x-amz-content-sha256,
    /// optional x-amz-security-token) to attach to a request. `query` is the
    /// canonical query string (without leading `?`), included in the signature.
    fn sign(
        &self,
        method: &str,
        url: &str,
        extra_headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Vec<(String, String)>> {
        // Long-lived keys only, so neither a session token nor an expiry.
        // `Static` is the SDK's own name for keys handed over directly rather
        // than resolved by a credentials provider.
        let creds = Credentials::new(&self.access_key, &self.secret_key, None, None, "Static");
        let identity = creds.into();

        let mut settings = SigningSettings::default();
        settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;

        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name("s3")
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| StoreError::Http(format!("sigv4 params: {e}")))?;

        // Headers that participate in the signature. `host` is mandatory.
        let mut sign_headers: Vec<(&str, &str)> = vec![("host", &self.host)];
        sign_headers.extend_from_slice(extra_headers);

        let signable = SignableRequest::new(
            method,
            url,
            sign_headers.iter().copied(),
            SignableBody::Bytes(body),
        )
        .map_err(|e| StoreError::Http(format!("sigv4 signable: {e}")))?;

        let (instructions, _signature) = sign(signable, &params.into())
            .map_err(|e| StoreError::Http(format!("sigv4 sign: {e}")))?
            .into_parts();

        let mut out = Vec::new();
        for (name, value) in instructions.headers() {
            out.push((name.to_string(), value.to_string()));
        }
        Ok(out)
    }

    /// Apply signed + extra headers to a ureq request builder.
    fn apply(req: ureq::Request, headers: &[(String, String)]) -> ureq::Request {
        headers.iter().fold(req, |r, (k, v)| r.set(k, v))
    }
}

impl ObjectStore for S3Store {
    fn describe(&self) -> String {
        format!("{} (prefix `{}`)", self.base, self.prefix)
    }

    fn location_uri(&self) -> String {
        self.uri.clone()
    }

    fn build_delta_object_store(&self) -> Result<Arc<DynObjectStore>> {
        let mut builder = AmazonS3Builder::new()
            .with_url(self.uri.as_str())
            .with_region(&self.region)
            .with_access_key_id(&self.access_key)
            .with_secret_access_key(&self.secret_key);
        // A path-style endpoint is how an S3-compatible service (MinIO, the
        // Google Cloud Storage interoperability API) is addressed; AWS itself
        // takes the default virtual-hosted style.
        if let Some(endpoint) = &self.endpoint {
            builder = builder
                .with_endpoint(endpoint)
                .with_allow_http(true)
                .with_virtual_hosted_style_request(false);
        }
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
        let signed = self.sign("GET", &url, &[], &[])?;
        let req = Self::apply(self.agent.get(&url), &signed);
        match req.call() {
            Ok(resp) => {
                let mut buf = Vec::new();
                resp.into_reader()
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
        let signed = self.sign("PUT", &url, &[], data)?;
        let req = Self::apply(self.agent.put(&url), &signed);
        match req.send_bytes(data) {
            Ok(_) => Ok(()),
            Err(e) => Err(StoreError::Http(format!("PUT {object}: {e}"))),
        }
    }

    fn put_if_absent(&self, key: &ObjectPath, data: &[u8]) -> Result<bool> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        // S3 conditional write: `If-None-Match: *` fails the PUT with 412 when
        // the object already exists. A 409 means another conditional write on
        // the same key is in flight — also "lost the race" to the caller.
        let signed = self.sign("PUT", &url, &[("if-none-match", "*")], data)?;
        // `sign` returns only the auth headers it derives (Authorization,
        // x-amz-date, …); `if-none-match` is a request header we sign but must
        // also send ourselves. Its name is in the signature's `SignedHeaders`,
        // so omitting it from the wire fails the signature (a 400), not the CAS.
        let req = Self::apply(self.agent.put(&url), &signed).set("if-none-match", "*");
        match req.send_bytes(data) {
            Ok(_) => Ok(true),
            Err(ureq::Error::Status(412 | 409, _)) => Ok(false),
            Err(e) => Err(StoreError::Http(format!("conditional PUT {object}: {e}"))),
        }
    }

    fn delete(&self, key: &ObjectPath) -> Result<()> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        let signed = self.sign("DELETE", &url, &[], &[])?;
        let req = Self::apply(self.agent.delete(&url), &signed);
        match req.call() {
            // DELETE is idempotent; a missing key is the goal state.
            Ok(_) | Err(ureq::Error::Status(404, _)) => Ok(()),
            Err(e) => Err(StoreError::Http(format!("DELETE {object}: {e}"))),
        }
    }

    fn list(&self, prefix: &ObjectPath) -> Result<Vec<FileRef>> {
        let object_prefix = object_key(&self.prefix, prefix);
        // ListObjectsV2, one level (delimiter=/), under the object prefix.
        let query = format!(
            "list-type=2&prefix={}%2F&delimiter=%2F",
            percent_encode(&object_prefix)
        );
        let url = format!("{}/?{}", self.base, query);
        let signed = self.sign("GET", &url, &[], &[])?;
        let req = Self::apply(self.agent.get(&url), &signed);
        let body = match req.call() {
            Ok(resp) => resp
                .into_string()
                .map_err(|e| StoreError::Http(format!("LIST body: {e}")))?,
            Err(e) => return Err(StoreError::Http(format!("LIST {object_prefix}: {e}"))),
        };

        let parsed: ListBucketResult = quick_xml::de::from_str(&body)
            .map_err(|e| StoreError::Http(format!("LIST parse: {e}")))?;

        Ok(parsed
            .contents
            .into_iter()
            .map(|c| FileRef {
                path: ObjectPath::new(super::key_name(&c.key)),
                size: c.size,
            })
            .collect())
    }

    fn source(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        // S3 presigns the URL: auth rides in the query string, no per-request header.
        Ok(DataFileLocation::Remote {
            url: self.presign_get(key)?,
            auth: None,
        })
    }

    fn sink(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        Ok(DataFileLocation::Remote {
            url: self.presign("PUT", key)?,
            auth: None,
        })
    }
}

impl S3Store {
    /// A time-limited GET URL for `key`, signed in the query string so the
    /// io_uring HTTP reader can range-read it with no auth headers.
    fn presign_get(&self, key: &ObjectPath) -> Result<url::Url> {
        self.presign("GET", key)
    }

    fn presign(&self, method: &str, key: &ObjectPath) -> Result<url::Url> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);

        // Long-lived keys only, so neither a session token nor an expiry.
        // `Static` is the SDK's own name for keys handed over directly rather
        // than resolved by a credentials provider.
        let creds = Credentials::new(&self.access_key, &self.secret_key, None, None, "Static");
        let identity = creds.into();

        let mut settings = SigningSettings::default();
        settings.signature_location = SignatureLocation::QueryParams;
        settings.expires_in = Some(Duration::from_secs(3600));

        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name("s3")
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| StoreError::Http(format!("sigv4 presign params: {e}")))?;

        // A presigned GET signs only `host`; the payload is unsigned so the ring
        // can add a `Range` header the signature doesn't cover.
        let host_header = [("host", self.host.as_str())];
        let signable = SignableRequest::new(
            method,
            &url,
            host_header.iter().copied(),
            SignableBody::UnsignedPayload,
        )
        .map_err(|e| StoreError::Http(format!("sigv4 presign signable: {e}")))?;

        let (instructions, _signature) = sign(signable, &params.into())
            .map_err(|e| StoreError::Http(format!("sigv4 presign: {e}")))?
            .into_parts();

        let mut signed = url::Url::parse(&url)
            .map_err(|e| StoreError::Http(format!("parsing presign url: {e}")))?;
        for (name, value) in instructions.params() {
            signed.query_pairs_mut().append_pair(name, value);
        }
        Ok(signed)
    }
}

/// ListObjectsV2 XML response (only the keys are needed).
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ListBucketResult {
    #[serde(default, rename = "Contents")]
    contents: Vec<Contents>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Contents {
    key: String,
    #[serde(default)]
    size: u64,
}

/// Split an `s3://bucket/prefix` (or `s3a://…`) URI into its bucket and
/// in-bucket prefix (empty when the URI names only a bucket).
fn parse_s3_uri(uri: &str) -> Result<(&str, &str)> {
    let rest = uri
        .strip_prefix("s3://")
        .or_else(|| uri.strip_prefix("s3a://"))
        .ok_or_else(|| StoreError::UnsupportedUri(uri.to_string()))?;
    Ok(rest.split_once('/').unwrap_or((rest, "")))
}

fn env_any(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| std::env::var(k).ok())
}

fn env_req(key: &str) -> Result<String> {
    std::env::var(key)
        .map_err(|_| StoreError::Config(format!("environment variable {key} not set")))
}

/// Percent-encode an S3 object key for a query-string value per RFC 3986
/// (unreserved chars pass through; `/` is encoded since it's a query value).
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
    fn list_response_parses_keys_and_sizes() {
        // A trimmed ListObjectsV2 response: each object carries its byte size,
        // which the remote reader needs to locate a Parquet footer.
        let xml = r#"<?xml version="1.0"?>
            <ListBucketResult>
              <Contents><Key>db/events/a.parquet</Key><Size>123</Size></Contents>
              <Contents><Key>db/events/b.parquet</Key><Size>4096</Size></Contents>
            </ListBucketResult>"#;
        let parsed: ListBucketResult = quick_xml::de::from_str(xml).unwrap();
        let objects: Vec<_> = parsed.contents.iter().map(|c| (&c.key, c.size)).collect();
        assert_eq!(
            objects,
            vec![
                (&"db/events/a.parquet".to_string(), 123),
                (&"db/events/b.parquet".to_string(), 4096),
            ]
        );
    }
}
