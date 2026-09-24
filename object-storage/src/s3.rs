//! S3 (and S3-compatible) backend: blocking HTTP via [`ureq`], with
//! authenticated requests signed through `aws_sigv4::http_request::sign` (a
//! pure function — no runtime). When no credentials are configured, requests
//! are sent anonymously instead, which supports public buckets and S3-compatible
//! endpoints that do not require authentication.
//!
//! Credentials come from one of two places. [`S3Store::with_env_credentials`]
//! reads them from the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//! and `AWS_SESSION_TOKEN` when the keys are an STS session's; optional region
//! from `AWS_REGION`/`AWS_DEFAULT_REGION`; an optional `AWS_ENDPOINT_URL`
//! selects a path-style S3-compatible endpoint like MinIO).
//! [`S3Store::with_credentials`] takes them explicitly, so a metastore can supply
//! a datastore's own key/secret/region/endpoint rather than relying on whatever
//! the process was started with.
//!
//! If neither key is present in the environment, the store uses anonymous
//! access; setting only one key is rejected as an incomplete configuration.
//! When no region is supplied, the store sends an unsigned `HeadBucket` request
//! and takes the region from S3's `x-amz-bucket-region` response header.
//! Either way the resolved parameters are kept on the store and exposed through
//! [`ObjectStore::connection`] so another client can address the same bucket
//! without resolving credentials again.

use super::{
    DataFileLocation, DirectoryListing, FileRef, ListedObject, ObjectPath, ObjectStore,
    ObjectVersion, Result, StoreConnection, StoreError, absolute_object_key, object_key,
    parse_iso8601_millis, percent_encode,
};
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    PayloadChecksumKind, SignableBody, SignableRequest, SignatureLocation, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use std::io::Read;
use std::time::{Duration, SystemTime};

#[derive(Debug)]
pub struct S3Store {
    /// The `s3://bucket/prefix` URI this store was opened with, kept verbatim
    /// as its addressable root (see [`ObjectStore::location_uri`]).
    uri: String,
    /// In-bucket prefix under which this datastore's keys live.
    prefix: String,
    region: String,
    /// What requests are signed with; `None` for anonymous access.
    keys: Option<S3Keys>,
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
#[derive(Clone)]
pub struct S3Credentials {
    /// Optional signing region. When not set, it is discovered with an
    /// unsigned `HeadBucket` request before the store is opened.
    pub region: Option<String>,
    pub keys: S3Keys,
    /// A path-style S3-compatible endpoint (e.g. MinIO). `None` uses AWS
    /// virtual-hosted style.
    pub endpoint: Option<String>,
}

/// The keys a store signs with: a long-lived pair, or an STS session's pair
/// with the token every request signed by it must carry.
#[derive(Clone)]
pub struct S3Keys {
    pub access_key: String,
    pub secret_key: String,
    /// The session token that accompanies an STS session's keys (temporary
    /// credentials from an assumed role, or vended by a catalog). `None` for
    /// long-lived keys.
    pub session_token: Option<String>,
}

/// Redacted: the keys must not leak into a log through a `{:?}` of the store
/// or the connection that holds them.
impl std::fmt::Debug for S3Keys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("S3Keys(redacted)")
    }
}

impl S3Keys {
    /// The credentials the SigV4 signer signs as. `Static` is the SDK's own name
    /// for keys handed over directly rather than resolved by a credentials
    /// provider; a session token rides along as `x-amz-security-token`.
    fn credentials(&self) -> Credentials {
        Credentials::new(
            &self.access_key,
            &self.secret_key,
            self.session_token.clone(),
            None,
            "Static",
        )
    }
}

impl S3Store {
    /// Open `s3://bucket/prefix` with the credentials, region, and endpoint the
    /// environment carries. If neither key is set, use anonymous access. A
    /// partial key pair is rejected rather than silently changing auth modes.
    pub fn with_env_credentials(uri: &str) -> Result<Self> {
        let (bucket, prefix) = parse_s3_uri(uri)?;

        let region = env_any(&["AWS_REGION", "AWS_DEFAULT_REGION"]);
        Self::build(
            uri,
            bucket,
            prefix,
            region,
            env_credentials()?,
            std::env::var("AWS_ENDPOINT_URL").ok(),
        )
    }

    /// Open `s3://bucket/prefix` anonymously against AWS's standard endpoint.
    /// This deliberately does not consult the environment, so a metastore can
    /// preserve its scoped-secret policy while allowing an uncovered public
    /// location to be read.
    pub fn anonymous(uri: &str) -> Result<Self> {
        let (bucket, prefix) = parse_s3_uri(uri)?;
        Self::build(uri, bucket, prefix, None, None, None)
    }

    /// Open anonymously with explicit connection parameters. Useful for an
    /// S3-compatible service whose endpoint is not AWS's standard endpoint.
    pub fn anonymous_with_options(
        uri: &str,
        region: impl Into<String>,
        endpoint: Option<String>,
    ) -> Result<Self> {
        let (bucket, prefix) = parse_s3_uri(uri)?;
        Self::build(uri, bucket, prefix, Some(region.into()), None, endpoint)
    }

    /// Whether this store sends S3 requests without SigV4 authentication.
    pub fn is_anonymous(&self) -> bool {
        self.keys.is_none()
    }

    /// Parse `s3://bucket/prefix` but take credentials/region/endpoint from
    /// `credentials` instead of the environment, the path a metastore uses to
    /// open a datastore with its own configured keys.
    pub fn with_credentials(uri: &str, credentials: S3Credentials) -> Result<Self> {
        let (bucket, prefix) = parse_s3_uri(uri)?;
        Self::build(
            uri,
            bucket,
            prefix,
            credentials.region,
            Some(credentials.keys),
            credentials.endpoint,
        )
    }

    /// Assemble the store from a parsed bucket/prefix and resolved connection
    /// parameters, computing the signing origin and `Host` from the optional
    /// endpoint.
    fn build(
        uri: &str,
        bucket: &str,
        prefix: &str,
        region: Option<String>,
        keys: Option<S3Keys>,
        endpoint: Option<String>,
    ) -> Result<Self> {
        let agent = ureq::AgentBuilder::new().build();
        let region = match region.filter(|region| !region.trim().is_empty()) {
            Some(region) => region,
            None => discover_bucket_region(&agent, bucket, endpoint.as_deref())?,
        };
        let (base, host) = match endpoint.as_deref() {
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
                let host = format!("{bucket}.s3.{region}.amazonaws.com");
                (format!("https://{host}"), host)
            }
        };

        Ok(Self {
            uri: uri.to_string(),
            prefix: prefix.to_string(),
            region,
            keys,
            endpoint,
            base,
            host,
            agent,
        })
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
        let Some(keys) = &self.keys else {
            return Ok(Vec::new());
        };
        let identity = keys.credentials().into();

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

    /// The error for a redirect `ureq` handed back as a success it did not
    /// follow. S3 answers a request addressed to another region's endpoint with
    /// a 301 `PermanentRedirect` that has no `Location` and names the bucket's
    /// region in `x-amz-bucket-region`; taken as a success, its error document
    /// would parse as an empty listing or be read back as an object's bytes.
    fn explain_redirect(&self, verb: &str, response: &ureq::Response) -> StoreError {
        let Some(actual) = response.header("x-amz-bucket-region") else {
            return StoreError::Http(format!(
                "{verb}: S3 redirected the request (HTTP {}) without naming the bucket's region",
                response.status()
            ));
        };
        let (bucket, _) = parse_s3_uri(&self.uri).expect("the store was opened from this URI");
        StoreError::WrongRegion {
            bucket: bucket.to_string(),
            configured: self.region.clone(),
            actual: actual.to_string(),
        }
    }
}

/// Whether `ureq` returned a redirect it did not follow as a success.
fn is_redirect(response: &ureq::Response) -> bool {
    (300..400).contains(&response.status())
}

impl ObjectStore for S3Store {
    fn describe(&self) -> String {
        let auth = if self.is_anonymous() {
            ", anonymous"
        } else {
            ""
        };
        format!("{} (prefix `{}`{auth})", self.base, self.prefix)
    }

    /// A no-op: S3 has no directories. A key is a flat string, and an object at
    /// `prefix/name` exists the moment it is PUT, with no parent to create first.
    fn create_dir(&self, _prefix: &ObjectPath) -> Result<()> {
        Ok(())
    }

    fn location_uri(&self) -> String {
        self.uri.clone()
    }

    fn connection(&self) -> StoreConnection {
        StoreConnection::S3 {
            uri: self.uri.clone(),
            region: self.region.clone(),
            credentials: self.keys.as_ref().map(|keys| S3Credentials {
                region: Some(self.region.clone()),
                keys: keys.clone(),
                endpoint: self.endpoint.clone(),
            }),
            endpoint: self.endpoint.clone(),
        }
    }

    fn get(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>> {
        Ok(self.fetch(key)?.map(|(bytes, _)| bytes))
    }

    fn put(&self, key: &ObjectPath, data: &[u8]) -> Result<()> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        let signed = self.sign("PUT", &url, &[], data)?;
        let req = Self::apply(self.agent.put(&url), &signed);
        match req.send_bytes(data) {
            Ok(response) if is_redirect(&response) => Err(self.explain_redirect("PUT", &response)),
            Ok(_) => Ok(()),
            Err(e) => Err(StoreError::Http(format!("PUT {object}: {e}"))),
        }
    }

    fn update(
        &self,
        key: &ObjectPath,
        apply: &mut dyn FnMut(Option<Vec<u8>>) -> Option<Vec<u8>>,
    ) -> Result<()> {
        super::update_by_version_swap(
            || self.read_versioned(key),
            |data, expected| self.put_if_version(key, data, expected),
            apply,
        )
    }

    fn delete(&self, key: &ObjectPath) -> Result<()> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        let signed = self.sign("DELETE", &url, &[], &[])?;
        let req = Self::apply(self.agent.delete(&url), &signed);
        match req.call() {
            Ok(response) if is_redirect(&response) => {
                Err(self.explain_redirect("DELETE", &response))
            }
            // DELETE is idempotent; a missing key is the goal state.
            Ok(_) | Err(ureq::Error::Status(404, _)) => Ok(()),
            Err(e) => Err(StoreError::Http(format!("DELETE {object}: {e}"))),
        }
    }

    fn list_with_name_prefix(
        &self,
        prefix: &ObjectPath,
        name_prefix: &str,
    ) -> Result<DirectoryListing> {
        let object_prefix = object_key(&self.prefix, prefix);
        let encoded_prefix = if object_prefix.is_empty() {
            percent_encode(name_prefix)
        } else {
            format!(
                "{}%2F{}",
                percent_encode(&object_prefix),
                percent_encode(name_prefix)
            )
        };
        let mut continuation: Option<String> = None;
        let mut objects = Vec::new();
        let mut prefixes = Vec::new();
        loop {
            let mut query = format!("list-type=2&prefix={encoded_prefix}&delimiter=%2F");
            if let Some(token) = &continuation {
                query.push_str("&continuation-token=");
                query.push_str(&percent_encode(token));
            }
            let url = format!("{}/?{}", self.base, query);
            let signed = self.sign("GET", &url, &[], &[])?;
            let req = Self::apply(self.agent.get(&url), &signed);
            let body = match req.call() {
                Ok(response) if is_redirect(&response) => {
                    return Err(self.explain_redirect("LIST", &response));
                }
                Ok(response) => response
                    .into_string()
                    .map_err(|error| StoreError::Http(format!("LIST body: {error}")))?,
                Err(error) => {
                    return Err(StoreError::Http(format!("LIST {object_prefix}: {error}")));
                }
            };
            let parsed: ListBucketResult = quick_xml::de::from_str(&body)
                .map_err(|error| StoreError::Http(format!("LIST parse: {error}")))?;

            objects.extend(
                parsed
                    .contents
                    .into_iter()
                    .map(|contents| {
                        let modified_unix_ms = parse_iso8601_millis(&contents.last_modified)
                            .ok_or_else(|| {
                                StoreError::Http(format!(
                                    "LIST object `{}` has an unparseable LastModified `{}`",
                                    contents.key, contents.last_modified
                                ))
                            })?;
                        Ok(ListedObject {
                            file: FileRef {
                                path: ObjectPath::new(super::key_name(&contents.key)),
                                size: contents.size,
                            },
                            modified_unix_ms,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            prefixes.extend(parsed.common_prefixes.into_iter().filter_map(|common| {
                let relative = super::relative_key(&object_prefix, &common.prefix);
                let relative = relative.trim_end_matches('/');
                (!relative.is_empty()).then(|| ObjectPath::new(relative))
            }));
            if !parsed.is_truncated {
                break;
            }
            continuation = Some(parsed.next_continuation_token.ok_or_else(|| {
                StoreError::Http("LIST response is truncated without a continuation token".into())
            })?);
        }
        Ok(DirectoryListing { objects, prefixes })
    }

    fn absolute_key(&self, key: &ObjectPath) -> Result<ObjectPath> {
        Ok(absolute_object_key(&self.prefix, key))
    }

    fn source(&self, key: &ObjectPath) -> Result<DataFileLocation> {
        // Authenticated S3 presigns the URL, while anonymous S3 returns the
        // same object URL without query parameters. Neither needs a per-request
        // header from the range reader.
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
    /// GET an object, returning its bytes and the `ETag` header when the
    /// service sent one; `None` if the object does not exist.
    fn fetch(&self, key: &ObjectPath) -> Result<Option<(Vec<u8>, Option<String>)>> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        let signed = self.sign("GET", &url, &[], &[])?;
        let req = Self::apply(self.agent.get(&url), &signed);
        match req.call() {
            Ok(response) if is_redirect(&response) => Err(self.explain_redirect("GET", &response)),
            Ok(resp) => {
                let etag = resp.header("etag").map(str::to_string);
                let mut buf = Vec::new();
                resp.into_reader()
                    .read_to_end(&mut buf)
                    .map_err(|source| StoreError::Io {
                        key: key.to_string(),
                        source,
                    })?;
                Ok(Some((buf, etag)))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(StoreError::Http(format!("GET {object}: {e}"))),
        }
    }

    /// The object's bytes with the [`ObjectVersion`] a conditional replace
    /// presents back. An object served without an `ETag` cannot anchor a
    /// compare-and-swap, so that is an error rather than an unguarded write.
    fn read_versioned(&self, key: &ObjectPath) -> Result<Option<(Vec<u8>, ObjectVersion)>> {
        match self.fetch(key)? {
            None => Ok(None),
            Some((bytes, Some(etag))) => Ok(Some((bytes, ObjectVersion(etag)))),
            Some((_, None)) => Err(StoreError::Http(format!(
                "GET {key}: response carries no ETag to condition a replace on"
            ))),
        }
    }

    /// PUT that succeeds only while the stored object is still at `expected`
    /// (`None`: the object must not exist yet). S3 speaks this as `If-Match` /
    /// `If-None-Match: *`; a lost race comes back 412 — or 409 while the
    /// concurrent write is still settling — both mapped to
    /// [`StoreError::VersionConflict`] so the caller retries.
    fn put_if_version(
        &self,
        key: &ObjectPath,
        data: &[u8],
        expected: Option<&ObjectVersion>,
    ) -> Result<()> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        let (header, value) = match expected {
            Some(version) => ("if-match", version.0.as_str()),
            None => ("if-none-match", "*"),
        };
        let signed = self.sign("PUT", &url, &[(header, value)], data)?;
        let req = Self::apply(self.agent.put(&url), &signed).set(header, value);
        match req.send_bytes(data) {
            Ok(response) if is_redirect(&response) => Err(self.explain_redirect("PUT", &response)),
            Ok(_) => Ok(()),
            Err(ureq::Error::Status(412 | 409, _)) => Err(StoreError::VersionConflict {
                key: key.to_string(),
            }),
            Err(e) => Err(StoreError::Http(format!("PUT {object}: {e}"))),
        }
    }

    /// A GET URL for `key`: time-limited and signed in the query string when
    /// credentials exist, bare when this store is anonymous. Either form lets
    /// the io_uring HTTP reader range-read it with no auth headers.
    fn presign_get(&self, key: &ObjectPath) -> Result<url::Url> {
        self.presign("GET", key)
    }

    fn presign(&self, method: &str, key: &ObjectPath) -> Result<url::Url> {
        let object = object_key(&self.prefix, key);
        let url = self.url_for(&object);
        let Some(keys) = &self.keys else {
            return url::Url::parse(&url)
                .map_err(|e| StoreError::Http(format!("parsing anonymous S3 url: {e}")));
        };
        let identity = keys.credentials().into();

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

/// The object keys and immediate child prefixes from ListObjectsV2.
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ListBucketResult {
    #[serde(default, rename = "Contents")]
    contents: Vec<Contents>,
    #[serde(default, rename = "CommonPrefixes")]
    common_prefixes: Vec<CommonPrefix>,
    #[serde(default)]
    is_truncated: bool,
    next_continuation_token: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct CommonPrefix {
    prefix: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Contents {
    key: String,
    #[serde(default)]
    size: u64,
    /// ISO-8601 `LastModified` (e.g. `2009-10-12T17:50:30.000Z`), parsed by
    /// [`parse_iso8601_millis`] for vacuum's orphan sweep.
    #[serde(default)]
    last_modified: String,
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

/// Ask the bucket which region owns it. AWS returns `x-amz-bucket-region` from
/// `HeadBucket` even when the unsigned request is denied, so a private bucket
/// can be located before credentials are signed for its region. A custom
/// endpoint gets the same path-style probe; compatible services that do not
/// implement the header need an explicit region instead.
fn discover_bucket_region(
    agent: &ureq::Agent,
    bucket: &str,
    endpoint: Option<&str>,
) -> Result<String> {
    let url = match endpoint {
        Some(endpoint) => format!("{}/{bucket}", endpoint.trim_end_matches('/')),
        None => format!("https://{bucket}.s3.amazonaws.com"),
    };
    let response = match agent.head(&url).call() {
        Ok(response) | Err(ureq::Error::Status(_, response)) => response,
        Err(error) => {
            return Err(StoreError::Http(format!(
                "discovering region for S3 bucket `{bucket}` with HEAD {url}: {error}"
            )));
        }
    };
    let status = response.status();
    response
        .header("x-amz-bucket-region")
        .filter(|region| !region.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            StoreError::Config(format!(
                "HEAD {url} returned {status} without x-amz-bucket-region; set the S3 region explicitly"
            ))
        })
}

fn env_any(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| std::env::var(k).ok())
}

/// The keys the environment carries, or `None` for anonymous access. A
/// session token without keys is meaningless and is ignored with them absent.
fn env_credentials() -> Result<Option<S3Keys>> {
    match (
        std::env::var("AWS_ACCESS_KEY_ID").ok(),
        std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
    ) {
        (Some(access_key), Some(secret_key)) => Ok(Some(S3Keys {
            access_key,
            secret_key,
            session_token: std::env::var("AWS_SESSION_TOKEN").ok(),
        })),
        (None, None) => Ok(None),
        _ => Err(StoreError::Config(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    /// Serve one response to the region-discovery HEAD request.
    fn region_server(region: Option<&str>) -> (String, JoinHandle<String>) {
        let region_header = region
            .map(|region| format!("x-amz-bucket-region: {region}\r\n"))
            .unwrap_or_default();
        serve_once(format!(
            "HTTP/1.1 403 Forbidden\r\n{region_header}Content-Length: 0\r\nConnection: close\r\n\r\n"
        ))
    }

    /// Answer the first request with `response`, handing back the request text.
    fn serve_once(response: String) -> (String, JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 1024];
            let read = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..read]).into_owned();
            stream.write_all(response.as_bytes()).unwrap();
            request
        });
        (endpoint, handle)
    }

    #[test]
    fn a_listing_redirected_to_the_buckets_region_names_both_regions() {
        let (endpoint, request) = serve_once(
            "HTTP/1.1 301 Moved Permanently\r\nx-amz-bucket-region: us-east-1\r\n\
             Content-Length: 0\r\nConnection: close\r\n\r\n"
                .to_string(),
        );
        let store =
            S3Store::anonymous_with_options("s3://moved-bucket/root", "eu-west-1", Some(endpoint))
                .unwrap();

        let error = store
            .list_with_name_prefix(&ObjectPath::default(), "part")
            .unwrap_err();

        assert!(
            matches!(
                &error,
                StoreError::WrongRegion { bucket, configured, actual }
                    if bucket == "moved-bucket" && configured == "eu-west-1" && actual == "us-east-1"
            ),
            "{error}"
        );
        assert!(
            request
                .join()
                .unwrap()
                .starts_with("GET /moved-bucket/?list-type=2")
        );
    }

    #[test]
    fn anonymous_requests_have_no_signature() {
        let store = S3Store::anonymous_with_options(
            "s3://public-bucket/root",
            "us-east-1",
            Some("http://objects.example".to_string()),
        )
        .unwrap();

        assert!(
            store
                .sign("GET", "http://objects.example", &[], &[])
                .unwrap()
                .is_empty()
        );
        let DataFileLocation::Remote { url, auth } =
            store.source(&ObjectPath::new("part.parquet")).unwrap()
        else {
            panic!("S3 source must be remote");
        };
        assert_eq!(
            url.as_str(),
            "http://objects.example/public-bucket/root/part.parquet"
        );
        assert!(url.query().is_none());
        assert!(auth.is_none());
        assert!(matches!(
            store.connection(),
            StoreConnection::S3 {
                credentials: None,
                ..
            }
        ));
    }

    #[test]
    fn a_missing_region_is_discovered_from_a_denied_head_bucket_request() {
        let (endpoint, request) = region_server(Some("eu-west-2"));

        let store = S3Store::with_credentials(
            "s3://private-bucket/root",
            S3Credentials {
                region: None,
                keys: S3Keys {
                    access_key: "access".to_string(),
                    secret_key: "secret".to_string(),
                    session_token: None,
                },
                endpoint: Some(endpoint),
            },
        )
        .unwrap();

        assert!(matches!(
            store.connection(),
            StoreConnection::S3 { region, .. } if region == "eu-west-2"
        ));
        let request = request.join().unwrap();
        assert!(request.starts_with("HEAD /private-bucket "));
        assert!(!request.to_ascii_lowercase().contains("authorization:"));
    }

    #[test]
    fn missing_region_header_asks_for_an_explicit_region() {
        let (endpoint, request) = region_server(None);

        let error = discover_bucket_region(
            &ureq::AgentBuilder::new().build(),
            "compatible-bucket",
            Some(&endpoint),
        )
        .unwrap_err();

        assert!(error.to_string().contains("set the S3 region explicitly"));
        request.join().unwrap();
    }

    #[test]
    fn configured_credentials_still_sign_requests() {
        let store = S3Store::with_credentials(
            "s3://private-bucket/root",
            S3Credentials {
                region: Some("us-east-1".to_string()),
                keys: S3Keys {
                    access_key: "access".to_string(),
                    secret_key: "secret".to_string(),
                    session_token: None,
                },
                endpoint: Some("http://objects.example".to_string()),
            },
        )
        .unwrap();

        let headers = store
            .sign(
                "GET",
                "http://objects.example/private-bucket/root/part.parquet",
                &[],
                &[],
            )
            .unwrap();
        assert!(
            headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        );
        let DataFileLocation::Remote { url, .. } =
            store.source(&ObjectPath::new("part.parquet")).unwrap()
        else {
            panic!("S3 source must be remote");
        };
        assert!(url.query().is_some());
        assert!(matches!(
            store.connection(),
            StoreConnection::S3 {
                credentials: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn session_credentials_carry_their_token_in_signatures_and_presigned_urls() {
        let store = S3Store::with_credentials(
            "s3://private-bucket/root",
            S3Credentials {
                region: Some("us-east-1".to_string()),
                keys: S3Keys {
                    access_key: "access".to_string(),
                    secret_key: "secret".to_string(),
                    session_token: Some("session-token".to_string()),
                },
                endpoint: Some("http://objects.example".to_string()),
            },
        )
        .unwrap();

        let headers = store
            .sign(
                "GET",
                "http://objects.example/private-bucket/root/part.parquet",
                &[],
                &[],
            )
            .unwrap();
        let DataFileLocation::Remote { url, .. } =
            store.source(&ObjectPath::new("part.parquet")).unwrap()
        else {
            panic!("S3 source must be remote");
        };

        assert!(headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("x-amz-security-token") && value == "session-token"
        }));
        assert!(
            url.query_pairs()
                .any(|(name, value)| name == "X-Amz-Security-Token" && value == "session-token")
        );
    }

    #[test]
    fn list_response_parses_keys_and_sizes() {
        // A trimmed ListObjectsV2 response: each object carries its byte size,
        // which the remote reader needs to locate a Parquet footer.
        let xml = r#"<?xml version="1.0"?>
            <ListBucketResult>
              <Contents><Key>db/events/a.parquet</Key><Size>123</Size></Contents>
              <Contents><Key>db/events/b.parquet</Key><Size>4096</Size></Contents>
              <CommonPrefixes><Prefix>db/events/2026/</Prefix></CommonPrefixes>
              <IsTruncated>true</IsTruncated>
              <NextContinuationToken>next-page</NextContinuationToken>
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
        assert_eq!(parsed.common_prefixes[0].prefix, "db/events/2026/");
        assert!(parsed.is_truncated);
        assert_eq!(parsed.next_continuation_token.as_deref(), Some("next-page"));
    }
}
