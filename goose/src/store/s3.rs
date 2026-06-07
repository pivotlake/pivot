//! S3 (and S3-compatible) backend: blocking HTTP via [`ureq`], requests signed
//! with SigV4 via `aws_sigv4::http_request::sign` (a pure function — no runtime).
//!
//! Credentials are read from the environment (`AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`); region from
//! `AWS_REGION`/`AWS_DEFAULT_REGION`. An optional `AWS_ENDPOINT_URL` selects a
//! path-style S3-compatible endpoint (MinIO, GCS XML interop) for tests.

use super::{ObjectStore, PutOutcome, Result, StoreError, join_prefix};
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    PayloadChecksumKind, SignableBody, SignableRequest, SignatureLocation, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use std::io::Read;
use std::time::{Duration, SystemTime};

#[derive(Debug)]
pub struct S3Store {
    bucket: String,
    /// In-bucket prefix under which this catalog's keys live.
    prefix: String,
    region: String,
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
    /// Base origin, e.g. `https://bucket.s3.us-east-1.amazonaws.com` (virtual
    /// hosted) or `http://localhost:9000/bucket` (path-style endpoint override).
    base: String,
    /// `Host` header value for signing.
    host: String,
    agent: ureq::Agent,
}

impl S3Store {
    /// Parse `s3://bucket/prefix` and resolve credentials/region/endpoint from
    /// the environment.
    pub fn from_uri(uri: &str) -> Result<Self> {
        let rest = uri
            .strip_prefix("s3://")
            .or_else(|| uri.strip_prefix("s3a://"))
            .ok_or_else(|| StoreError::UnsupportedUri(uri.to_string()))?;
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));

        let region = env_any(&["AWS_REGION", "AWS_DEFAULT_REGION"])
            .unwrap_or_else(|| "us-east-1".to_string());
        let access_key = env_req("AWS_ACCESS_KEY_ID")?;
        let secret_key = env_req("AWS_SECRET_ACCESS_KEY")?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

        let (base, host) = match std::env::var("AWS_ENDPOINT_URL").ok() {
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
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            region,
            access_key,
            secret_key,
            session_token,
            base,
            host,
            agent: ureq::AgentBuilder::new().build(),
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
        let creds = Credentials::new(
            &self.access_key,
            &self.secret_key,
            self.session_token.clone(),
            None,
            "goose-env",
        );
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
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let object = join_prefix(&self.prefix, key);
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

    fn put_if_absent(&self, key: &str, data: &[u8]) -> Result<PutOutcome> {
        let object = join_prefix(&self.prefix, key);
        let url = self.url_for(&object);
        // `If-None-Match: *` makes S3 reject the write if the object exists —
        // the conditional create that backs a catalog CAS commit.
        let extra = [("if-none-match", "*")];
        let signed = self.sign("PUT", &url, &extra, data)?;
        let req = Self::apply(self.agent.put(&url), &signed).set("If-None-Match", "*");
        match req.send_bytes(data) {
            Ok(_) => Ok(PutOutcome::Created),
            // 412 Precondition Failed (and some endpoints 409) => lost the race.
            Err(ureq::Error::Status(412, _)) | Err(ureq::Error::Status(409, _)) => {
                Ok(PutOutcome::AlreadyExists)
            }
            Err(e) => Err(StoreError::Http(format!("PUT {object}: {e}"))),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let object_prefix = join_prefix(&self.prefix, prefix);
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

        // Strip the in-bucket prefix so callers get catalog-relative keys.
        let strip = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("{}/", self.prefix.trim_matches('/'))
        };
        Ok(parsed
            .contents
            .into_iter()
            .map(|c| c.key.strip_prefix(&strip).unwrap_or(&c.key).to_string())
            .collect())
    }

    fn presign_get(&self, key: &str) -> Result<url::Url> {
        let object = join_prefix(&self.prefix, key);
        let url = self.url_for(&object);

        let creds = Credentials::new(
            &self.access_key,
            &self.secret_key,
            self.session_token.clone(),
            None,
            "goose-env",
        );
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
            "GET",
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

    fn describe(&self) -> String {
        format!("s3://{}/{}", self.bucket, self.prefix)
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
