//! An in-process Google Cloud Storage emulator: the slice of the GCS XML API
//! the datastore speaks (object GET whole or by range, PUT plain and
//! conditional, DELETE, HEAD, and a one-level bucket listing), served over
//! plain HTTP on loopback.
//!
//! It exists because a GCS backend cannot be exercised against MinIO the way
//! the S3 one is: the two differ exactly where it matters, in how a request is
//! authenticated and how a conditional create is expressed. So the emulator
//! **requires** the `Authorization: Bearer` header on every request and honours
//! `x-goog-if-generation-match: 0`, the CAS every table commit rides on.
//!
//! Both clients that address a GCS location run against it: the datastore's own
//! blocking [`GcsStore`](crate::store::GcsStore) and the async object-store
//! client Delta Kernel reads a `_delta_log` with, which is why the responses
//! carry the `ETag`/`Last-Modified` metadata that client insists on.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::SystemTime;

/// A running emulator: the origin to point a store at, and the bearer token it
/// accepts. The listener thread is detached and lives for the process.
pub struct GcsEmulator {
    endpoint: String,
    token: String,
}

impl GcsEmulator {
    /// Bind an ephemeral loopback port and serve `bucket`, accepting `token` as
    /// the bearer credential.
    pub fn start(bucket: &str, token: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let bucket = Bucket {
            name: bucket.to_string(),
            token: token.to_string(),
            objects: Mutex::new(BTreeMap::new()),
        };
        let bucket = Arc::new(bucket);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let bucket = bucket.clone();
                thread::spawn(move || serve_connection(stream, &bucket));
            }
        });
        Self {
            endpoint: format!("http://127.0.0.1:{port}"),
            token: token.to_string(),
        }
    }

    /// The origin a store addresses this emulator at.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The bearer token it accepts.
    pub fn token(&self) -> &str {
        &self.token
    }
}

/// The one bucket an emulator serves, and everything in it.
struct Bucket {
    name: String,
    token: String,
    objects: Mutex<BTreeMap<String, StoredObject>>,
}

#[derive(Clone)]
struct StoredObject {
    bytes: Vec<u8>,
    modified: SystemTime,
    /// GCS's per-object version counter, bumped on every write. Only its
    /// "0 means no live object" role is used, by the conditional PUT.
    generation: u64,
}

/// Serve requests off one connection until the peer closes it: the clients
/// pool connections, so a connection outlives its first request.
fn serve_connection(mut stream: TcpStream, bucket: &Bucket) {
    let mut buffered = Vec::new();
    while let Some(request) = read_request(&mut stream, &mut buffered) {
        let response = route(bucket, &request);
        if stream.write_all(&response).is_err() {
            return;
        }
    }
}

struct Request {
    method: String,
    /// Origin-form target: path plus any query string.
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// The path and the raw query string, split at the `?`.
    fn path_and_query(&self) -> (&str, &str) {
        self.target.split_once('?').unwrap_or((&self.target, ""))
    }

    /// A query parameter's percent-decoded value.
    fn query_param(&self, name: &str) -> Option<String> {
        let (_, query) = self.path_and_query();
        query
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .find(|(key, _)| *key == name)
            .map(|(_, value)| percent_decode(value))
    }
}

/// Read one request (head, then any `Content-Length` body) off `stream`,
/// carrying leftover bytes between calls in `buffered`. `None` once the peer
/// stops sending.
fn read_request(stream: &mut TcpStream, buffered: &mut Vec<u8>) -> Option<Request> {
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(at) = find(buffered, b"\r\n\r\n") {
            break at + 4;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return None,
            Ok(read) => buffered.extend_from_slice(&chunk[..read]),
        }
    };
    let head = String::from_utf8_lossy(&buffered[..head_end]).to_string();
    buffered.drain(..head_end);

    let mut lines = head.lines();
    let mut request_line = lines.next()?.split_whitespace();
    let method = request_line.next()?.to_string();
    let target = request_line.next()?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();

    let length: usize = headers
        .iter()
        .find(|(name, _)| name == "content-length")
        .and_then(|(_, value)| value.parse().ok())
        .unwrap_or(0);
    while buffered.len() < length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return None,
            Ok(read) => buffered.extend_from_slice(&chunk[..read]),
        }
    }
    let body = buffered.drain(..length).collect();

    Some(Request {
        method,
        target,
        headers,
        body,
    })
}

/// Dispatch a request to its handler, after checking the bearer token and that
/// it addresses the bucket this emulator serves.
fn route(bucket: &Bucket, request: &Request) -> Vec<u8> {
    if request.header("authorization") != Some(&format!("Bearer {}", bucket.token)) {
        return response(401, "Unauthorized", &[], b"");
    }
    // Decoded whole: a client is free to escape any character of the path,
    // bucket name included.
    let (path, _) = request.path_and_query();
    let path = percent_decode(path);
    let Some(rest) = path.strip_prefix(&format!("/{}", bucket.name)) else {
        return response(404, "Not Found", &[], b"");
    };
    let object = rest.trim_start_matches('/').to_string();

    match (request.method.as_str(), object.is_empty()) {
        ("GET", true) => list(bucket, request),
        ("GET", false) => get(bucket, &object, request),
        ("HEAD", false) => head(bucket, &object),
        ("PUT", false) => put(bucket, &object, request),
        ("DELETE", false) => delete(bucket, &object),
        _ => response(405, "Method Not Allowed", &[], b""),
    }
}

fn get(bucket: &Bucket, name: &str, request: &Request) -> Vec<u8> {
    let Some(object) = bucket.objects.lock().unwrap().get(name).cloned() else {
        return response(404, "Not Found", &[], b"");
    };
    let total = object.bytes.len();
    let mut headers = object_headers(&object);
    let (status, reason, body) = match request.header("range").and_then(parse_range) {
        Some((start, end)) => {
            let end = end.unwrap_or(total - 1).min(total - 1);
            headers.push((
                "Content-Range".to_string(),
                format!("bytes {start}-{end}/{total}"),
            ));
            (206, "Partial Content", &object.bytes[start..=end])
        }
        None => (200, "OK", &object.bytes[..]),
    };
    response(status, reason, &headers, body)
}

fn head(bucket: &Bucket, name: &str) -> Vec<u8> {
    match bucket.objects.lock().unwrap().get(name) {
        // A HEAD answers with the headers of the GET it stands in for, the
        // object's length included, and no body.
        Some(object) => {
            response_of_length(200, "OK", &object_headers(object), object.bytes.len(), b"")
        }
        None => response(404, "Not Found", &[], b""),
    }
}

fn put(bucket: &Bucket, name: &str, request: &Request) -> Vec<u8> {
    let mut objects = bucket.objects.lock().unwrap();
    let existing = objects.get(name);
    // `x-goog-if-generation-match: 0` writes only when no live object exists.
    if request.header("x-goog-if-generation-match") == Some("0") && existing.is_some() {
        return response(412, "Precondition Failed", &[], b"");
    }
    let object = StoredObject {
        bytes: request.body.clone(),
        modified: SystemTime::now(),
        generation: existing.map_or(1, |object| object.generation + 1),
    };
    let headers = object_headers(&object);
    objects.insert(name.to_string(), object);
    response(200, "OK", &headers, b"")
}

fn delete(bucket: &Bucket, name: &str) -> Vec<u8> {
    match bucket.objects.lock().unwrap().remove(name) {
        Some(_) => response(204, "No Content", &[], b""),
        None => response(404, "Not Found", &[], b""),
    }
}

/// A `ListBucketResult` for the objects under `prefix`, with anything below a
/// `delimiter` boundary rolled up into `CommonPrefixes` instead.
fn list(bucket: &Bucket, request: &Request) -> Vec<u8> {
    let prefix = request.query_param("prefix").unwrap_or_default();
    let delimiter = request.query_param("delimiter");
    let start_after = request.query_param("start-after").unwrap_or_default();
    let max_keys: usize = request
        .query_param("max-keys")
        .and_then(|value| value.parse().ok())
        .unwrap_or(usize::MAX);

    let mut contents = String::new();
    let mut common_prefixes: Vec<String> = Vec::new();
    let objects = bucket.objects.lock().unwrap();
    for (name, object) in objects.iter() {
        if !name.starts_with(&prefix) || name.as_str() <= start_after.as_str() {
            continue;
        }
        let rolled_up = delimiter.as_deref().and_then(|delimiter| {
            name[prefix.len()..]
                .find(delimiter)
                .map(|at| at + prefix.len())
        });
        if let Some(at) = rolled_up {
            let rolled_up = format!("{}{}", &name[..at], delimiter.as_deref().unwrap());
            if !common_prefixes.contains(&rolled_up) {
                common_prefixes.push(rolled_up);
            }
            continue;
        }
        if contents.matches("<Contents>").count() >= max_keys {
            break;
        }
        contents.push_str(&format!(
            "<Contents><Key>{name}</Key><Size>{}</Size><LastModified>{}</LastModified>\
             <ETag>{}</ETag></Contents>",
            object.bytes.len(),
            rfc3339(object.modified),
            etag(object),
        ));
    }
    let prefixes: String = common_prefixes
        .iter()
        .map(|prefix| format!("<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>"))
        .collect();
    let body = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <ListBucketResult><IsTruncated>false</IsTruncated>{contents}{prefixes}</ListBucketResult>"
    );
    response(
        200,
        "OK",
        &[("Content-Type".to_string(), "application/xml".to_string())],
        body.as_bytes(),
    )
}

/// The per-object metadata every object response carries. The async client used
/// for the `_delta_log` rejects a response missing any of it.
fn object_headers(object: &StoredObject) -> Vec<(String, String)> {
    vec![
        ("ETag".to_string(), etag(object)),
        ("Last-Modified".to_string(), rfc2822(object.modified)),
        (
            "x-goog-generation".to_string(),
            object.generation.to_string(),
        ),
        ("Accept-Ranges".to_string(), "bytes".to_string()),
    ]
}

/// An object's entity tag: its generation is exactly the "changed since you
/// last looked" counter an ETag stands for.
fn etag(object: &StoredObject) -> String {
    format!("\"{}\"", object.generation)
}

fn rfc2822(at: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(at).to_rfc2822()
}

fn rfc3339(at: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(at).to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Serialize a response whose `Content-Length` is the body it carries.
fn response(status: u16, reason: &str, headers: &[(String, String)], body: &[u8]) -> Vec<u8> {
    response_of_length(status, reason, headers, body.len(), body)
}

/// Serialize a response announcing `content_length`, which a HEAD states
/// without sending the bytes. `Connection: keep-alive` because the clients pool
/// connections across requests.
fn response_of_length(
    status: u16,
    reason: &str,
    headers: &[(String, String)],
    content_length: usize,
    body: &[u8],
) -> Vec<u8> {
    let headers: String = headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect();
    let mut out = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         {headers}\
         Connection: keep-alive\r\n\
         Content-Length: {content_length}\r\n\r\n"
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

/// Parse a `Range: bytes=start-[end]` header. An open-ended range leaves the
/// end to the caller, which clamps it to the object.
fn parse_range(header: &str) -> Option<(usize, Option<usize>)> {
    let (start, end) = header.trim().strip_prefix("bytes=")?.split_once('-')?;
    Some((start.parse().ok()?, end.parse().ok()))
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
