//! Object-store request/response **policy** over the sans-IO HTTP/1.1 helpers
//! ([`super::http1`]).
//!
//! `http1` parses a response head and tracks an identity body; this layer adds
//! the policy: build the request (a `Range` GET, or a PUT/POST upload), and
//! [`parse_response`] checks a response head against an accepted status range and
//! (for a fixed-slot read) a body-size bound. One function serves both reads and
//! uploads on both platform backends, so the policy lives in one socket-free,
//! unit-testable place.

use super::http1;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtoError {
    #[error("{0}")]
    Http1(#[from] http1::Http1Error),
    #[error("unexpected http status {status}: {snippet}")]
    UnexpectedStatus { status: u16, snippet: String },
    #[error("response missing Content-Length")]
    MissingContentLength,
    #[error("response body ({got} bytes) larger than the {want} requested")]
    BodyTooLarge { got: u64, want: usize },
}

/// Build an origin-form HTTP/1.1 `Range` GET for `[offset, offset + len)`.
///
/// `Connection: keep-alive` so the connection can be returned to the pool, and
/// `Accept-Encoding: identity` so the body length matches the requested range
/// exactly (no transfer/content encoding to undo). `auth`, when present, is the
/// `Authorization` header value (e.g. `Bearer <token>`) for a store that
/// authenticates each request rather than signing the URL.
pub fn build_range_get(
    host: &str,
    target: &str,
    offset: u64,
    len: usize,
    auth: Option<&str>,
) -> Vec<u8> {
    let end = offset + len as u64 - 1;
    let auth = auth.map_or(String::new(), |a| format!("Authorization: {a}\r\n"));
    format!(
        "GET {target} HTTP/1.1\r\n\
         Host: {host}\r\n\
         Range: bytes={offset}-{end}\r\n\
         {auth}\
         Accept-Encoding: identity\r\n\
         Connection: keep-alive\r\n\
         User-Agent: pivotdb-dispatch/0.1\r\n\
         \r\n"
    )
    .into_bytes()
}

/// Build an origin-form HTTP/1.1 upload request (`PUT`/`POST`) carrying a
/// `content_length`-byte body.
///
/// `Connection: keep-alive` so a connection can be returned to the pool and reused
/// (by a later upload or read to the same host), exactly like a range GET — the
/// caller drains the response body first. `content_type`, when the backend
/// requires one (GCS wants `application/octet-stream`), and `auth` (a `Bearer`
/// token, when the URL is not presigned) are emitted only when present.
pub fn build_put(
    method: &str,
    host: &str,
    target: &str,
    content_length: usize,
    content_type: Option<&str>,
    auth: Option<&str>,
) -> Vec<u8> {
    let content_type = content_type.map_or(String::new(), |c| format!("Content-Type: {c}\r\n"));
    let auth = auth.map_or(String::new(), |a| format!("Authorization: {a}\r\n"));
    format!(
        "{method} {target} HTTP/1.1\r\n\
         Host: {host}\r\n\
         Content-Length: {content_length}\r\n\
         {content_type}\
         {auth}\
         Connection: keep-alive\r\n\
         User-Agent: pivotdb-dispatch/0.1\r\n\
         \r\n"
    )
    .into_bytes()
}

/// Outcome of parsing a (possibly incomplete) response head against a policy.
pub enum RespParse {
    /// Need more bytes before the head is complete.
    Incomplete,
    /// The head parsed with an accepted status.
    Ready {
        /// Length of the header section, including the terminating `\r\n\r\n`; the
        /// body begins at this offset in the parsed buffer.
        head_len: usize,
        /// `Content-Length`, if the header was present.
        content_length: Option<u64>,
    },
}

/// Parse and policy-check a response head from `buf` — shared by both platform
/// backends and by reads and uploads.
///
/// A status outside `status_range` (inclusive) is rejected with a snippet of the
/// captured head/body, so a failed request surfaces the server's explanation.
/// `max_body`, when `Some` (a read into a fixed cache slot), additionally requires
/// a `Content-Length` present and no larger than it — range responses are
/// identity-framed and must fit the slot. `None` (an upload) leaves the response
/// body length unconstrained. Returns [`RespParse::Incomplete`] until the head has
/// fully arrived.
pub fn parse_response(
    buf: &[u8],
    status_range: (u16, u16),
    max_body: Option<u64>,
) -> Result<RespParse, ProtoError> {
    let head = match http1::parse_response_head(buf)? {
        http1::HeadStatus::Incomplete => return Ok(RespParse::Incomplete),
        http1::HeadStatus::Complete(head) => head,
    };

    let (lo, hi) = status_range;
    if head.status < lo || head.status > hi {
        let snippet = String::from_utf8_lossy(&buf[..buf.len().min(800)]).into_owned();
        return Err(ProtoError::UnexpectedStatus {
            status: head.status,
            snippet,
        });
    }

    if let Some(max) = max_body {
        let content_length = head
            .content_length
            .ok_or(ProtoError::MissingContentLength)?;
        if content_length > max {
            return Err(ProtoError::BodyTooLarge {
                got: content_length,
                want: max as usize,
            });
        }
    }

    Ok(RespParse::Ready {
        head_len: head.head_len,
        content_length: head.content_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_get_has_expected_request_line_and_headers() {
        let req = build_range_get("example.com", "/data.parquet", 4096, 8192, None);
        let text = String::from_utf8(req).unwrap();
        assert!(text.starts_with("GET /data.parquet HTTP/1.1\r\n"));
        assert!(text.contains("Host: example.com\r\n"));
        // [4096, 4096+8192) -> bytes=4096-12287
        assert!(text.contains("Range: bytes=4096-12287\r\n"));
        assert!(!text.contains("Authorization"));
        assert!(text.ends_with("\r\n\r\n"));
    }

    #[test]
    fn range_get_emits_authorization_when_present() {
        let req = build_range_get("example.com", "/data.parquet", 0, 1, Some("Bearer tok"));
        let text = String::from_utf8(req).unwrap();
        assert!(text.contains("Authorization: Bearer tok\r\n"));
    }

    #[test]
    fn parses_complete_head_within_policy() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\nContent-Range: bytes 0-9/100\r\n\r\nXXXXXXXXXX";
        match parse_response(raw, (206, 206), Some(4096)).unwrap() {
            RespParse::Ready {
                head_len,
                content_length,
            } => {
                assert_eq!(content_length, Some(10));
                // head_len points at the first body byte.
                assert_eq!(&raw[head_len..], b"XXXXXXXXXX");
            }
            RespParse::Incomplete => panic!("expected complete head"),
        }
    }

    #[test]
    fn reports_incomplete_when_terminator_missing() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\n";
        assert!(matches!(
            parse_response(raw, (206, 206), Some(4096)).unwrap(),
            RespParse::Incomplete
        ));
    }

    #[test]
    fn rejects_status_outside_the_range() {
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Length: 3\r\n\r\nno!";
        assert!(matches!(
            parse_response(raw, (206, 206), Some(4096)),
            Err(ProtoError::UnexpectedStatus { status: 404, .. })
        ));
    }

    #[test]
    fn rejects_body_larger_than_the_slot() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 9000\r\n\r\n";
        assert!(matches!(
            parse_response(raw, (206, 206), Some(4096)),
            Err(ProtoError::BodyTooLarge { .. })
        ));
    }

    #[test]
    fn upload_accepts_any_2xx_with_no_content_length_bound() {
        let raw = b"HTTP/1.1 201 Created\r\n\r\n";
        assert!(matches!(
            parse_response(raw, (200, 299), None).unwrap(),
            RespParse::Ready {
                content_length: None,
                ..
            }
        ));
    }
}
