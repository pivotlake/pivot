//! Object-store range-read **policy** over the sans-IO HTTP/1.1 helpers
//! ([`super::http1`]).
//!
//! `http1` parses a response head and tracks an identity body; this layer adds
//! the range-read policy on top: build the `Range` GET, and accept only a `206
//! Partial Content` response whose `Content-Length` body fits the requested slot.
//! Keeping the policy here (rather than in `http1`) means both platform backends
//! share identical, socket-free, unit-testable logic.

use super::http1;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtoError {
    #[error("{0}")]
    Http1(#[from] http1::Http1Error),
    #[error("unexpected http status {0} (expected 206 Partial Content)")]
    UnexpectedStatus(u16),
    #[error("range response missing Content-Length")]
    MissingContentLength,
    #[error("range response body ({got} bytes) larger than requested ({want} bytes)")]
    BodyTooLarge { got: u64, want: usize },
    #[error("unexpected upload status {0} (expected a 2xx response)")]
    UnexpectedUploadStatus(u16),
}

/// Build a whole-object upload request. Always a `PUT` (the only upload method
/// pivot uses).
pub fn build_upload(host: &str, target: &str, len: usize, auth: Option<&str>) -> Vec<u8> {
    let auth = auth.map_or(String::new(), |a| format!("Authorization: {a}\r\n"));
    format!(
        "PUT {target} HTTP/1.1\r\n\
         Host: {host}\r\n\
         {auth}\
         Content-Type: application/octet-stream\r\n\
         Content-Length: {len}\r\n\
         Connection: keep-alive\r\n\
         User-Agent: pivotdb-dispatch/0.1\r\n\
         \r\n"
    )
    .into_bytes()
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

/// A parsed (and validated `206`) response head.
#[derive(Debug, Clone, Copy)]
pub struct ResponseHead {
    /// Length of the header section, including the terminating `\r\n\r\n`. The
    /// body begins at this offset in the buffer that was parsed.
    pub head_len: usize,
    /// Body length from `Content-Length`.
    pub content_length: u64,
    /// Whether the connection can go back to the keep-alive pool once
    /// `content_length` body bytes are drained. False when the response body's
    /// framing is unknown (no `Content-Length`, so possibly chunked or
    /// close-delimited): pooling it would leave unread bytes that corrupt the
    /// next response read on that connection.
    pub reuse_connection: bool,
}

/// Outcome of trying to parse a (possibly incomplete) response head.
pub enum HeadParse {
    /// Need more bytes before the head is complete.
    Incomplete,
    Complete(ResponseHead),
}

/// Try to parse the response head from `buf`. Returns [`HeadParse::Incomplete`]
/// if the header section hasn't fully arrived yet.
///
/// Validates that the status is `206 Partial Content` and that a `Content-Length`
/// is present and no larger than `requested_len` — range responses are never
/// chunked and must fit the destination slot region.
pub fn parse_get_response_head(buf: &[u8], requested_len: usize) -> Result<HeadParse, ProtoError> {
    let head = match http1::parse_response_head(buf)? {
        http1::HeadStatus::Incomplete => return Ok(HeadParse::Incomplete),
        http1::HeadStatus::Complete(head) => head,
    };

    if head.status != 206 {
        return Err(ProtoError::UnexpectedStatus(head.status));
    }

    // A range response is always identity-framed (we send `Accept-Encoding:
    // identity` and don't request multipart), so it must carry a Content-Length
    // — the body size we land in the slot.
    let content_length = head
        .content_length
        .ok_or(ProtoError::MissingContentLength)?;

    if content_length > requested_len as u64 {
        return Err(ProtoError::BodyTooLarge {
            got: content_length,
            want: requested_len,
        });
    }

    Ok(HeadParse::Complete(ResponseHead {
        head_len: head.head_len,
        content_length,
        reuse_connection: true,
    }))
}

/// Parse a successful upload response. Upload APIs commonly return either an
/// empty body or a small JSON description; a `Content-Length`-framed body is
/// drained before pooling the connection. A 2xx response *without* a
/// `Content-Length` (e.g. a chunked body) still completes the upload, but its
/// body cannot be tracked by the identity decoder, so the head is marked
/// non-reusable and the transport closes the connection instead of pooling it.
pub fn parse_upload_response_head(buf: &[u8]) -> Result<HeadParse, ProtoError> {
    let head = match http1::parse_response_head(buf)? {
        http1::HeadStatus::Incomplete => return Ok(HeadParse::Incomplete),
        http1::HeadStatus::Complete(head) => head,
    };
    if !(200..300).contains(&head.status) {
        return Err(ProtoError::UnexpectedUploadStatus(head.status));
    }
    Ok(HeadParse::Complete(ResponseHead {
        head_len: head.head_len,
        content_length: head.content_length.unwrap_or(0),
        reuse_connection: head.content_length.is_some(),
    }))
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
    fn parses_complete_206_head() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\nContent-Range: bytes 0-9/100\r\n\r\nXXXXXXXXXX";
        match parse_get_response_head(raw, 4096).unwrap() {
            HeadParse::Complete(h) => {
                assert_eq!(h.content_length, 10);
                // head_len points at the first body byte.
                assert_eq!(&raw[h.head_len..], b"XXXXXXXXXX");
            }
            HeadParse::Incomplete => panic!("expected complete head"),
        }
    }

    #[test]
    fn reports_incomplete_when_terminator_missing() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\n";
        assert!(matches!(
            parse_get_response_head(raw, 4096).unwrap(),
            HeadParse::Incomplete
        ));
    }

    #[test]
    fn rejects_non_206_status() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n";
        assert!(matches!(
            parse_get_response_head(raw, 4096),
            Err(ProtoError::UnexpectedStatus(200))
        ));
    }

    #[test]
    fn upload_response_with_content_length_is_reusable() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n{}\r\n";

        let HeadParse::Complete(head) = parse_upload_response_head(raw).unwrap() else {
            panic!("expected complete head");
        };

        assert_eq!(head.content_length, 4);
        assert!(head.reuse_connection);
    }

    #[test]
    fn upload_response_without_content_length_is_not_reusable() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nok\r\n0\r\n\r\n";

        let HeadParse::Complete(head) = parse_upload_response_head(raw).unwrap() else {
            panic!("expected complete head");
        };

        assert!(!head.reuse_connection, "unknown body framing must not pool");
    }

    #[test]
    fn rejects_body_larger_than_requested() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 9000\r\n\r\n";
        assert!(matches!(
            parse_get_response_head(raw, 4096),
            Err(ProtoError::BodyTooLarge { .. })
        ));
    }
}
