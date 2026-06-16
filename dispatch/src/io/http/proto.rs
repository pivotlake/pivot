//! Object-store range-read **policy** over the sans-IO HTTP/1.1 core
//! ([`super::http1`]).
//!
//! The core is general (any method, any status, any body framing); this layer
//! pins it to what the cache's remote reads require: a `Range` GET, a `206
//! Partial Content` response, and an identity body no larger than the requested
//! range (so it lands exactly in the destination slot). Keeping this policy out
//! of the core means the core stays reusable for future `PUT`s, non-`206`
//! responses, and chunked bodies — and that both platform backends share
//! identical, socket-free, unit-testable logic.

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
}

/// Build an origin-form HTTP/1.1 `Range` GET for `[offset, offset + len)`.
///
/// `Connection: keep-alive` so the connection can be returned to the pool, and
/// `Accept-Encoding: identity` so the body length matches the requested range
/// exactly (no transfer/content encoding to undo).
pub fn build_range_get(host: &str, target: &str, offset: u64, len: usize) -> Vec<u8> {
    let end = offset + len as u64 - 1;
    format!(
        "GET {target} HTTP/1.1\r\n\
         Host: {host}\r\n\
         Range: bytes={offset}-{end}\r\n\
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
pub fn parse_response_head(buf: &[u8], requested_len: usize) -> Result<HeadParse, ProtoError> {
    let head = match http1::parse_response_head(buf, &http::Method::GET)? {
        http1::HeadStatus::Incomplete => return Ok(HeadParse::Incomplete),
        http1::HeadStatus::Complete(head) => head,
    };

    let status = head.parts.status.as_u16();
    if status != 206 {
        return Err(ProtoError::UnexpectedStatus(status));
    }

    // A range response is always identity-framed (we send `Accept-Encoding:
    // identity` and don't request multipart). Anything else — chunked, or no
    // length at all — means we can't size the body against the slot.
    let content_length = match head.framing {
        http1::Framing::Length(n) => n,
        _ => return Err(ProtoError::MissingContentLength),
    };

    if content_length > requested_len as u64 {
        return Err(ProtoError::BodyTooLarge {
            got: content_length,
            want: requested_len,
        });
    }

    Ok(HeadParse::Complete(ResponseHead {
        head_len: head.head_len,
        content_length,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_get_has_expected_request_line_and_headers() {
        let req = build_range_get("example.com", "/data.parquet", 4096, 8192);
        let text = String::from_utf8(req).unwrap();
        assert!(text.starts_with("GET /data.parquet HTTP/1.1\r\n"));
        assert!(text.contains("Host: example.com\r\n"));
        // [4096, 4096+8192) -> bytes=4096-12287
        assert!(text.contains("Range: bytes=4096-12287\r\n"));
        assert!(text.ends_with("\r\n\r\n"));
    }

    #[test]
    fn parses_complete_206_head() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\nContent-Range: bytes 0-9/100\r\n\r\nXXXXXXXXXX";
        match parse_response_head(raw, 4096).unwrap() {
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
            parse_response_head(raw, 4096).unwrap(),
            HeadParse::Incomplete
        ));
    }

    #[test]
    fn rejects_non_206_status() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n";
        assert!(matches!(
            parse_response_head(raw, 4096),
            Err(ProtoError::UnexpectedStatus(200))
        ));
    }

    #[test]
    fn rejects_body_larger_than_requested() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 9000\r\n\r\n";
        assert!(matches!(
            parse_response_head(raw, 4096),
            Err(ProtoError::BodyTooLarge { .. })
        ));
    }
}
