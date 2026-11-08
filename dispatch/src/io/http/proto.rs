//! Object-store read **policy** over the sans-IO HTTP/1.1 helpers
//! ([`super::http1`]).
//!
//! `http1` parses a response head and tracks an identity body; this layer adds
//! the policy on top: build the GET, with a `Range` or of the whole object,
//! and accept only the response that answers it: a `206 Partial Content`
//! whose `Content-Length` is the requested range's, or a `200 OK` that
//! declares its body's length, since that is what the download is sized by.
//! Keeping
//! the policy here (rather than in `http1`) means both platform backends share
//! identical, socket-free, unit-testable logic.

use super::http1;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtoError {
    #[error("{0}")]
    Http1(#[from] http1::Http1Error),
    #[error("unexpected http status {status} (expected {expected})")]
    UnexpectedStatus { status: u16, expected: u16 },
    #[error("GET response missing Content-Length, which sizes its body")]
    MissingContentLength,
    #[error("a retried whole-object request answered with Content-Length {retry} after {first}")]
    WholeObjectLengthChanged { first: u64, retry: u64 },
    #[error("range response body ({got} bytes) does not match the expected {want} bytes")]
    BodyLengthMismatch { got: u64, want: usize },
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

/// Build an origin-form HTTP/1.1 GET: of `[offset, offset + len)` when
/// `range` is `Some((offset, len))`, else of the whole object.
///
/// `Connection: keep-alive` so the connection can be returned to the pool, and
/// `Accept-Encoding: identity` so the body length is the requested range's
/// exactly, or the whole object's own (no transfer/content encoding to undo).
/// `auth`, when present, is the `Authorization` header value (e.g. `Bearer
/// <token>`) for a store that authenticates each request rather than signing
/// the URL.
pub fn build_get(
    host: &str,
    target: &str,
    range: Option<(u64, usize)>,
    auth: Option<&str>,
) -> Vec<u8> {
    let range = range.map_or(String::new(), |(offset, len)| {
        format!("Range: bytes={offset}-{}\r\n", offset + len as u64 - 1)
    });
    let auth = auth.map_or(String::new(), |a| format!("Authorization: {a}\r\n"));
    format!(
        "GET {target} HTTP/1.1\r\n\
         Host: {host}\r\n\
         {range}\
         {auth}\
         Accept-Encoding: identity\r\n\
         Connection: keep-alive\r\n\
         User-Agent: pivotdb-dispatch/0.1\r\n\
         \r\n"
    )
    .into_bytes()
}

/// A parsed (and validated) response head.
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

/// Try to parse the head of a GET response from `buf`. Returns
/// [`HeadParse::Incomplete`] if the header section hasn't fully arrived yet.
///
/// A GET is always identity-framed (`Accept-Encoding: identity`, no multipart
/// range), so the head must carry a `Content-Length`: the body size that
/// lands. With `expected_len`, the GET was of a range and the response must
/// be a `206 Partial Content` of exactly that many bytes, the range clamped
/// to the object's size: a shorter body (a truncating proxy, a replaced
/// object) would leave part of the destination slot unwritten, and a longer
/// one would overflow it, so both are rejected before any body byte lands.
/// Without, the GET was of the whole object and the response must be a `200
/// OK`, whose declared length is the only thing the download's destination
/// can be sized by: a body of unknown framing (chunked, close-delimited) is
/// refused rather than grown into.
pub fn parse_get_response_head(
    buf: &[u8],
    expected_len: Option<usize>,
) -> Result<HeadParse, ProtoError> {
    let head = match http1::parse_response_head(buf)? {
        http1::HeadStatus::Incomplete => return Ok(HeadParse::Incomplete),
        http1::HeadStatus::Complete(head) => head,
    };
    let expected_status = if expected_len.is_some() { 206 } else { 200 };
    if head.status != expected_status {
        return Err(ProtoError::UnexpectedStatus {
            status: head.status,
            expected: expected_status,
        });
    }
    let content_length = head
        .content_length
        .ok_or(ProtoError::MissingContentLength)?;
    if let Some(expected_len) = expected_len
        && content_length != expected_len as u64
    {
        return Err(ProtoError::BodyLengthMismatch {
            got: content_length,
            want: expected_len,
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
        let req = build_get("example.com", "/data.parquet", Some((4096, 8192)), None);
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
        let req = build_get(
            "example.com",
            "/data.parquet",
            Some((0, 1)),
            Some("Bearer tok"),
        );
        let text = String::from_utf8(req).unwrap();
        assert!(text.contains("Authorization: Bearer tok\r\n"));
    }

    #[test]
    fn parses_complete_206_head() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\nContent-Range: bytes 0-9/100\r\n\r\nXXXXXXXXXX";
        match parse_get_response_head(raw, Some(10)).unwrap() {
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
            parse_get_response_head(raw, Some(4096)).unwrap(),
            HeadParse::Incomplete
        ));
    }

    #[test]
    fn rejects_non_206_status() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n";
        assert!(matches!(
            parse_get_response_head(raw, Some(4096)),
            Err(ProtoError::UnexpectedStatus {
                status: 200,
                expected: 206
            })
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
    fn whole_object_get_carries_no_range() {
        let req = build_get("example.com", "/list.avro", None, Some("Bearer tok"));

        let text = String::from_utf8(req).unwrap();

        assert!(text.starts_with("GET /list.avro HTTP/1.1\r\n"));
        assert!(!text.contains("Range:"));
        assert!(text.contains("Authorization: Bearer tok\r\n"));
        assert!(text.contains("Accept-Encoding: identity\r\n"));
    }

    #[test]
    fn whole_object_head_yields_the_declared_length() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nXX";

        let HeadParse::Complete(head) = parse_get_response_head(raw, None).unwrap() else {
            panic!("expected complete head");
        };

        assert_eq!(head.content_length, 12);
        assert_eq!(&raw[head.head_len..], b"XX");
        assert!(head.reuse_connection);
    }

    #[test]
    fn whole_object_head_rejects_a_partial_content_status() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 12\r\n\r\n";

        let result = parse_get_response_head(raw, None);

        assert!(matches!(
            result,
            Err(ProtoError::UnexpectedStatus {
                status: 206,
                expected: 200
            })
        ));
    }

    #[test]
    fn whole_object_head_without_a_length_is_refused() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";

        let result = parse_get_response_head(raw, None);

        assert!(matches!(result, Err(ProtoError::MissingContentLength)));
    }

    #[test]
    fn rejects_body_larger_than_expected() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 9000\r\n\r\n";
        assert!(matches!(
            parse_get_response_head(raw, Some(4096)),
            Err(ProtoError::BodyLengthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_body_shorter_than_expected() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 2048\r\n\r\n";

        let result = parse_get_response_head(raw, Some(4096));

        assert!(matches!(
            result,
            Err(ProtoError::BodyLengthMismatch {
                got: 2048,
                want: 4096
            })
        ));
    }
}
