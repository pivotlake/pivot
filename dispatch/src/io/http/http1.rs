//! Sans-IO HTTP/1.1 response helpers for the object-store range-read path.
//!
//! The transport (io_uring on Linux, blocking `std::net` elsewhere) owns the
//! socket and the buffers; this module just turns received bytes into a parsed
//! head and tracks body progress — it never touches a socket. Keeping it here
//! lets both backends share one implementation and lets it be unit-tested
//! without a socket.
//!
//! The scope is deliberately exactly what a `Range` GET into a fixed cache slot
//! needs: parse the response head (status + `Content-Length`) and decode an
//! identity body straight into the slot. There is no request encoding (the
//! request is a fixed `Range` GET, formatted in [`super::proto`]), no
//! chunked/close-delimited framing, and no general method/status handling.
//!
//! (An earlier revision generalised this into a full sans-IO core — the typed
//! `http` model, chunked decoding, request encoding — but nothing on the range
//! path used it, so it was trimmed back to the live surface. Re-introduce those
//! pieces alongside the first non-range consumer that needs them.)

use thiserror::Error;

/// Header slots for one parse. Range/object-store responses are small; this is
/// generous headroom without an unbounded allocation.
const MAX_HEADERS: usize = 64;

#[derive(Debug, Error)]
pub enum Http1Error {
    #[error("malformed http response: {0}")]
    Parse(#[from] httparse::Error),
    #[error("http response missing status code")]
    MissingStatus,
    #[error("invalid Content-Length")]
    InvalidContentLength,
    #[error("body longer than declared Content-Length")]
    BodyOverflow,
}

/// A parsed response head.
#[derive(Debug, Clone, Copy)]
pub struct ParsedHead {
    /// Byte length of the head including the terminating `\r\n\r\n`; the body
    /// begins at this offset in the parsed buffer.
    pub head_len: usize,
    /// Response status code.
    pub status: u16,
    /// `Content-Length`, if the header was present (and a valid integer).
    pub content_length: Option<u64>,
}

/// Outcome of parsing a (possibly incomplete) response head.
pub enum HeadStatus {
    /// The head hasn't fully arrived; feed more bytes and retry.
    Incomplete,
    Complete(ParsedHead),
}

/// Parse a response head from `buf`. Imposes no status/length *policy* — the
/// caller ([`super::proto`]) decides what's acceptable — and returns
/// [`HeadStatus::Incomplete`] until the blank-line terminator has arrived.
pub fn parse_response_head(buf: &[u8]) -> Result<HeadStatus, Http1Error> {
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut resp = httparse::Response::new(&mut headers);
    let head_len = match resp.parse(buf)? {
        httparse::Status::Partial => return Ok(HeadStatus::Incomplete),
        httparse::Status::Complete(n) => n,
    };

    let status = resp.code.ok_or(Http1Error::MissingStatus)?;

    let content_length = match resp
        .headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case("content-length"))
    {
        Some(h) => Some(
            std::str::from_utf8(h.value)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .ok_or(Http1Error::InvalidContentLength)?,
        ),
        None => None,
    };

    Ok(HeadStatus::Complete(ParsedHead {
        head_len,
        status,
        content_length,
    }))
}

/// What the transport should do to read more of the (identity) body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPlan {
    /// Body fully received; no more reads.
    Done,
    /// Receive up to `max` more bytes straight into the destination (zero-copy),
    /// then report them via [`BodyDecoder::consumed`].
    Direct { max: u64 },
}

/// Tracks an identity (`Content-Length`-framed) response body landing straight
/// in the cache slot.
///
/// The transport recvs/decrypts body bytes directly into the destination and
/// reports them with [`consumed`](Self::consumed) (the zero-copy path). The
/// bytes that arrive in the same packet as the head already sit in a scratch
/// buffer, so those are copied out with [`decode`](Self::decode).
#[derive(Debug)]
pub struct BodyDecoder {
    remaining: u64,
}

impl BodyDecoder {
    pub fn new(content_length: u64) -> Self {
        Self {
            remaining: content_length,
        }
    }

    /// Whether the whole body has been received.
    pub fn is_complete(&self) -> bool {
        self.remaining == 0
    }

    /// How the transport should read next.
    pub fn read_plan(&self) -> ReadPlan {
        if self.remaining == 0 {
            ReadPlan::Done
        } else {
            ReadPlan::Direct {
                max: self.remaining,
            }
        }
    }

    /// Report `n` body bytes received straight into the destination (the
    /// zero-copy [`ReadPlan::Direct`] path).
    pub fn consumed(&mut self, n: u64) -> Result<(), Http1Error> {
        if n > self.remaining {
            return Err(Http1Error::BodyOverflow);
        }
        self.remaining -= n;
        Ok(())
    }

    /// Copy body bytes out of `input` (the bytes that arrived alongside the
    /// head), invoking `sink` with them. Returns how many were consumed —
    /// capped at the declared body length, so trailing non-body bytes are left.
    pub fn decode(&mut self, input: &[u8], mut sink: impl FnMut(&[u8])) -> usize {
        let take = self.remaining.min(input.len() as u64) as usize;
        if take > 0 {
            sink(&input[..take]);
            self.remaining -= take as u64;
        }
        take
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(buf: &[u8]) -> ParsedHead {
        match parse_response_head(buf).unwrap() {
            HeadStatus::Complete(h) => h,
            HeadStatus::Incomplete => panic!("expected complete head"),
        }
    }

    #[test]
    fn parses_status_and_content_length() {
        let raw = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\n\r\nXXXX";
        let h = parse(raw);
        assert_eq!(h.status, 206);
        assert_eq!(h.content_length, Some(10));
        assert_eq!(&raw[h.head_len..], b"XXXX");
    }

    #[test]
    fn non_206_status_parses_without_policy() {
        let h = parse(b"HTTP/1.1 404 Not Found\r\nContent-Length: 3\r\n\r\n");
        assert_eq!(h.status, 404);
        assert_eq!(h.content_length, Some(3));
    }

    #[test]
    fn missing_content_length_is_none() {
        let h = parse(b"HTTP/1.1 200 OK\r\nServer: x\r\n\r\n");
        assert_eq!(h.content_length, None);
    }

    #[test]
    fn invalid_content_length_errors() {
        assert!(matches!(
            parse_response_head(b"HTTP/1.1 206 x\r\nContent-Length: notanum\r\n\r\n"),
            Err(Http1Error::InvalidContentLength)
        ));
    }

    #[test]
    fn incomplete_head_reported() {
        assert!(matches!(
            parse_response_head(b"HTTP/1.1 206 Partial\r\nContent-Length: 10\r\n").unwrap(),
            HeadStatus::Incomplete
        ));
    }

    #[test]
    fn body_direct_path() {
        let mut d = BodyDecoder::new(10);
        assert_eq!(d.read_plan(), ReadPlan::Direct { max: 10 });
        d.consumed(4).unwrap();
        assert_eq!(d.read_plan(), ReadPlan::Direct { max: 6 });
        d.consumed(6).unwrap();
        assert!(d.is_complete());
        assert_eq!(d.read_plan(), ReadPlan::Done);
    }

    #[test]
    fn body_overflow_rejected() {
        let mut d = BodyDecoder::new(4);
        assert!(matches!(d.consumed(5), Err(Http1Error::BodyOverflow)));
    }

    #[test]
    fn decode_copies_up_to_body_len_and_caps_trailing() {
        let mut d = BodyDecoder::new(5);
        let mut got = Vec::new();
        // Feed extra trailing bytes; decode must stop at the body boundary.
        let n = d.decode(b"hello world", |c| got.extend_from_slice(c));
        assert_eq!(n, 5);
        assert_eq!(got, b"hello");
        assert!(d.is_complete());
    }
}
