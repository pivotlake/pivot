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
    #[error("malformed chunked body")]
    MalformedChunk,
    #[error("connection closed before the response body completed")]
    TruncatedBody,
}

/// A parsed response head.
#[derive(Debug, Clone)]
pub struct ParsedHead {
    /// Byte length of the head including the terminating `\r\n\r\n`; the body
    /// begins at this offset in the parsed buffer.
    pub head_len: usize,
    /// Response status code.
    pub status: u16,
    /// `Content-Length`, if the header was present (and a valid integer).
    pub content_length: Option<u64>,
    /// Whether `Transfer-Encoding` says the body comes in chunks.
    pub is_chunked: bool,
    /// The `ETag` header as the server sent it, quotes included, if the
    /// response carries one.
    pub etag: Option<String>,
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

    let is_chunked = resp
        .headers
        .iter()
        .filter(|h| h.name.eq_ignore_ascii_case("transfer-encoding"))
        .any(|h| {
            String::from_utf8_lossy(h.value)
                .split(',')
                .any(|coding| coding.trim().eq_ignore_ascii_case("chunked"))
        });

    let etag = resp
        .headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case("etag"))
        .map(|h| String::from_utf8_lossy(h.value).trim().to_string());

    Ok(HeadStatus::Complete(ParsedHead {
        head_len,
        status,
        content_length,
        is_chunked,
        etag,
    }))
}

/// How a response says where its body ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyFraming {
    /// The body is this many bytes.
    Length(u64),
    /// The body comes in chunks, each led by its size, and ends with a chunk
    /// of no bytes.
    Chunked,
    /// The body is everything the peer sends before it closes the
    /// connection.
    UntilClose,
}

impl BodyFraming {
    /// The framing of the body under `head`. A chunked transfer encoding
    /// takes precedence over a `Content-Length`, as HTTP/1.1 has it.
    pub fn of(head: &ParsedHead) -> Self {
        match head.content_length {
            _ if head.is_chunked => Self::Chunked,
            Some(length) => Self::Length(length),
            None => Self::UntilClose,
        }
    }
}

/// A response body that is kept whole, such as the small document an object
/// store answers a `POST` with. The transport hands it the body bytes as they
/// arrive and asks whether the body is all there.
#[derive(Debug)]
pub struct KeptBody {
    framing: BodyFraming,
    /// The bytes received, as they came off the connection.
    received: Vec<u8>,
    /// The body, once all of it has arrived.
    body: Option<Vec<u8>>,
}

impl KeptBody {
    pub fn new(framing: BodyFraming) -> Self {
        let mut kept = Self {
            framing,
            received: Vec::new(),
            body: None,
        };
        if framing == BodyFraming::Length(0) {
            kept.body = Some(Vec::new());
        }
        kept
    }

    /// Take `bytes`, the next of the response after its head.
    pub fn receive(&mut self, bytes: &[u8]) -> Result<(), Http1Error> {
        if self.body.is_some() {
            return Ok(());
        }
        self.received.extend_from_slice(bytes);
        match self.framing {
            BodyFraming::Length(length) => {
                if self.received.len() as u64 >= length {
                    self.received.truncate(length as usize);
                    self.body = Some(std::mem::take(&mut self.received));
                }
            }
            BodyFraming::Chunked => self.body = Self::decode_chunks(&self.received)?,
            BodyFraming::UntilClose => {}
        }
        Ok(())
    }

    /// Record that the peer closed the connection. That ends a body framed
    /// by the close, and cuts any other short.
    pub fn record_peer_closed(&mut self) -> Result<(), Http1Error> {
        if self.body.is_some() {
            return Ok(());
        }
        if self.framing != BodyFraming::UntilClose {
            return Err(Http1Error::TruncatedBody);
        }
        self.body = Some(std::mem::take(&mut self.received));
        Ok(())
    }

    /// Whether the whole body has arrived.
    pub fn is_complete(&self) -> bool {
        self.body.is_some()
    }

    /// The body. Empty until it [is complete](Self::is_complete).
    pub fn take_body(&mut self) -> Vec<u8> {
        self.body.take().unwrap_or_default()
    }

    /// The body `encoded` holds in chunks, or `None` while its last chunk,
    /// the one of no bytes, has not arrived.
    fn decode_chunks(encoded: &[u8]) -> Result<Option<Vec<u8>>, Http1Error> {
        let mut body = Vec::new();
        let mut rest = encoded;
        loop {
            let Some(line_end) = rest.windows(2).position(|pair| pair == b"\r\n") else {
                return Ok(None);
            };
            // A chunk's size is in hexadecimal, and may be followed by
            // extensions after a semicolon.
            let size_text =
                std::str::from_utf8(&rest[..line_end]).map_err(|_| Http1Error::MalformedChunk)?;
            let size_digits = size_text.split(';').next().unwrap_or("").trim();
            let size =
                usize::from_str_radix(size_digits, 16).map_err(|_| Http1Error::MalformedChunk)?;
            rest = &rest[line_end + 2..];
            if size == 0 {
                return Ok(Some(body));
            }
            if rest.len() < size + 2 {
                return Ok(None);
            }
            if &rest[size..size + 2] != b"\r\n" {
                return Err(Http1Error::MalformedChunk);
            }
            body.extend_from_slice(&rest[..size]);
            rest = &rest[size + 2..];
        }
    }
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
    fn a_chunked_body_is_whole_once_its_last_chunk_arrives() {
        let mut kept = KeptBody::new(BodyFraming::Chunked);

        kept.receive(b"3\r\nabc\r\n2;name=value\r\nde\r\n").unwrap();
        let was_complete_before_the_last_chunk = kept.is_complete();
        kept.receive(b"0\r\n\r\n").unwrap();

        assert!(!was_complete_before_the_last_chunk);
        assert_eq!(kept.take_body(), b"abcde");
    }

    #[test]
    fn a_body_framed_by_the_close_ends_when_the_peer_closes() {
        let mut kept = KeptBody::new(BodyFraming::UntilClose);

        kept.receive(b"<Done/>").unwrap();
        let was_complete_before_the_close = kept.is_complete();
        kept.record_peer_closed().unwrap();

        assert!(!was_complete_before_the_close);
        assert_eq!(kept.take_body(), b"<Done/>");
    }

    #[test]
    fn a_close_before_the_declared_length_is_a_truncated_body() {
        let mut kept = KeptBody::new(BodyFraming::Length(10));

        kept.receive(b"short").unwrap();
        let outcome = kept.record_peer_closed();

        assert!(matches!(outcome, Err(Http1Error::TruncatedBody)));
    }

    #[test]
    fn a_chunked_transfer_encoding_is_read_from_the_head() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";

        let head = parse(raw);

        assert_eq!(BodyFraming::of(&head), BodyFraming::Chunked);
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
    fn parses_the_etag_header() {
        let raw = b"HTTP/1.1 200 OK\r\nETag: \"abc123\"\r\nContent-Length: 0\r\n\r\n";

        let head = parse(raw);

        assert_eq!(head.etag.as_deref(), Some("\"abc123\""));
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
