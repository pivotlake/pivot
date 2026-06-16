//! Sans-IO HTTP/1.1 client core.
//!
//! This is the protocol engine: it turns bytes into requests and responses and
//! back, and **never touches a socket**. The transport (the io_uring engine on
//! Linux, blocking `std::net` elsewhere) owns the socket and the buffers; it
//! hands raw bytes to this module and takes raw bytes from it. That split is what
//! lets us keep a custom completion-based io_uring as the *only* I/O path — a
//! sans-IO core has nothing to adapt to readiness-vs-completion — while still
//! getting a real, typed, RFC-shaped HTTP implementation (arbitrary methods and
//! status codes, chunked transfer-coding, request bodies for `PUT`).
//!
//! It is built on the same crates `hyper` itself uses — [`http`] for the typed
//! message model and [`httparse`] for byte-level head/chunk parsing — rather than
//! forking hyper's (private, IO-coupled) `proto` modules.
//!
//! # Shape
//!
//! - [`write_request_head`] serializes a request line + headers (the body, if
//!   any, is sent separately by the transport).
//! - [`parse_response_head`] parses a response head into [`http::response::Parts`]
//!   and determines the body [`Framing`] (per RFC 7230 §3.3).
//! - [`BodyDecoder`] consumes body bytes incrementally according to that framing.
//!   It exposes two paths so the transport can stay zero-copy where possible:
//!   - **Direct** ([`ReadPlan::Direct`]) for identity bodies — the transport reads
//!     straight into its final destination and reports the count via
//!     [`BodyDecoder::consumed`]; no intermediate buffer, no copy.
//!   - **Buffered** ([`ReadPlan::Chunked`]) for chunked bodies — the transport
//!     reads into scratch and calls [`BodyDecoder::decode`], which strips the
//!     chunk framing and emits decoded runs to a sink.
//!
//! The range-read policy that today's object-store path layers on top (require
//! `206`, body no larger than the requested range) lives in [`super::proto`].

use http::{HeaderMap, HeaderName, HeaderValue, Method, Response, StatusCode, Version};
use std::io::Write as _;
use thiserror::Error;

/// Cap on response headers parsed in one head. Range/object-store responses are
/// small; this is generous headroom without an unbounded allocation.
const MAX_HEADERS: usize = 64;

#[derive(Debug, Error)]
pub enum Http1Error {
    #[error("malformed http message: {0}")]
    Parse(#[from] httparse::Error),
    #[error("response missing status code")]
    MissingStatus,
    #[error("invalid status code {0}")]
    InvalidStatus(u16),
    #[error("invalid header name")]
    InvalidHeaderName,
    #[error("invalid header value")]
    InvalidHeaderValue,
    #[error("invalid Content-Length")]
    InvalidContentLength,
    #[error("malformed chunked body")]
    InvalidChunk,
    #[error("body longer than declared length")]
    BodyOverflow,
    #[error("direct read path not valid for this body framing")]
    DirectNotSupported,
    #[error("connection closed mid-body")]
    Truncated,
}

// ============================================================================
// Request encoding
// ============================================================================

/// Serialize an HTTP/1.1 request head (request line + headers + the blank-line
/// terminator) into `out`.
///
/// The request *body*, if any, is written by the caller after the head — this
/// only emits framing for the head. The caller is responsible for the headers
/// HTTP/1.1 requires for its message (notably `Host`, and `Content-Length` or
/// `Transfer-Encoding` when there is a body); we serialize exactly what's in
/// `req` and infer nothing, so the typed [`http::Request`] is the single source
/// of truth.
///
/// The target is taken in origin form (`path[?query]`) from the request URI,
/// which is what an HTTP/1.1 origin server expects.
// Part of the core's public surface ahead of its consumer: today the range path
// formats its fixed GET in `proto::build_range_get`; this is the entry point for
// the general request/`PUT` path. Exercised by the unit tests.
#[allow(dead_code)]
pub fn write_request_head<T>(req: &http::Request<T>, out: &mut Vec<u8>) {
    let target = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    // Writing into a Vec is infallible.
    let _ = write!(out, "{} {} HTTP/1.1\r\n", req.method().as_str(), target);
    for (name, value) in req.headers() {
        out.extend_from_slice(name.as_str().as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"\r\n");
}

// ============================================================================
// Response head parsing + framing
// ============================================================================

/// How a message body's length is delimited, determined from the response head
/// and the request method per RFC 7230 §3.3.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// `Content-Length: n` — exactly `n` body bytes follow.
    Length(u64),
    /// `Transfer-Encoding: chunked` — body framed as chunks (see [`BodyDecoder`]).
    Chunked,
    /// Neither length nor chunked: the body runs until the connection closes.
    CloseDelimited,
    /// No body permitted: `HEAD` response, or status `1xx`/`204`/`304`.
    Empty,
}

/// A fully parsed response head.
#[derive(Debug)]
pub struct ParsedHead {
    /// Byte length of the head, including the terminating `\r\n\r\n`; the body
    /// (if any) begins at this offset in the parsed buffer.
    pub head_len: usize,
    /// Typed status line + headers.
    pub parts: http::response::Parts,
    /// How to read the body that follows.
    pub framing: Framing,
}

/// Outcome of parsing a (possibly incomplete) response head.
pub enum HeadStatus {
    /// The head hasn't fully arrived; feed more bytes and retry.
    Incomplete,
    Complete(ParsedHead),
}

/// Parse a response head from `buf`. `request_method` is needed because body
/// framing depends on it (a `HEAD` response carries headers but no body).
///
/// Returns [`HeadStatus::Incomplete`] if the terminating blank line hasn't
/// arrived yet. Imposes no status-code policy — a `404` or `500` parses fine;
/// callers apply their own expectations.
pub fn parse_response_head(buf: &[u8], request_method: &Method) -> Result<HeadStatus, Http1Error> {
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut resp = httparse::Response::new(&mut headers);
    let head_len = match resp.parse(buf)? {
        httparse::Status::Partial => return Ok(HeadStatus::Incomplete),
        httparse::Status::Complete(n) => n,
    };

    let code = resp.code.ok_or(Http1Error::MissingStatus)?;
    let status = StatusCode::from_u16(code).map_err(|_| Http1Error::InvalidStatus(code))?;

    let mut header_map = HeaderMap::with_capacity(resp.headers.len());
    for h in resp.headers.iter() {
        let name = HeaderName::from_bytes(h.name.as_bytes())
            .map_err(|_| Http1Error::InvalidHeaderName)?;
        let value = HeaderValue::from_bytes(h.value).map_err(|_| Http1Error::InvalidHeaderValue)?;
        header_map.append(name, value);
    }

    let framing = determine_framing(request_method, status, &header_map)?;

    // Assemble the typed parts. We don't keep httparse's `reason` — `http` derives
    // the canonical reason phrase from the status code.
    let mut response = Response::new(());
    *response.status_mut() = status;
    *response.version_mut() = Version::HTTP_11;
    *response.headers_mut() = header_map;
    let (parts, ()) = response.into_parts();

    Ok(HeadStatus::Complete(ParsedHead {
        head_len,
        parts,
        framing,
    }))
}

/// Apply RFC 7230 §3.3.3's response-body length rules. `Transfer-Encoding` wins
/// over `Content-Length`; certain statuses/methods are always bodyless.
fn determine_framing(
    method: &Method,
    status: StatusCode,
    headers: &HeaderMap,
) -> Result<Framing, Http1Error> {
    let code = status.as_u16();
    let bodyless = method == Method::HEAD || (100..200).contains(&code) || code == 204 || code == 304;
    if bodyless {
        return Ok(Framing::Empty);
    }

    if let Some(te) = headers.get(http::header::TRANSFER_ENCODING) {
        let te = te.to_str().map_err(|_| Http1Error::InvalidHeaderValue)?;
        // chunked must be the final coding; if it is, the body is chunk-framed.
        // Any other final coding leaves the length unknown → close-delimited.
        let final_is_chunked = te
            .rsplit(',')
            .next()
            .map(|s| s.trim().eq_ignore_ascii_case("chunked"))
            .unwrap_or(false);
        return Ok(if final_is_chunked {
            Framing::Chunked
        } else {
            Framing::CloseDelimited
        });
    }

    if let Some(cl) = headers.get(http::header::CONTENT_LENGTH) {
        let n: u64 = cl
            .to_str()
            .map_err(|_| Http1Error::InvalidHeaderValue)?
            .trim()
            .parse()
            .map_err(|_| Http1Error::InvalidContentLength)?;
        return Ok(Framing::Length(n));
    }

    Ok(Framing::CloseDelimited)
}

// ============================================================================
// Body decoding
// ============================================================================

/// What the transport should do to read more of the body, given the decoder's
/// current state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPlan {
    /// Body fully received; no more reads.
    Done,
    /// Identity body: receive up to `max` more bytes straight into the final
    /// destination (zero-copy), then report them via [`BodyDecoder::consumed`].
    /// `max` is exact for `Content-Length`; `u64::MAX` for close-delimited.
    Direct { max: u64 },
    /// Chunked body: receive into a scratch buffer and pass it to
    /// [`BodyDecoder::decode`], which strips the chunk framing.
    Chunked,
}

/// Incrementally decodes a response body according to its [`Framing`].
///
/// See the module docs for the Direct vs. Buffered read paths. A decoder handles
/// exactly one message body; build a fresh one per response.
#[derive(Debug)]
pub struct BodyDecoder {
    state: BodyState,
}

#[derive(Debug)]
enum BodyState {
    Length { remaining: u64 },
    Chunked(ChunkedState),
    Close { complete: bool },
    Empty,
}

impl BodyDecoder {
    pub fn new(framing: Framing) -> Self {
        let state = match framing {
            Framing::Length(n) => BodyState::Length { remaining: n },
            Framing::Chunked => BodyState::Chunked(ChunkedState::new()),
            Framing::CloseDelimited => BodyState::Close { complete: false },
            Framing::Empty => BodyState::Empty,
        };
        Self { state }
    }

    /// Whether the whole body has been received.
    pub fn is_complete(&self) -> bool {
        match &self.state {
            BodyState::Length { remaining } => *remaining == 0,
            BodyState::Chunked(c) => c.is_done(),
            BodyState::Close { complete } => *complete,
            BodyState::Empty => true,
        }
    }

    /// How the transport should read next.
    pub fn read_plan(&self) -> ReadPlan {
        match &self.state {
            BodyState::Empty => ReadPlan::Done,
            BodyState::Length { remaining } => {
                if *remaining == 0 {
                    ReadPlan::Done
                } else {
                    ReadPlan::Direct { max: *remaining }
                }
            }
            BodyState::Close { complete } => {
                if *complete {
                    ReadPlan::Done
                } else {
                    ReadPlan::Direct { max: u64::MAX }
                }
            }
            BodyState::Chunked(c) => {
                if c.is_done() {
                    ReadPlan::Done
                } else {
                    ReadPlan::Chunked
                }
            }
        }
    }

    /// Zero-copy path: report that `n` body bytes were received straight into the
    /// destination (see [`ReadPlan::Direct`]). Only valid for identity framings.
    pub fn consumed(&mut self, n: u64) -> Result<(), Http1Error> {
        match &mut self.state {
            BodyState::Length { remaining } => {
                if n > *remaining {
                    return Err(Http1Error::BodyOverflow);
                }
                *remaining -= n;
                Ok(())
            }
            // Length is unknown for a close-delimited body; nothing to track.
            BodyState::Close { .. } => Ok(()),
            _ => Err(Http1Error::DirectNotSupported),
        }
    }

    /// Buffered path: decode body bytes from `input`, invoking `sink` with each
    /// run of decoded payload bytes (chunk framing stripped). Returns how many
    /// bytes of `input` were consumed — which may be **less than** `input.len()`
    /// once the body ends, leaving any trailing bytes (e.g. a pipelined next
    /// response) for the caller.
    ///
    /// Required for chunked bodies; also works for identity framings (so tests
    /// and non-zero-copy transports can use a single path).
    pub fn decode(&mut self, input: &[u8], mut sink: impl FnMut(&[u8])) -> Result<usize, Http1Error> {
        match &mut self.state {
            BodyState::Empty => Ok(0),
            BodyState::Length { remaining } => {
                let take = (*remaining).min(input.len() as u64) as usize;
                if take > 0 {
                    sink(&input[..take]);
                }
                *remaining -= take as u64;
                Ok(take)
            }
            BodyState::Close { complete } => {
                if *complete {
                    return Ok(0);
                }
                if !input.is_empty() {
                    sink(input);
                }
                Ok(input.len())
            }
            BodyState::Chunked(c) => c.decode(input, sink),
        }
    }

    /// Signal that the connection closed. Completes a close-delimited body; for a
    /// length/chunked body that isn't finished, this is a truncation error.
    // No caller on the range path (identity bodies finish via `consumed`); needed
    // once close-delimited responses are read. Exercised by the unit tests.
    #[allow(dead_code)]
    pub fn feed_eof(&mut self) -> Result<(), Http1Error> {
        match &mut self.state {
            BodyState::Close { complete } => {
                *complete = true;
                Ok(())
            }
            BodyState::Empty => Ok(()),
            BodyState::Length { remaining } if *remaining == 0 => Ok(()),
            BodyState::Chunked(c) if c.is_done() => Ok(()),
            _ => Err(Http1Error::Truncated),
        }
    }
}

// ---------------------------------------------------------------------------
// Chunked transfer-coding decoder (RFC 7230 §4.1)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ChunkedState {
    stage: ChunkStage,
    /// Carries a partial chunk-size line or trailer line across `decode` calls.
    /// Chunk *data* never passes through here — only the small control lines.
    line: Vec<u8>,
}

/// All variants are `Copy` so `decode` can `match self.stage` by value, freeing
/// it to mutate `self.stage`/`self.line` within an arm without borrow conflicts.
#[derive(Debug, Clone, Copy)]
enum ChunkStage {
    /// Reading the `chunk-size [chunk-ext] CRLF` line.
    Size,
    /// Reading `remaining` bytes of chunk data.
    Data { remaining: u64 },
    /// Consuming the CRLF that follows chunk data. `seen` = CRLF bytes matched.
    DataCrlf { seen: u8 },
    /// After the zero-size chunk: consuming optional trailers up to the final
    /// blank line.
    Trailers,
    Done,
}

impl ChunkedState {
    fn new() -> Self {
        Self {
            stage: ChunkStage::Size,
            line: Vec::new(),
        }
    }

    fn is_done(&self) -> bool {
        matches!(self.stage, ChunkStage::Done)
    }

    fn decode(&mut self, input: &[u8], mut sink: impl FnMut(&[u8])) -> Result<usize, Http1Error> {
        let mut pos = 0;
        loop {
            match self.stage {
                ChunkStage::Done => return Ok(pos),

                ChunkStage::Size => {
                    if self.line.is_empty() {
                        // Fast path: parse straight out of `input` (no copy).
                        match httparse::parse_chunk_size(&input[pos..])
                            .map_err(|_| Http1Error::InvalidChunk)?
                        {
                            httparse::Status::Complete((consumed, size)) => {
                                pos += consumed;
                                self.begin_chunk(size);
                            }
                            httparse::Status::Partial => {
                                // The size line is split across reads; carry it.
                                self.line.extend_from_slice(&input[pos..]);
                                return Ok(input.len());
                            }
                        }
                    } else {
                        // Complete a carried size line up to its terminating LF,
                        // copying only the (tiny) control line, never chunk data.
                        match memchr::memchr(b'\n', &input[pos..]) {
                            Some(j) => {
                                self.line.extend_from_slice(&input[pos..=pos + j]);
                                pos += j + 1;
                                let size = match httparse::parse_chunk_size(&self.line)
                                    .map_err(|_| Http1Error::InvalidChunk)?
                                {
                                    httparse::Status::Complete((_, size)) => size,
                                    httparse::Status::Partial => return Err(Http1Error::InvalidChunk),
                                };
                                self.line.clear();
                                self.begin_chunk(size);
                            }
                            None => {
                                self.line.extend_from_slice(&input[pos..]);
                                return Ok(input.len());
                            }
                        }
                    }
                }

                ChunkStage::Data { remaining } => {
                    if pos >= input.len() {
                        return Ok(pos);
                    }
                    let take = remaining.min((input.len() - pos) as u64) as usize;
                    sink(&input[pos..pos + take]);
                    pos += take;
                    let left = remaining - take as u64;
                    self.stage = if left == 0 {
                        ChunkStage::DataCrlf { seen: 0 }
                    } else {
                        ChunkStage::Data { remaining: left }
                    };
                }

                ChunkStage::DataCrlf { mut seen } => {
                    while seen < 2 {
                        if pos >= input.len() {
                            self.stage = ChunkStage::DataCrlf { seen };
                            return Ok(pos);
                        }
                        let b = input[pos];
                        pos += 1;
                        match (seen, b) {
                            (0, b'\r') => seen = 1,
                            (1, b'\n') => seen = 2,
                            // Tolerate a bare LF terminator (some servers omit CR).
                            (0, b'\n') => seen = 2,
                            _ => return Err(Http1Error::InvalidChunk),
                        }
                    }
                    self.stage = ChunkStage::Size;
                }

                ChunkStage::Trailers => {
                    match memchr::memchr(b'\n', &input[pos..]) {
                        Some(j) => {
                            self.line.extend_from_slice(&input[pos..=pos + j]);
                            pos += j + 1;
                            // A line of only CR/LF is the blank line that ends the
                            // trailer section; any other line is a trailer header
                            // we don't surface — drop it and keep going.
                            let blank = self.line.iter().all(|&b| b == b'\r' || b == b'\n');
                            self.line.clear();
                            if blank {
                                self.stage = ChunkStage::Done;
                            }
                        }
                        None => {
                            self.line.extend_from_slice(&input[pos..]);
                            return Ok(input.len());
                        }
                    }
                }
            }
        }
    }

    fn begin_chunk(&mut self, size: u64) {
        self.stage = if size == 0 {
            // The zero-size chunk ends the body; trailers (often none) follow.
            ChunkStage::Trailers
        } else {
            ChunkStage::Data { remaining: size }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- request encoding ---

    #[test]
    fn writes_origin_form_request_head() {
        let req = http::Request::builder()
            .method(Method::GET)
            .uri("/data.parquet?v=1")
            .header("Host", "example.com")
            .header("Range", "bytes=0-1023")
            .body(())
            .unwrap();
        let mut out = Vec::new();
        write_request_head(&req, &mut out);
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("GET /data.parquet?v=1 HTTP/1.1\r\n"));
        assert!(text.contains("host: example.com\r\n")); // http lowercases names
        assert!(text.contains("range: bytes=0-1023\r\n"));
        assert!(text.ends_with("\r\n\r\n"));
    }

    #[test]
    fn writes_put_with_body_headers() {
        let req = http::Request::builder()
            .method(Method::PUT)
            .uri("/obj")
            .header("Host", "s3.example.com")
            .header("Content-Length", "5")
            .body(())
            .unwrap();
        let mut out = Vec::new();
        write_request_head(&req, &mut out);
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("PUT /obj HTTP/1.1\r\n"));
        assert!(text.contains("content-length: 5\r\n"));
    }

    // --- head parsing + framing ---

    fn parse(buf: &[u8], method: Method) -> ParsedHead {
        match parse_response_head(buf, &method).unwrap() {
            HeadStatus::Complete(h) => h,
            HeadStatus::Incomplete => panic!("expected complete head"),
        }
    }

    #[test]
    fn parses_content_length_framing() {
        let h = parse(
            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\n\r\nXXXX",
            Method::GET,
        );
        assert_eq!(h.parts.status, StatusCode::PARTIAL_CONTENT);
        assert_eq!(h.framing, Framing::Length(10));
        assert_eq!(h.head_len, b"HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\n\r\n".len());
    }

    #[test]
    fn parses_chunked_framing() {
        let h = parse(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
            Method::GET,
        );
        assert_eq!(h.framing, Framing::Chunked);
    }

    #[test]
    fn non_206_status_parses_without_policy() {
        let h = parse(b"HTTP/1.1 404 Not Found\r\nContent-Length: 3\r\n\r\n", Method::GET);
        assert_eq!(h.parts.status, StatusCode::NOT_FOUND);
        assert_eq!(h.framing, Framing::Length(3));
    }

    #[test]
    fn head_request_and_204_304_are_bodyless() {
        // HEAD response carries Content-Length but no body.
        let h = parse(b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\n", Method::HEAD);
        assert_eq!(h.framing, Framing::Empty);
        let h = parse(b"HTTP/1.1 204 No Content\r\n\r\n", Method::GET);
        assert_eq!(h.framing, Framing::Empty);
        let h = parse(b"HTTP/1.1 304 Not Modified\r\n\r\n", Method::GET);
        assert_eq!(h.framing, Framing::Empty);
    }

    #[test]
    fn missing_length_and_te_is_close_delimited() {
        let h = parse(b"HTTP/1.1 200 OK\r\nServer: x\r\n\r\n", Method::GET);
        assert_eq!(h.framing, Framing::CloseDelimited);
    }

    #[test]
    fn incomplete_head_reported() {
        assert!(matches!(
            parse_response_head(b"HTTP/1.1 206 Partial\r\nContent-Length: 10\r\n", &Method::GET).unwrap(),
            HeadStatus::Incomplete
        ));
    }

    // --- identity body: Direct + decode paths ---

    #[test]
    fn length_body_direct_path() {
        let mut d = BodyDecoder::new(Framing::Length(10));
        assert_eq!(d.read_plan(), ReadPlan::Direct { max: 10 });
        d.consumed(4).unwrap();
        assert_eq!(d.read_plan(), ReadPlan::Direct { max: 6 });
        d.consumed(6).unwrap();
        assert!(d.is_complete());
        assert_eq!(d.read_plan(), ReadPlan::Done);
    }

    #[test]
    fn length_body_overflow_rejected() {
        let mut d = BodyDecoder::new(Framing::Length(4));
        assert!(matches!(d.consumed(5), Err(Http1Error::BodyOverflow)));
    }

    #[test]
    fn length_body_decode_path() {
        let mut d = BodyDecoder::new(Framing::Length(5));
        let mut got = Vec::new();
        // Feed extra trailing bytes; decode must stop at the body boundary.
        let n = d.decode(b"hello world", |c| got.extend_from_slice(c)).unwrap();
        assert_eq!(n, 5);
        assert_eq!(got, b"hello");
        assert!(d.is_complete());
    }

    #[test]
    fn empty_body_is_immediately_complete() {
        let mut d = BodyDecoder::new(Framing::Empty);
        assert!(d.is_complete());
        assert_eq!(d.read_plan(), ReadPlan::Done);
        assert_eq!(d.decode(b"unexpected", |_| panic!("no body")).unwrap(), 0);
    }

    // --- close-delimited ---

    #[test]
    fn close_delimited_completes_on_eof() {
        let mut d = BodyDecoder::new(Framing::CloseDelimited);
        let mut got = Vec::new();
        d.decode(b"abc", |c| got.extend_from_slice(c)).unwrap();
        d.decode(b"def", |c| got.extend_from_slice(c)).unwrap();
        assert!(!d.is_complete());
        d.feed_eof().unwrap();
        assert!(d.is_complete());
        assert_eq!(got, b"abcdef");
    }

    #[test]
    fn length_body_eof_before_complete_is_truncation() {
        let mut d = BodyDecoder::new(Framing::Length(10));
        d.consumed(4).unwrap();
        assert!(matches!(d.feed_eof(), Err(Http1Error::Truncated)));
    }

    // --- chunked ---

    fn decode_all(framing: Framing, feeds: &[&[u8]]) -> Vec<u8> {
        let mut d = BodyDecoder::new(framing);
        let mut got = Vec::new();
        for feed in feeds {
            d.decode(feed, |c| got.extend_from_slice(c)).unwrap();
        }
        assert!(d.is_complete(), "body should be complete after all feeds");
        got
    }

    #[test]
    fn chunked_single_buffer() {
        let body = b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n";
        assert_eq!(decode_all(Framing::Chunked, &[body]), b"Wikipedia");
    }

    #[test]
    fn chunked_with_trailers() {
        let body = b"4\r\nWiki\r\n0\r\nX-Checksum: abc\r\n\r\n";
        assert_eq!(decode_all(Framing::Chunked, &[body]), b"Wiki");
    }

    #[test]
    fn chunked_with_extension_on_size_line() {
        let body = b"4;foo=bar\r\nWiki\r\n0\r\n\r\n";
        assert_eq!(decode_all(Framing::Chunked, &[body]), b"Wiki");
    }

    #[test]
    fn chunked_split_mid_size_line() {
        // 22-byte chunk: size line "16\r\n" (0x16 = 22) split between '1' and '6'.
        let feeds: &[&[u8]] = &[b"1", b"6\r\n", b"0123456789abcdefABCDEF\r\n0\r\n\r\n"];
        assert_eq!(decode_all(Framing::Chunked, feeds), b"0123456789abcdefABCDEF");
    }

    #[test]
    fn chunked_split_mid_data_and_crlf() {
        // Data and its trailing CRLF split across many small feeds.
        let feeds: &[&[u8]] = &[b"5\r\nhe", b"ll", b"o\r", b"\n0\r\n\r\n"];
        assert_eq!(decode_all(Framing::Chunked, feeds), b"hello");
    }

    #[test]
    fn chunked_byte_at_a_time() {
        let body = b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n";
        let feeds: Vec<&[u8]> = body.chunks(1).collect();
        assert_eq!(decode_all(Framing::Chunked, &feeds), b"abcde");
    }

    #[test]
    fn chunked_stops_at_body_end_leaving_trailing_bytes() {
        // A pipelined next response sits right after the terminating chunk.
        let mut d = BodyDecoder::new(Framing::Chunked);
        let input = b"4\r\nWiki\r\n0\r\n\r\nHTTP/1.1 200 OK\r\n";
        let mut got = Vec::new();
        let n = d.decode(input, |c| got.extend_from_slice(c)).unwrap();
        assert!(d.is_complete());
        assert_eq!(got, b"Wiki");
        assert_eq!(&input[n..], b"HTTP/1.1 200 OK\r\n");
    }

    #[test]
    fn chunked_eof_mid_body_is_truncation() {
        let mut d = BodyDecoder::new(Framing::Chunked);
        d.decode(b"4\r\nWi", |_| {}).unwrap();
        assert!(matches!(d.feed_eof(), Err(Http1Error::Truncated)));
    }
}
