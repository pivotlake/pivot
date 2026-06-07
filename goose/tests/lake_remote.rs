//! Remote-read end-to-end test: serve a Parquet file over a loopback HTTP server
//! that honours byte-range requests, then read it through the goose pipeline —
//! footer via a suffix-range GET, column chunks via the io_uring ring — and
//! check the rows come back. Plain HTTP (not TLS) so the worker's default
//! requester needs no injected trust; the S3/GCS auth layer (presigned URLs) is
//! orthogonal and exercised against a live endpoint, not here.

mod common;
use common::*;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use url::Url;

use dispatch::Projection;
use goose::parquet::{table_input, ParquetTable};

/// Serve `bytes` over loopback HTTP, answering `Range` requests with `206`.
/// Returns the bound URL. The server thread is detached and lives for the
/// process; each connection handles keep-alive requests in a loop (the ring
/// pools connections).
fn serve_with_ranges(bytes: Vec<u8>) -> Url {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let bytes = Arc::new(bytes);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let bytes = bytes.clone();
            thread::spawn(move || handle_conn(stream, &bytes));
        }
    });
    Url::parse(&format!("http://127.0.0.1:{port}/data.parquet")).unwrap()
}

fn handle_conn(mut stream: TcpStream, bytes: &[u8]) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        // Read until we have a full request head (`\r\n\r\n`). Range GETs have
        // no body, so the head is the whole request.
        let head_end = loop {
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                break pos + 4;
            }
            match stream.read(&mut tmp) {
                Ok(0) => return,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(_) => return,
            }
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        buf.drain(..head_end);

        let (start, end) = parse_range(&head, bytes.len());
        let body = &bytes[start..=end];
        let resp = format!(
            "HTTP/1.1 206 Partial Content\r\n\
             Content-Length: {}\r\n\
             Content-Range: bytes {start}-{end}/{}\r\n\
             Accept-Ranges: bytes\r\n\
             Connection: keep-alive\r\n\r\n",
            body.len(),
            bytes.len(),
        );
        if stream.write_all(resp.as_bytes()).is_err() || stream.write_all(body).is_err() {
            return;
        }
    }
}

/// Parse the `Range: bytes=…` header into an inclusive `[start, end]`.
fn parse_range(head: &str, total: usize) -> (usize, usize) {
    let spec = head
        .lines()
        .find_map(|l| l.strip_prefix("Range:").or_else(|| l.strip_prefix("range:")))
        .and_then(|v| v.trim().strip_prefix("bytes="))
        .unwrap_or("")
        .trim();
    match spec.split_once('-') {
        // Suffix range `-N`: the last N bytes.
        Some(("", n)) => {
            let n: usize = n.parse().unwrap_or(total);
            (total.saturating_sub(n), total - 1)
        }
        // `A-B` or `A-`.
        Some((a, b)) => {
            let start = a.parse().unwrap_or(0);
            let end = if b.is_empty() {
                total - 1
            } else {
                b.parse::<usize>().unwrap_or(total - 1).min(total - 1)
            };
            (start, end)
        }
        None => (0, total - 1),
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn parquet_bytes(batch: &arrow_array::RecordBatch) -> Vec<u8> {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("d.parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), batch.schema(), Some(props))
            .unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    std::fs::read(&path).unwrap()
}

#[test]
fn scans_a_remote_parquet_file_over_http() {
    let dispatch = dispatch(1);

    let batch = strings_and_ints(&["a", "b", "c", "d"], &[10, 20, 30, 40]);
    let url = serve_with_ranges(parquet_bytes(&batch));

    // Build the table from the remote URL (footer over HTTP), on a worker.
    let table = {
        let url = url.clone();
        dispatch
            .run_on_worker(move || Arc::new(ParquetTable::from_remote_files(&[url]).unwrap()))
            .expect("from_remote_files on a worker")
    };
    assert!(!table.row_groups().is_empty(), "footer parsed from the remote file");

    // Scan it: the column chunks are fetched as HTTP range reads on the ring.
    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 4);
    let mut vals = collect_i64s(&results, 1);
    vals.sort();
    assert_eq!(vals, vec![10, 20, 30, 40]);
}
