//! Remote-read end-to-end test: serve a Parquet file over a loopback HTTP server
//! that honours byte-range requests, then read it through the catalog pipeline —
//! footer and column chunks both via the io_uring ring — and check the rows come
//! back. Plain HTTP (not TLS) so the worker's default requester needs no injected
//! trust; the S3/GCS auth layer (presigned URLs) is orthogonal and exercised
//! against a live endpoint, not here.

mod common;
use common::*;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use url::Url;

use catalog::parquet::{ParquetTable, table_input};
use dispatch::Projection;

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
            thread::spawn(move || handle_conn(stream, &bytes, Duration::ZERO, None));
        }
    });
    Url::parse(&format!("http://127.0.0.1:{port}/data.parquet")).unwrap()
}

/// Serve `bytes` over loopback HTTP like [`serve_with_ranges`], but additionally
/// record the **peak number of reads in flight at once** into `peak` — the count
/// of requests being served simultaneously, a direct proxy for the reader's
/// read-ahead depth — and hold each response back by `delay` so concurrent reads
/// actually overlap (without it each finishes before the next is issued and
/// nothing piles up). The serial `try_recv -> None` bug pins the peak at 1. Any
/// path is served. Returns the bound port.
fn serve_counting_peak(bytes: Vec<u8>, peak: Arc<AtomicUsize>, delay: Duration) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let bytes = Arc::new(bytes);
    let live = Arc::new(AtomicUsize::new(0));
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let (bytes, peak, live) = (bytes.clone(), peak.clone(), live.clone());
            thread::spawn(move || handle_conn(stream, &bytes, delay, Some((live, peak))));
        }
    });
    port
}

fn handle_conn(
    mut stream: TcpStream,
    bytes: &[u8],
    delay: Duration,
    inflight: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
) {
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

        // Count this read as in flight while the (held) response is outstanding,
        // so the peak reflects how many reads overlap. Holding the response is
        // what lets them overlap at all.
        if let Some((live, peak)) = &inflight {
            peak.fetch_max(live.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
        }
        if !delay.is_zero() {
            thread::sleep(delay);
        }
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
        let wrote = stream.write_all(resp.as_bytes()).is_ok() && stream.write_all(body).is_ok();
        if let Some((live, _)) = &inflight {
            live.fetch_sub(1, Ordering::SeqCst);
        }
        if !wrote {
            return;
        }
    }
}

/// Parse the `Range: bytes=…` header into an inclusive `[start, end]`.
fn parse_range(head: &str, total: usize) -> (usize, usize) {
    let spec = head
        .lines()
        .find_map(|l| {
            l.strip_prefix("Range:")
                .or_else(|| l.strip_prefix("range:"))
        })
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
    // One row group per row, so a multi-row file produces several row groups —
    // exercising the remote fetcher's many-in-flight concurrency.
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(1))
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path).unwrap(),
        batch.schema(),
        Some(props),
    )
    .unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    std::fs::read(&path).unwrap()
}

/// A Parquet file well past the 64KB footer-probe window — random (so SNAPPY
/// can't shrink it) byte-string rows in one row group. Loading the footer caches
/// only the tail, so a later scan must actually fetch the column chunk rather
/// than finding it already cached from the footer probe.
fn wide_parquet_bytes() -> Vec<u8> {
    let mut s = 0x2545F491_4F6CDD1Du64;
    let buf: Vec<u8> = (0..256 * 512)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            b'a' + (s % 26) as u8
        })
        .collect();
    let strings: Vec<String> = buf
        .chunks(512)
        .map(|c| String::from_utf8(c.to_vec()).unwrap())
        .collect();
    let names: Vec<&str> = strings.iter().map(String::as_str).collect();
    let values: Vec<i64> = (0..256).collect();
    let batch = strings_and_ints(&names, &values);

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("wide.parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path).unwrap(),
        batch.schema(),
        Some(props),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    std::fs::read(&path).unwrap()
}

#[test]
fn materializes_a_remote_footer_over_the_ring() {
    let dispatch = dispatch(1);

    let batch = strings_and_ints(&["a", "b", "c", "d"], &[10, 20, 30, 40]);
    let bytes = parquet_bytes(&batch);
    let size = bytes.len() as u64;
    let url = serve_with_ranges(bytes);

    // Materialize the footer through `ParquetTable::from_remote_files` — the
    // on-ring path: the catalog-recorded size locates the footer, so its tail
    // window is fetched through the file cache exactly like a column chunk. No
    // HEAD, no suffix probe.
    let table = Arc::new(
        ParquetTable::from_remote_files(&dispatch, &[(url, size)]).expect("materialize footers"),
    );
    // Four rows, one row group each (writer set to one row per group).
    assert_eq!(table.row_groups().len(), 4, "all row-group footers parsed");

    // And the materialized table scans back the rows over the ring.
    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 4);
    let mut vals = collect_i64s(&results, 1);
    vals.sort();
    assert_eq!(vals, vec![10, 20, 30, 40]);
}

#[test]
fn materializes_many_remote_footers_concurrently() {
    // Several remote files (each its own loopback server) materialized at once,
    // exercising the metadata fetcher's many-in-flight HTTP path: footers route
    // back to the right in-flight file by `(location, offset)`, across workers.
    // A roomy cache pool, since every in-flight footer and row group pins a slot.
    let dispatch = dispatch_with_buffers(4, 128);

    let mut files = Vec::new();
    let mut expected: Vec<i64> = Vec::new();
    for i in 0..8i64 {
        // Two rows -> two row groups (writer is one row per group).
        let batch = strings_and_ints(&["x", "y"], &[i * 2, i * 2 + 1]);
        let bytes = parquet_bytes(&batch);
        let size = bytes.len() as u64;
        files.push((serve_with_ranges(bytes), size));
        expected.push(i * 2);
        expected.push(i * 2 + 1);
    }

    let table =
        Arc::new(ParquetTable::from_remote_files(&dispatch, &files).expect("materialize footers"));
    assert_eq!(table.row_groups().len(), 16, "8 files x 2 row groups each");

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 16);
    let mut vals = collect_i64s(&results, 1);
    vals.sort();
    assert_eq!(vals, expected);
}

/// A one-worker footer load over many remote files keeps several reads in flight
/// at once (the serial `FileInjector::try_recv -> None` bug pins it at one).
/// Read-ahead is the io_uring path; the non-Linux HTTP engine reads synchronously.
#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "read-ahead is io_uring-only")]
fn footer_load_keeps_many_reads_in_flight() {
    let dispatch = dispatch_with_buffers(1, 256);
    let peak = Arc::new(AtomicUsize::new(0));
    let bytes = parquet_bytes(&strings_and_ints(&["a"], &[1]));
    let size = bytes.len() as u64;
    let port = serve_counting_peak(bytes, peak.clone(), Duration::from_millis(50));
    let files: Vec<(Url, u64)> = (0..16)
        .map(|i| {
            (
                Url::parse(&format!("http://127.0.0.1:{port}/f{i}")).unwrap(),
                size,
            )
        })
        .collect();

    ParquetTable::from_remote_files(&dispatch, &files).unwrap();

    assert!(peak.load(Ordering::SeqCst) >= 4, "footer loads serialised");
}

/// A one-worker scan across many remote files keeps several column-chunk reads
/// in flight at once (the serial `RowGroupInjector::try_recv -> None` bug pins it
/// at one). The peak is reset after the footer load so it measures only the scan.
/// Read-ahead is the io_uring path; non-Linux reads sync.
#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "read-ahead is io_uring-only")]
fn row_group_scan_keeps_many_reads_in_flight() {
    let dispatch = dispatch_with_buffers(1, 256);
    let peak = Arc::new(AtomicUsize::new(0));
    let bytes = wide_parquet_bytes();
    let size = bytes.len() as u64;
    let port = serve_counting_peak(bytes, peak.clone(), Duration::from_millis(50));
    let files: Vec<(Url, u64)> = (0..16)
        .map(|i| {
            (
                Url::parse(&format!("http://127.0.0.1:{port}/f{i}")).unwrap(),
                size,
            )
        })
        .collect();
    let table = Arc::new(ParquetTable::from_remote_files(&dispatch, &files).unwrap());
    peak.store(0, Ordering::SeqCst);

    table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();

    assert!(
        peak.load(Ordering::SeqCst) >= 4,
        "row-group reads serialised"
    );
}
