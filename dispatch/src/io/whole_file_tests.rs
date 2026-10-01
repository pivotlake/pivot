use super::tests::{client_config, read_head, server_tls_config};
use super::*;
use crate::io::{ReadRequestId, RemoteFile, open_direct_read};
use crate::memory::{BUFFER_SIZE, init_test_free_pool, memory_ctx};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

fn payload(length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| (index.wrapping_mul(37) % 251) as u8)
        .collect()
}

fn whole_file(port: u16, tls: bool) -> OpenFile {
    let scheme = if tls { "https" } else { "http" };
    OpenFile::Remote(Arc::new(
        RemoteFile::open_whole(
            format!("{scheme}://localhost:{port}/object")
                .parse()
                .unwrap(),
            None,
        )
        .unwrap(),
    ))
}

fn fetch(requester: &mut IORequester, file: OpenFile) -> std::result::Result<Vec<u8>, String> {
    requester
        .request_whole(
            1,
            2,
            ReadRequestId(0),
            file,
            &mut StatsCollector::disabled(),
        )
        .map_err(|error| error.to_string())?;
    let mut bytes = None;
    loop {
        for completion in requester.completions().unwrap() {
            if let Err(failed) = completion {
                return Err(failed.error.to_string());
            }
        }
        for ready in requester.take_ready_reads() {
            bytes = Some(ready.response.into_bytes());
        }
        if !requester.has_pending() {
            break;
        }
        requester.wait().unwrap();
    }
    bytes.ok_or_else(|| "whole read returned no response".to_string())
}

fn serve(mut stream: impl Read + Write, response: &[u8], split: bool) {
    let request = String::from_utf8(read_head(&mut stream)).unwrap();
    assert!(request.starts_with("GET /object HTTP/1.1\r\n"));
    assert!(!request.contains("Range:"));
    if split {
        for byte in response.iter().take(39) {
            stream.write_all(&[*byte]).unwrap();
            stream.flush().unwrap();
        }
        stream
            .write_all(&response[39.min(response.len())..])
            .unwrap();
    } else {
        stream.write_all(response).unwrap();
    }
    stream.flush().unwrap();
}

fn server(body: Vec<u8>, tls: bool) -> (u16, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = std::thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut response =
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
        response.extend_from_slice(&body);
        if tls {
            let connection = rustls::ServerConnection::new(server_tls_config()).unwrap();
            serve(
                rustls::StreamOwned::new(connection, socket),
                &response,
                true,
            );
        } else {
            serve(socket, &response, true);
        }
    });
    (port, handle)
}

#[test]
fn whole_get_handles_split_headers_and_reuses_memory_cache() {
    init_test_free_pool(4);
    let expected = payload(81_237);
    let (port, server) = server(expected.clone(), false);
    let file = whole_file(port, false);
    let mut requester = IORequester::default();

    let first = fetch(&mut requester, file.clone()).unwrap();
    server.join().unwrap();
    let cached = fetch(&mut requester, file).unwrap();

    assert_eq!(first, expected);
    assert_eq!(cached, expected);
}

#[test]
fn whole_https_get_streams_more_than_the_ring_can_hold() {
    init_test_free_pool(4);
    let expected = payload(11 * BUFFER_SIZE + 123);
    let (port, server) = server(expected.clone(), true);
    let mut requester = IORequester::with_http_config(client_config());

    let actual = fetch(&mut requester, whole_file(port, true)).unwrap();

    assert_eq!(actual, expected);
    server.join().unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn whole_disk_cache_survives_restart_with_an_exact_partial_tail() {
    init_test_free_pool(4);
    let directory = tempfile::tempdir().unwrap();
    let expected = payload(7 * BUFFER_SIZE + 123);
    let (port, server) = server(expected.clone(), false);
    let file = whole_file(port, false);
    let cache = Arc::new(DiskCache::open(directory.path().to_path_buf(), 64 << 20, 8).unwrap());
    let mut requester = IORequester::new(Some(cache));

    assert_eq!(fetch(&mut requester, file.clone()).unwrap(), expected);
    server.join().unwrap();
    drop(requester);
    memory_ctx().compressed_cache().clear();
    let cache = Arc::new(DiskCache::open(directory.path().to_path_buf(), 64 << 20, 8).unwrap());
    let mut requester = IORequester::new(Some(cache));
    let cached = fetch(&mut requester, file).unwrap();

    assert_eq!(cached, expected);
}

#[cfg(target_os = "linux")]
#[test]
fn empty_objects_complete_and_survive_disk_cache_restart() {
    init_test_free_pool(4);
    let directory = tempfile::tempdir().unwrap();
    let (port, server) = server(Vec::new(), true);
    let file = whole_file(port, true);
    let cache = Arc::new(DiskCache::open(directory.path().to_path_buf(), 1 << 20, 8).unwrap());
    let mut requester = IORequester::with_config(client_config(), Some(cache));

    assert!(fetch(&mut requester, file.clone()).unwrap().is_empty());
    server.join().unwrap();
    drop(requester);
    memory_ctx().compressed_cache().clear();
    let cache = Arc::new(DiskCache::open(directory.path().to_path_buf(), 1 << 20, 8).unwrap());
    let mut requester = IORequester::with_config(client_config(), Some(cache));

    assert!(fetch(&mut requester, file).unwrap().is_empty());
}

#[test]
fn local_whole_reads_are_bounded_and_include_the_final_partial_block() {
    init_test_free_pool(4);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metadata");
    let expected = payload(6 * BUFFER_SIZE + 317);
    std::fs::write(&path, &expected).unwrap();
    let file = OpenFile::Local(open_direct_read(&path).unwrap());
    let mut requester = IORequester::default();

    let actual = fetch(&mut requester, file).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn unsupported_framing_fails_without_a_size_lookup() {
    init_test_free_pool(4);
    for head in [
        "HTTP/1.1 200 OK\r\n\r\n",
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 5\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 5\r\n\r\n",
        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle =
            std::thread::spawn(move || serve(listener.accept().unwrap().0, head.as_bytes(), false));
        let mut requester = IORequester::default();

        let result = fetch(&mut requester, whole_file(port, false));

        assert!(result.is_err());
        assert!(!requester.has_pending());
        handle.join().unwrap();
    }
}

#[test]
fn a_truncated_get_never_makes_its_partial_cache_a_whole_object() {
    init_test_free_pool(4);
    let directory = tempfile::tempdir().unwrap();
    let expected = payload(3 * BUFFER_SIZE + 123);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let body = expected.clone();
    let handle = std::thread::spawn(move || {
        for truncate in [true, false] {
            let mut response =
                format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
            response.extend_from_slice(if truncate {
                &body[..BUFFER_SIZE + 91]
            } else {
                &body
            });
            serve(listener.accept().unwrap().0, &response, false);
        }
    });
    let file = whole_file(port, false);
    let cache = Arc::new(DiskCache::open(directory.path().to_path_buf(), 64 << 20, 8).unwrap());
    let mut requester = IORequester::new(Some(cache));

    let failed = fetch(&mut requester, file.clone());
    drop(requester);
    memory_ctx().compressed_cache().clear();
    let cache = Arc::new(DiskCache::open(directory.path().to_path_buf(), 64 << 20, 8).unwrap());
    let mut requester = IORequester::new(Some(cache));
    let retry = fetch(&mut requester, file).unwrap();

    assert!(failed.unwrap_err().contains("closed mid-response"));
    assert_eq!(retry, expected);
    handle.join().unwrap();
}

#[test]
fn a_rejected_local_submission_leaves_no_pending_whole_read() {
    init_test_free_pool(4);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metadata");
    std::fs::write(&path, b"metadata").unwrap();
    let file = OpenFile::Local(open_direct_read(&path).unwrap());
    let mut requester = IORequester::default();
    requester.backend.reject_reads();

    let result = fetch(&mut requester, file);

    assert!(result.is_err());
    assert!(!requester.has_pending());
}

#[test]
fn whole_get_retries_a_stale_pooled_connection_before_receiving_headers() {
    init_test_free_pool(4);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = std::thread::spawn(move || {
        for value in *b"ab" {
            let (mut socket, _) = listener.accept().unwrap();
            let request = String::from_utf8(read_head(&mut socket)).unwrap();
            assert!(!request.contains("Range:"));
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n")
                .unwrap();
            socket.write_all(&[value]).unwrap();
        }
    });
    let mut requester = IORequester::default();
    let files: Vec<_> = ["first", "second"]
        .map(|path| {
            OpenFile::Remote(Arc::new(
                RemoteFile::open_whole(
                    format!("http://localhost:{port}/{path}").parse().unwrap(),
                    None,
                )
                .unwrap(),
            ))
        })
        .into_iter()
        .collect();

    let first = fetch(&mut requester, files[0].clone()).unwrap();
    let second = fetch(&mut requester, files[1].clone()).unwrap();

    assert_eq!(first, b"a");
    assert_eq!(second, b"b");
    handle.join().unwrap();
}

#[test]
fn many_small_whole_reads_share_ring_slots() {
    init_test_free_pool(4);
    let directory = tempfile::tempdir().unwrap();
    let mut requester = IORequester::default();
    for index in 0..40 {
        let path = directory.path().join(index.to_string());
        std::fs::write(&path, vec![index as u8; 5_123]).unwrap();
        let file = OpenFile::Local(open_direct_read(&path).unwrap());
        requester
            .request_whole(
                1,
                2,
                ReadRequestId(index),
                file,
                &mut StatsCollector::disabled(),
            )
            .unwrap();
    }

    let mut completed = Vec::new();
    while requester.has_pending() {
        requester.wait().unwrap();
        for completion in requester.completions().unwrap() {
            assert!(completion.is_ok());
        }
        completed.extend(requester.take_ready_reads());
    }

    assert_eq!(completed.len(), 40);
    for ready in completed {
        let value = ready.response.id().0 as u8;
        assert_eq!(ready.response.into_bytes(), vec![value; 5_123]);
    }
}

#[test]
fn empty_whole_get_returns_an_empty_response() {
    init_test_free_pool(4);
    let (port, server) = server(Vec::new(), false);
    let mut requester = IORequester::default();

    let bytes = fetch(&mut requester, whole_file(port, false)).unwrap();

    assert!(bytes.is_empty());
    server.join().unwrap();
}

#[test]
fn a_failed_cache_write_does_not_fail_the_whole_download() {
    init_test_free_pool(4);
    let directory = tempfile::tempdir().unwrap();
    let expected = payload(BUFFER_SIZE + 123);
    let (port, server) = server(expected.clone(), false);
    let cache = Arc::new(DiskCache::open(directory.path().to_path_buf(), 64 << 20, 8).unwrap());
    let mut requester = IORequester::new(Some(cache));
    requester.cached_io.reject_whole_cache_writes();

    let actual = fetch(&mut requester, whole_file(port, false)).unwrap();

    assert_eq!(actual, expected);
    assert!(
        std::fs::read_dir(directory.path())
            .unwrap()
            .all(|entry| entry.unwrap().path().extension().is_none())
    );
    server.join().unwrap();
}
