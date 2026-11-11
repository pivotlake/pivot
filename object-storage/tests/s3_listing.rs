//! The request an S3 store sends for a listing that starts from a cursor.

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::thread::JoinHandle;

use object_storage::{ObjectPath, ObjectStore, S3Store};

/// Answer the first request with `response`, handing back the request text.
fn serve_once(response: String) -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 1024];
        let read = stream.read(&mut request).unwrap();
        let request = String::from_utf8_lossy(&request[..read]).into_owned();
        stream.write_all(response.as_bytes()).unwrap();
        request
    });
    (endpoint, handle)
}

#[test]
fn a_listing_from_a_cursor_starts_the_request_after_that_child_key() {
    let (endpoint, request) =
        serve_once("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string());
    let store =
        S3Store::anonymous_with_options("s3://bucket/root", "eu-west-1", Some(endpoint)).unwrap();

    let _ = store.list_filtered(&ObjectPath::new("t/metadata"), "v", Some("v02"));

    let request = request.join().unwrap();
    assert!(
        request.starts_with(
            "GET /bucket/?list-type=2&prefix=root%2Ft%2Fmetadata%2Fv&delimiter=%2F\
             &start-after=root%2Ft%2Fmetadata%2Fv02"
        ),
        "{request}"
    );
}
