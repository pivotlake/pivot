//! Small cache-backed reads executed by a single Dispatch worker.

use super::{FileRange, OpenFile, OperatorIO, ReadResponse};
use crate::DataFlowDispatcher;
use crate::api::OperatorSpec;
use crate::data_flow::{Result, WorkStatus};
use crate::memory::memory_ctx;
use crate::operations::channels::Sender;
use crate::operations::nullary::{Nullary, NullaryFactory, NullaryOperatorFactory};

/// Runs one raw file-range read through Dispatch's ordinary requester and cache
/// stack. Calls are round-robined across workers and each call constructs only
/// the one operator graph that will actually execute.
#[derive(Clone)]
pub struct DispatchReader {
    dispatcher: DataFlowDispatcher,
}

impl DispatchReader {
    pub fn new(dispatcher: DataFlowDispatcher) -> Self {
        Self { dispatcher }
    }

    /// Block until `range` has been resolved from the shared caches or backing
    /// store. Async callers should invoke this on a blocking executor thread.
    pub fn read(&self, open_file: OpenFile, range: FileRange) -> Result<Vec<u8>> {
        let worker = self.dispatcher.next_worker();
        let factory = NullaryOperatorFactory::new(ReadOnceFactory { open_file, range });
        let mut results = OperatorSpec::new(self.dispatcher.clone(), [factory])
            .execute_on(worker)
            .collect()?;
        Ok(results
            .pop()
            .expect("read-once dataflow should produce exactly one result"))
    }
}

struct ReadOnceFactory {
    open_file: OpenFile,
    range: FileRange,
}

impl NullaryFactory<Vec<u8>> for ReadOnceFactory {
    type Nullary = ReadOnce;

    fn build_nullary(self) -> Self::Nullary {
        match &self.open_file {
            OpenFile::Remote(remote) if remote.is_immutable() => memory_ctx()
                .compressed_cache()
                .open_immutable_entry(self.open_file.clone()),
            _ => memory_ctx()
                .compressed_cache()
                .open_entry(self.open_file.clone()),
        }
        ReadOnce {
            open_file: Some(self.open_file),
            range: self.range,
            complete: false,
        }
    }
}

struct ReadOnce {
    open_file: Option<OpenFile>,
    range: FileRange,
    complete: bool,
}

impl Nullary<Vec<u8>> for ReadOnce {
    fn run(
        &mut self,
        _sender: &mut dyn Sender<Vec<u8>>,
        io: &mut OperatorIO,
    ) -> crate::operations::nullary::Result<WorkStatus> {
        let Some(open_file) = self.open_file.take() else {
            return Ok(WorkStatus::Pending);
        };
        io.read(open_file, [self.range])
            .map_err(|error| crate::operations::nullary::Error::Other(Box::new(error)))?;
        Ok(WorkStatus::Ran)
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<Vec<u8>>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> crate::operations::nullary::Result<()> {
        sender.send(response.into_bytes())?;
        self.complete = true;
        Ok(())
    }

    fn finish(
        &mut self,
        _sender: &mut dyn Sender<Vec<u8>>,
    ) -> crate::operations::nullary::Result<bool> {
        Ok(self.complete)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use url::Url;

    use super::*;
    use crate::Dispatch;
    use crate::io::RemoteFile;

    const OBJECT_SIZE: usize = 8192;

    #[test]
    fn immutable_reopen_hits_memory_cache_with_new_query_credentials() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(AtomicUsize::new(0));
        let server_requests = requests.clone();
        let server = thread::spawn(move || serve_ranges(listener, server_requests));

        let dispatch = Dispatch::spin_up(1, 16, None);
        let reader = DispatchReader::new(dispatch.dispatcher().clone());
        let first = immutable_file(port, "first");
        let second = immutable_file(port, "refreshed");
        let range = FileRange::new(123, 333);

        let expected: Vec<u8> = (range.offset..range.offset + range.len)
            .map(pattern)
            .collect();
        assert_eq!(reader.read(first, range).unwrap(), expected);
        assert_eq!(reader.read(second, range).unwrap(), expected);
        assert_eq!(requests.load(Ordering::Relaxed), 1);

        dispatch.exit();
        server.join().unwrap();
    }

    fn immutable_file(port: u16, signature: &str) -> OpenFile {
        let url = Url::parse(&format!(
            "http://127.0.0.1:{port}/bucket/manifest.avro?signature={signature}"
        ))
        .unwrap();
        OpenFile::Remote(Arc::new(
            RemoteFile::open_immutable(url, None, OBJECT_SIZE as u64).unwrap(),
        ))
    }

    fn serve_ranges(listener: TcpListener, requests: Arc<AtomicUsize>) {
        let (mut stream, _) = listener.accept().unwrap();
        loop {
            let head = read_head(&mut stream);
            if head.is_empty() {
                break;
            }
            requests.fetch_add(1, Ordering::Relaxed);
            let (start, requested_end) = parse_range(&head);
            let end = requested_end.min(OBJECT_SIZE - 1);
            let body: Vec<u8> = (start..=end).map(pattern).collect();
            let response = format!(
                "HTTP/1.1 206 Partial Content\r\n\
                 Content-Length: {}\r\n\
                 Content-Range: bytes {start}-{end}/{OBJECT_SIZE}\r\n\
                 Connection: keep-alive\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
            stream.flush().unwrap();
        }
    }

    fn read_head(stream: &mut impl Read) -> Vec<u8> {
        let mut head = Vec::new();
        let mut byte = [0; 1];
        loop {
            match stream.read(&mut byte) {
                Ok(0) => break,
                Ok(_) => {
                    head.push(byte[0]);
                    if head.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                Err(error) => panic!("reading test HTTP request: {error}"),
            }
        }
        head
    }

    fn parse_range(head: &[u8]) -> (usize, usize) {
        let head = String::from_utf8_lossy(head);
        let range = head
            .lines()
            .find(|line| line.to_ascii_lowercase().starts_with("range:"))
            .unwrap()
            .split('=')
            .nth(1)
            .unwrap();
        let (start, end) = range.split_once('-').unwrap();
        (start.parse().unwrap(), end.parse().unwrap())
    }

    fn pattern(offset: usize) -> u8 {
        (offset % 251) as u8
    }
}
