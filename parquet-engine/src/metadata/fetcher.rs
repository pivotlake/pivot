//! Fetch stage for Parquet footers.
//!
//! The operator submits logical file ranges through `OperatorIO`; dispatch's
//! requester resolves cache hits, allocates missing compressed-cache extents,
//! and immediately performs the physical reads. This
//! module owns only the footer-specific state machine: probe the tail, request
//! the exact footer when it is larger than the probe, and request the offset
//! indexes the footer points at when the bytes read so far do not hold them.
//! A writer puts the offset indexes right before the footer, so the probe
//! usually holds both and a file takes one read.

use super::FileRowGroups;
use crate::thrift::footer::FileMetaData;
use crate::types::columns::TableColumns;
use crate::types::metadata::RowGroupMetadata;
use crate::types::table::{
    Error, FOOTER_PROBE_BYTES, Result, locate_offset_indexes, parse_footer_thrift,
    parse_offset_indexes, row_groups_from_metadata,
};
use dispatch::io::{FileRange, OpenFile, OperatorIO, ReadRequestId, ReadResponse};
use dispatch::memory::memory_ctx;
use dispatch::{Sender, Unary};
use object_storage::{DataFile, FileRef};
use std::collections::HashMap;
use std::sync::Arc;

const PARQUET_MAGIC: [u8; 4] = *b"PAR1";

/// Local footer reads outstanding before another file is admitted. Footers are
/// small and scattered, so modest concurrency hides per-file latency.
const MAX_DISK_IN_FLIGHT: usize = 32;

/// Parse the 4-byte footer length from a file's `tail` (whose final 8 bytes are
/// `[footer_len][PAR1]`).
pub(super) fn footer_len_from_tail(tail: &[u8]) -> Result<usize> {
    if tail.len() < 8 || tail[tail.len() - 4..] != PARQUET_MAGIC {
        return Err(Error::InvalidFooter("missing PAR1 magic".to_string()));
    }
    let len = &tail[tail.len() - 8..tail.len() - 4];
    Ok(u32::from_le_bytes(len.try_into().unwrap()) as usize)
}

pub(super) struct FileRowGroupsFetcher {
    /// Footer domain state keyed by the node-local logical request id.
    in_flight: HashMap<ReadRequestId, FooterRead>,
    table_columns: TableColumns,
}

impl FileRowGroupsFetcher {
    pub(super) fn new(table_columns: TableColumns) -> Self {
        Self {
            in_flight: HashMap::new(),
            table_columns,
        }
    }

    fn request_region(
        &mut self,
        request: FooterRead,
        range: FileRange,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let id = io.read(
            &memory_ctx().page_caches().compressed,
            request.open_file.clone(),
            [range],
        )?;
        self.in_flight.insert(id, request);
        Ok(())
    }
}

impl Unary<DataFile, FileRowGroups> for FileRowGroupsFetcher {
    fn consume(
        &mut self,
        file: DataFile,
        _sender: &mut dyn Sender<FileRowGroups>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let DataFile {
            file: file_ref,
            source,
        } = file;
        let size = file_ref.size as usize;
        let open_file = source.open_read(file_ref.size).map_err(crate::op_err)?;
        let request = FooterRead::new(file_ref, open_file, size);
        let range = request.probe_range();
        self.request_region(request, range, io)
    }

    fn ready_for_more_work(&mut self) -> bool {
        let disk_in_flight = self
            .in_flight
            .values()
            .filter(|request| matches!(request.open_file, OpenFile::Local(_)))
            .count();
        let http_in_flight = self.in_flight.len() - disk_in_flight;
        disk_in_flight < MAX_DISK_IN_FLIGHT && http_in_flight < crate::http_readahead()
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<FileRowGroups>,
        io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let mut request = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown footer request");
        let bytes = response.into_bytes();

        match request
            .parse_region(&bytes, &self.table_columns)
            .map_err(crate::op_err)?
        {
            FooterProgress::Read(range) => self.request_region(request, range, io)?,
            FooterProgress::Done(row_groups) => sender.send(FileRowGroups {
                file: request.file,
                row_groups: row_groups.into_iter().map(Arc::new).collect(),
            })?,
        }
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<FileRowGroups>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}

/// Domain state for one file's footer read. Physical block accounting lives in
/// dispatch; this remembers only how the next completed byte range is parsed.
struct FooterRead {
    file: FileRef,
    open_file: OpenFile,
    size: usize,
    awaiting: AwaitedRegion,
}

/// What the read in flight for a file brings.
enum AwaitedRegion {
    /// The file's tail, which holds the footer's length and, unless the
    /// footer is longer than the probe, the footer.
    Tail,
    /// The footer, the last `footer_len` bytes of the region, and the
    /// stretch of the file before it, where its offset indexes are when the
    /// file has any.
    Footer { footer_len: usize },
    /// The offset indexes of the footer already parsed.
    OffsetIndexes(Box<FileMetaData>),
}

impl FooterRead {
    fn new(file: FileRef, open_file: OpenFile, size: usize) -> Self {
        Self {
            file,
            open_file,
            size,
            awaiting: AwaitedRegion::Tail,
        }
    }

    /// Where the footer ends: before its length and the closing magic, the
    /// file's last eight bytes.
    fn footer_end(&self) -> usize {
        self.size - 8
    }

    fn probe_range(&self) -> FileRange {
        let probe = self.size.min(FOOTER_PROBE_BYTES);
        FileRange::new(self.size - probe, probe)
    }

    /// Take the bytes of the region that was read, which ends where the
    /// footer ends, or is the offset indexes alone.
    fn parse_region(
        &mut self,
        bytes: &[u8],
        table_columns: &TableColumns,
    ) -> Result<FooterProgress> {
        let (file_meta, held_offset, held) =
            match std::mem::replace(&mut self.awaiting, AwaitedRegion::Tail) {
                AwaitedRegion::Tail => {
                    let footer_len = footer_len_from_tail(bytes)?;
                    if footer_len + 8 > self.size {
                        return Err(Error::InvalidFooter(format!(
                            "footer length {footer_len} exceeds file size {}",
                            self.size
                        )));
                    }
                    if footer_len + 8 > bytes.len() {
                        // The offset indexes of a footer end where it
                        // starts and take a fraction of its bytes, a few per
                        // page against a chunk's whole description, so
                        // reading half as many bytes again before the footer
                        // usually brings them with it.
                        let footer_start = self.footer_end() - footer_len;
                        let start = footer_start.saturating_sub(footer_len / 2);
                        self.awaiting = AwaitedRegion::Footer { footer_len };
                        return Ok(FooterProgress::Read(FileRange::new(
                            start,
                            self.footer_end() - start,
                        )));
                    }
                    let held = &bytes[..bytes.len() - 8];
                    let footer = &held[held.len() - footer_len..];
                    (
                        parse_footer_thrift(footer)?,
                        self.footer_end() - held.len(),
                        held,
                    )
                }
                AwaitedRegion::Footer { footer_len } => {
                    let footer = &bytes[bytes.len() - footer_len..];
                    (
                        parse_footer_thrift(footer)?,
                        self.footer_end() - bytes.len(),
                        bytes,
                    )
                }
                AwaitedRegion::OffsetIndexes(file_meta) => {
                    let indexes = locate_offset_indexes(&file_meta, self.size)?
                        .expect("offset indexes were requested because the footer points at some");
                    (*file_meta, indexes.start, bytes)
                }
            };

        let offset_indexes = match locate_offset_indexes(&file_meta, self.size)? {
            Some(indexes)
                if indexes.start < held_offset || indexes.end > held_offset + held.len() =>
            {
                let range = FileRange::new(indexes.start, indexes.len());
                self.awaiting = AwaitedRegion::OffsetIndexes(Box::new(file_meta));
                return Ok(FooterProgress::Read(range));
            }
            _ => parse_offset_indexes(&file_meta, held_offset, held)?,
        };
        Ok(FooterProgress::Done(row_groups_from_metadata(
            file_meta,
            offset_indexes,
            self.open_file.clone(),
            table_columns,
        )?))
    }
}

enum FooterProgress {
    /// The next region of the file to read.
    Read(FileRange),
    Done(Vec<RowGroupMetadata>),
}
