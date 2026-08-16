//! Fetch stage: reads each file's Parquet footer — through the io_uring ring and
//! the compressed cache, exactly like a column-chunk read — and emits one
//! [`FileRowGroups`] per file (its [`FileRef`] paired with its row groups). The
//! footer-reading analog of the column-chunk scan fetcher: it stages one pending
//! IO request per region it reads, and the worker's request tracker does all the
//! cache resolution, read dedup, and in-flight bounding. The footer-specific
//! part is the per-file state machine ([`FooterRead`], round-tripped as the
//! request state): read the tail probe window, parse the footer, and — if it
//! overflowed the probe — read it exactly before parsing.

use super::FileRowGroups;
use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::types::table::{Error, FOOTER_PROBE_BYTES, Result, row_groups_from_footer};
use crate::store::{DataFile, FileRef};
use dispatch::io::{
    CacheTiers, CompletedIoRequest, FileRange, OpenFile, OperatorIO, PendingIoRequest, RangePart,
};
use dispatch::memory::memory_ctx;
use dispatch::{Sender, Unary};
use planner::catalog::Column;
use std::sync::Arc;

const PARQUET_MAGIC: [u8; 4] = *b"PAR1";

/// Footer reads one worker keeps unanswered before it stops consuming more
/// files. Consuming a file opens its descriptor, so without this bound a
/// many-thousand-file table would open every file up front; with it, open
/// descriptors and staged reads grow only as answers come back. Well above the
/// claim depth a column-chunk scan uses: footers are small and scattered, so
/// keeping many in flight hides per-file read latency.
const MAX_UNANSWERED_FOOTER_READS: usize = 32;

/// Parse the 4-byte footer length from a file's `tail` (whose final 8 bytes are
/// `[footer_len][PAR1]`). The reader fetches the tail through the cache, so it
/// has the bytes in hand rather than `seek`+`read`ing them.
pub(super) fn footer_len_from_tail(tail: &[u8]) -> Result<usize> {
    if tail.len() < 8 || tail[tail.len() - 4..] != PARQUET_MAGIC {
        return Err(Error::InvalidFooter("missing PAR1 magic".to_string()));
    }
    let len = &tail[tail.len() - 8..tail.len() - 4];
    Ok(u32::from_le_bytes(len.try_into().unwrap()) as usize)
}

/// Reads files' footers and emits one [`FileRowGroups`] per file, keeping many
/// reads in flight (bounded by [`MAX_UNANSWERED_FOOTER_READS`]).
pub(super) struct FileRowGroupsFetcher {
    /// Requests staged but not yet answered. `finish` waits for this to drain,
    /// and `ready_for_more_work` bounds it.
    unanswered_requests: usize,
    /// The table's declared schema, reconciled with each parsed footer's
    /// schema (see `apply_declared_types`); empty when the table declares
    /// none.
    declared_columns: Arc<[Column]>,
}

impl FileRowGroupsFetcher {
    pub(super) fn new(declared_columns: Arc<[Column]>) -> Self {
        Self {
            unanswered_requests: 0,
            declared_columns,
        }
    }
}

impl Unary<DataFile, FileRowGroups> for FileRowGroupsFetcher {
    fn consume(
        &mut self,
        file: DataFile,
        _sender: &mut dyn Sender<FileRowGroups>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        // Open the file's transport. The size — which locates the footer's tail
        // window with no HEAD/suffix probe — is already carried by the data
        // file (`stat`ed at listing time for a local file, from the store
        // listing for a remote one). The `file_ref` rides through to the emitted
        // `FileRowGroups`.
        let DataFile {
            file: file_ref,
            source,
        } = file;
        let size = file_ref.size as usize;
        let open_file = source
            .open_read(file_ref.size)
            .map_err(crate::parquet::op_err)?;
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());

        let probe = size.min(FOOTER_PROBE_BYTES);
        let footer_read = FooterRead {
            file: file_ref,
            open_file,
            size,
            reading_exact: false,
        };
        self.unanswered_requests += 1;
        io.push_read(footer_read.read_region(size - probe, probe));
        Ok(())
    }

    fn process_io_response(
        &mut self,
        sender: &mut dyn Sender<FileRowGroups>,
        io: &mut OperatorIO,
        response: CompletedIoRequest,
    ) -> dispatch::UnaryResult<()> {
        let footer_read: Box<FooterRead> = response
            .state
            .downcast()
            .expect("a footer request rides its FooterRead as state");
        match footer_read
            .parse_region(region_bytes(response.ranges), &self.declared_columns)
            .map_err(crate::parquet::op_err)?
        {
            Parsed::RowGroups { file, row_groups } => {
                self.unanswered_requests -= 1;
                let row_groups = row_groups.into_iter().map(Arc::new).collect();
                sender.send(FileRowGroups { file, row_groups })?;
            }
            // The footer overflowed the probe; the exact re-read goes back
            // through the same staging path, and its response lands here again.
            Parsed::ExactReadNeeded(request) => io.push_read(request),
        }
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.unanswered_requests < MAX_UNANSWERED_FOOTER_READS
    }

    fn finish(&mut self, _sender: &mut dyn Sender<FileRowGroups>) -> dispatch::UnaryResult<bool> {
        Ok(self.unanswered_requests == 0)
    }
}

/// One file's footer read, round-tripped as the state of each region's IO
/// request: first the tail probe window, then — only if the footer overflows
/// it — the exact footer.
struct FooterRead {
    /// The file's durable identity, stamped onto the emitted [`FileRowGroups`].
    file: FileRef,
    /// The open file (it keeps the handle alive, and travels into the row
    /// groups as their file).
    open_file: OpenFile,
    /// Total file size, known when the read starts.
    size: usize,
    /// `false` while reading the tail probe window; `true` once the footer was
    /// found to overflow the probe and the exact footer is being read.
    reading_exact: bool,
}

/// What parsing a completed region yielded: the file's row groups, or the
/// exact-footer re-read to stage because the footer overflowed the probe.
enum Parsed {
    RowGroups {
        file: FileRef,
        row_groups: Vec<RowGroupMetadata>,
    },
    ExactReadNeeded(PendingIoRequest),
}

impl FooterRead {
    /// The IO request reading `[offset, offset + len)` of the file, carrying
    /// this state along. Footer bytes must arrive exactly as they are on disk,
    /// so only the compressed cache may serve them.
    fn read_region(self, offset: usize, len: usize) -> PendingIoRequest {
        PendingIoRequest {
            file: self.open_file.clone(),
            ranges: vec![FileRange { offset, len }],
            tiers: CacheTiers::CompressedOnly,
            state: Box::new(self),
        }
    }

    /// Parse the just-completed region's `bytes`: the file's row groups, or
    /// the exact re-read to stage if the footer overflowed the probe window.
    fn parse_region(self, bytes: Vec<u8>, declared_columns: &[Column]) -> Result<Parsed> {
        let footer: &[u8] = if self.reading_exact {
            // The exact read returned precisely the footer.
            &bytes
        } else {
            let footer_len = footer_len_from_tail(&bytes)?;
            // A footer that can't fit within the file is corrupt — reject it
            // before the offset math below can underflow.
            if footer_len + 8 > self.size {
                return Err(Error::InvalidFooter(format!(
                    "footer length {footer_len} exceeds file size {}",
                    self.size
                )));
            }
            if footer_len + 8 > bytes.len() {
                // Footer overflowed the probe window — read it exactly.
                let exact_offset = self.size - 8 - footer_len;
                let exact = FooterRead {
                    reading_exact: true,
                    ..self
                };
                return Ok(Parsed::ExactReadNeeded(
                    exact.read_region(exact_offset, footer_len),
                ));
            }
            let start = bytes.len() - 8 - footer_len;
            &bytes[start..bytes.len() - 8]
        };

        let row_groups = row_groups_from_footer(footer, self.open_file.clone(), declared_columns)?;
        Ok(Parsed::RowGroups {
            file: self.file,
            row_groups,
        })
    }
}

/// Concatenate a completed footer request's bytes. The request asked for one
/// range, compressed-cache only, so the response is that range's raw parts.
fn region_bytes(mut ranges: Vec<Vec<RangePart>>) -> Vec<u8> {
    let parts = ranges.pop().expect("a footer request reads one range");
    debug_assert!(ranges.is_empty(), "a footer request reads one range");
    let mut out = Vec::new();
    for part in parts {
        match part {
            RangePart::Compressed { bytes, .. } => {
                for run in bytes {
                    out.extend_from_slice(&run);
                }
            }
            RangePart::Decompressed { .. } => {
                unreachable!("a compressed-only request never gets decompressed parts")
            }
        }
    }
    out
}
