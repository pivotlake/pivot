//! The [`DictPrefetcher`]: a pre-fetch stage that reads dictionary pages for
//! many row groups at once, prunes by equality pushdown, and only then forwards
//! the survivors to the [`RowGroupFetcher`](super::fetcher::RowGroupFetcher) for
//! the (one-at-a-time) full data read.
//!
//! For a high-cardinality equality point-lookup (`WHERE col = const`), min/max
//! statistics barely prune (every row group's range spans the constant), so the
//! win comes from dictionary pruning: a row group whose dictionary excludes the
//! constant can hold no matching row. The plain scan reads each survivor's
//! *entire* column chunk (dictionary page + all data pages) and only discovers
//! the dictionary excludes the constant after the data pages are already on
//! disk — wasting that I/O on cold reads.
//!
//! This stage splits the read in two:
//!
//! 1. **Dictionary phase (here, many at once).** For each row group with a
//!    prunable projected equality column, read just that column's dictionary
//!    page. Many row groups' dictionary reads are kept in flight together
//!    ([`MAX_DICT_IN_FLIGHT`]) — they're small, so this saturates the disk
//!    without the per-row-group serialization the data fetcher uses. When the
//!    bytes land, decompress the dictionary and scan it for the constant.
//! 2. **Data phase (downstream fetcher, one at a time).** A row group whose
//!    dictionary *excludes* the constant is dropped — its data pages are never
//!    read. A survivor is forwarded as a full [`RowGroupRequest`], built *now*
//!    so the dictionary blocks this stage just cached are skipped and only the
//!    data pages are read.
//!
//! Row groups without a prunable column (no equality predicate, no dictionary,
//! or PLAIN-fallback data pages) are forwarded straight through as full
//! requests, exactly as the plain scan would issue them.

use crate::parquet::reading::ScanEqualityPredicate;
use crate::parquet::reading::decoding::dictionary_excludes_constant;
use crate::parquet::reading::decompressor::Decompressor;
use crate::parquet::reading::indexer::create_compressed_pages;
use crate::parquet::request_tracker::{PendingRequest, ReadRequest, RequestTracker};
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::projection::Projection;
use crate::parquet::types::table::ParquetTable;
use crate::parquet::types::thrift::general::PageType;
use crate::parquet::RowGroupRequest;
use arrow_array::{ArrayRef, Scalar};
use arrow_schema::DataType;
use bytes::Bytes;
use dispatch::io::{FileLocation, FsRequest, HttpRequest};
use dispatch::memory::{CacheLookup, SlabAllocator, memory_ctx};
use dispatch::{Sender, Unary, UnaryFactory};
use std::sync::Arc;

/// Dictionary reads kept in flight per worker before admitting the next row
/// group's probe. Larger than the data fetcher's cap (1): dictionary pages are
/// small, so issuing many concurrently saturates the disk and hides latency —
/// this is the "load many dictionary pages at once" phase.
const MAX_DICT_IN_FLIGHT: usize = 32;
/// HTTP-backed dictionary reads in flight per worker (remote objects).
const MAX_DICT_HTTP_IN_FLIGHT: usize = 32;

/// Whether dictionary pruning can actually decide membership for `data_type`.
/// Only the fixed-width primitive decoders implement the raw-dictionary scan
/// ([`Dict::contains`](crate::parquet::reading::decoding::column_decoders)); the
/// bytes/view decoders fall back to the trait default that reports every value
/// "present", so probing them would read each dictionary page only to never
/// prune — pure overhead. Restricting the prefetch to these types keeps it to
/// the cases where it can pay off; everything else is forwarded unprobed,
/// exactly as the plain scan would read it.
fn supports_dict_pruning(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::UInt16
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Tracks the IO for one row group's dictionary-page probe: the predicate
/// column's `[dictionary_page_offset, data_page_offset)` bytes, plus what we
/// need to evaluate the equality pushdown once they land.
struct DictProbe {
    /// The row group being probed — forwarded as a full request if it survives.
    metadata: QueryRowGroupMetadata,
    /// The pushed-down equality constant to scan the dictionary for.
    value: Scalar<ArrayRef>,
    /// Arrow type of the predicate column (selects the typed dictionary scan).
    data_type: DataType,
    /// Max definition level of the predicate column's chunk.
    max_def_level: i16,
    /// Cache lookups covering the dictionary page's byte range.
    parts: Vec<CacheLookup>,
    pending_fs: Vec<FsRequest>,
    pending_http: Vec<HttpRequest>,
    /// Outstanding reads before the dictionary bytes are all resident.
    remaining: usize,
}

impl DictProbe {
    fn complete_one(&mut self) {
        self.remaining -= 1;
    }

    fn complete(&self) -> bool {
        self.remaining == 0
    }
}

impl PendingRequest for DictProbe {
    fn pending_fs(&mut self) -> &mut Vec<FsRequest> {
        &mut self.pending_fs
    }

    fn pending_http(&mut self) -> &mut Vec<HttpRequest> {
        &mut self.pending_http
    }
}

/// Factory producing one [`DictPrefetcher`] per worker.
pub struct DictPrefetcherFactory {
    table: Arc<ParquetTable>,
    projection: Projection,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
}

impl DictPrefetcherFactory {
    pub fn new(
        table: Arc<ParquetTable>,
        projection: Projection,
        eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    ) -> Self {
        Self {
            table,
            projection,
            eq_predicates,
        }
    }
}

impl UnaryFactory<QueryRowGroupMetadata, RowGroupRequest> for DictPrefetcherFactory {
    type Unary = DictPrefetcher;

    fn build_unary(self) -> DictPrefetcher {
        DictPrefetcher {
            table: self.table,
            projection: self.projection,
            eq_predicates: self.eq_predicates,
            tracker: RequestTracker::default(),
            decompressor: Decompressor::default(),
            allocator: SlabAllocator::new(true),
            dbg_pruned: 0,
            dbg_forwarded_after_probe: 0,
            dbg_passthrough: 0,
        }
    }
}

/// See the [module docs](self). Consumes row-group metadata, prunes by
/// dictionary contents, and emits surviving row groups as full requests.
pub struct DictPrefetcher {
    table: Arc<ParquetTable>,
    projection: Projection,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    tracker: RequestTracker<DictProbe>,
    decompressor: Decompressor,
    allocator: SlabAllocator,
    /// Debug counters (gated on PIVOT_DEBUG_PREFETCH): row groups pruned by
    /// dictionary, forwarded after probing, and forwarded without a probe.
    dbg_pruned: usize,
    dbg_forwarded_after_probe: usize,
    dbg_passthrough: usize,
}

impl DictPrefetcher {
    /// The projected equality-predicate column whose dictionary we can probe to
    /// prune this row group, or `None` if there's no such column (so the row
    /// group is forwarded as a full request unchanged). Requires the column to
    /// be projected, of a type whose dictionary scan can decide membership
    /// ([`supports_dict_pruning`]), dictionary-encoded, and with all data pages
    /// dictionary encoded — the same soundness condition the decoder prunes
    /// under.
    fn prunable_predicate(&self, metadata: &QueryRowGroupMetadata) -> Option<&ScanEqualityPredicate> {
        let columns = metadata.columns();
        let schema = self.table.schema();
        self.eq_predicates.iter().find(|p| {
            self.projection.indices().contains(&p.column_idx)
                && supports_dict_pruning(schema.field(p.column_idx).data_type())
                && columns
                    .get(p.column_idx)
                    .is_some_and(|c| c.data_pages_all_dictionary && c.dictionary_page_offset.is_some())
        })
    }

    /// Forward a row group as a full request (built now, so any already-cached
    /// dictionary blocks are skipped and only the missing data pages are read).
    fn forward<S: Sender<RowGroupRequest>>(
        &self,
        metadata: QueryRowGroupMetadata,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        sender.send(RowGroupRequest::from(metadata, &self.projection))?;
        Ok(())
    }

    /// Decide a completed probe: drop the row group if its dictionary excludes
    /// the constant, otherwise forward it as a full request. Any parse or
    /// decompress failure falls back to forwarding (always sound — the normal
    /// decode path handles it).
    fn resolve_probe<S: Sender<RowGroupRequest>>(
        &mut self,
        probe: DictProbe,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let DictProbe {
            metadata,
            value,
            data_type,
            max_def_level,
            parts,
            ..
        } = probe;
        // Dictionary page bytes, in file order (valid now: every block landed).
        let buffers: Vec<Bytes> = parts.into_iter().map(|p| p.into_data()).collect();

        let pruned = self.dictionary_prunes(&metadata, &value, &data_type, max_def_level, buffers);
        if pruned {
            // Dictionary excludes the constant: no row in this group can match,
            // so its data pages are never read.
            self.dbg_pruned += 1;
            return Ok(());
        }
        self.dbg_forwarded_after_probe += 1;
        self.forward(metadata, sender)
    }

    /// Parse + decompress the dictionary page and scan it for the constant.
    /// Returns `true` only when the dictionary provably excludes it.
    fn dictionary_prunes(
        &mut self,
        metadata: &QueryRowGroupMetadata,
        value: &Scalar<ArrayRef>,
        data_type: &DataType,
        max_def_level: i16,
        buffers: Vec<Bytes>,
    ) -> bool {
        let Ok(pages) = create_compressed_pages(0, metadata.clone(), &buffers) else {
            return false;
        };
        let Some(dict) = pages
            .into_iter()
            .find(|p| p.header.r#type == PageType::DICTIONARY_PAGE)
        else {
            return false;
        };
        let Ok(decompressed) = self.decompressor.decompress(dict) else {
            return false;
        };
        dictionary_excludes_constant(data_type, max_def_level, value, decompressed, &mut self.allocator)
    }

    /// A landed read: credit every probe waiting on it, then resolve any that
    /// now have their full dictionary. Mirrors the row-group fetcher's delivery.
    fn deliver<S: Sender<RowGroupRequest>>(
        &mut self,
        read: ReadRequest,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let waiters = self.tracker.complete(&read);
        for &slot in &waiters {
            self.tracker.request_for_slot(slot).unwrap().complete_one();
        }
        for slot in waiters {
            if self
                .tracker
                .request_for_slot(slot)
                .is_some_and(|p| p.complete())
            {
                let probe = self.tracker.take_request_at_slot(slot);
                self.resolve_probe(probe, sender)?;
            }
        }
        Ok(())
    }
}

impl Unary<QueryRowGroupMetadata, RowGroupRequest> for DictPrefetcher {
    fn consume<S: Sender<RowGroupRequest>>(
        &mut self,
        metadata: QueryRowGroupMetadata,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let Some(predicate) = self.prunable_predicate(&metadata) else {
            // No prunable column: forward straight through, exactly as the plain
            // scan would.
            self.dbg_passthrough += 1;
            return self.forward(metadata, sender);
        };

        let col = &metadata.columns()[predicate.column_idx];
        let dict_offset = col.dictionary_page_offset.unwrap() as usize;
        let dict_len = col.data_page_offset as usize - dict_offset;
        let location = metadata.get_metadata().location.clone();
        let value = predicate.value.clone();
        let data_type = self
            .table
            .schema()
            .field(predicate.column_idx)
            .data_type()
            .clone();
        let max_def_level = col.max_def_level;

        let parts = memory_ctx().file_cache().get(&location, dict_offset, dict_len);
        let mut pending_fs = vec![];
        let mut pending_http = vec![];
        for lookup in &parts {
            for block in lookup.missing() {
                match &location {
                    FileLocation::Local(file) => pending_fs.push(FsRequest {
                        file: file.clone(),
                        block: block.clone(),
                    }),
                    FileLocation::Remote(remote) => pending_http.push(HttpRequest {
                        remote: remote.clone(),
                        block: block.clone(),
                    }),
                }
            }
        }

        let probe = DictProbe {
            metadata,
            value,
            data_type,
            max_def_level,
            remaining: pending_fs.len() + pending_http.len(),
            parts,
            pending_fs,
            pending_http,
        };

        let slot = self.tracker.admit_request(probe);
        // A fully-cached dictionary (hot path) has no pending reads: resolve it
        // immediately rather than waiting on a completion that never comes.
        if self
            .tracker
            .request_for_slot(slot)
            .is_some_and(|p| p.complete())
        {
            let probe = self.tracker.take_request_at_slot(slot);
            self.resolve_probe(probe, sender)?;
        }
        Ok(())
    }

    fn next_fs_requests(&mut self) -> dispatch::UnaryResult<Vec<FsRequest>> {
        Ok(self.tracker.take_fs_requests())
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        Ok(self.tracker.take_http_requests())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.tracker.disk_in_flight() < MAX_DICT_IN_FLIGHT
            && self.tracker.http_in_flight() < MAX_DICT_HTTP_IN_FLIGHT
    }

    fn process_fs_response<S: Sender<RowGroupRequest>>(
        &mut self,
        sender: &mut S,
        request: FsRequest,
    ) -> dispatch::UnaryResult<()> {
        self.deliver(ReadRequest::of_fs(&request), sender)
    }

    fn process_http_response<S: Sender<RowGroupRequest>>(
        &mut self,
        sender: &mut S,
        request: HttpRequest,
    ) -> dispatch::UnaryResult<()> {
        self.deliver(ReadRequest::of_http(&request), sender)
    }

    fn finish<S: Sender<RowGroupRequest>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        // Done once every admitted probe has drained (its survivor forwarded or
        // its row group pruned).
        let idle = self.tracker.is_idle();
        if idle
            && (self.dbg_pruned + self.dbg_forwarded_after_probe + self.dbg_passthrough) > 0
            && std::env::var_os("PIVOT_DEBUG_PREFETCH").is_some()
        {
            eprintln!(
                "[dict-prefetch] pruned={} forwarded_after_probe={} passthrough={}",
                self.dbg_pruned, self.dbg_forwarded_after_probe, self.dbg_passthrough
            );
            self.dbg_pruned = 0;
            self.dbg_forwarded_after_probe = 0;
            self.dbg_passthrough = 0;
        }
        Ok(idle)
    }
}
