//! Phase A → phase B bridge of the **staged scan** (see
//! [`scan`](super::scan) for the full topology).
//!
//! A scan with a pushed [`ScanFilter`](planner::catalog::ScanFilter) fetches in
//! two phases so that I/O for provably-dead data is never issued:
//!
//! 1. **Phase A** fetches and decodes only the *filter columns* of each row
//!    group. The [`StagingDecoder`] here terminates that phase: it assembles
//!    the filter-column batches, evaluates the compiled filter on them, and
//!    records which rows survive.
//! 2. **Phase B**: once a row group's filter columns are fully evaluated, the
//!    decoder emits one [`RowGroupRequest`] for the *full* projection —
//!    carrying the surviving row indices — back into a second
//!    [`RowGroupFetcher`](super::fetching::RowGroupFetcherFactory). A row
//!    group with **zero** survivors emits nothing: the bytes of its remaining
//!    columns are never requested.
//!
//! The phase-B request re-lists the filter columns too — their chunks were
//! just read, so the fetcher's file-cache lookup finds them resident and
//! issues no new I/O for them — which keeps phase B identical to an ordinary
//! (late-materialization-style) filtered row-group read: the regular indexer /
//! decompressor / decoder stages emit only the surviving rows, with pages that
//! hold no survivor skipped via their
//! [`FilterMask`](crate::parquet::types::filter_mask::FilterMask).
//!
//! ## Correctness
//!
//! The `Filter` operator above the scan still runs unchanged, so this is
//! purely an I/O optimization: phase B emits exactly the rows the filter
//! accepts (an all-survivors group degrades to a plain unfiltered read), and
//! re-filtering surviving rows is idempotent. Pages of one row group all
//! arrive at the worker that indexed its buffer (the decompress → decode
//! channel routes by worker), so each row group is evaluated — and its phase-B
//! request emitted — exactly once, by exactly one `StagingDecoder`.
//!
//! ## Why row groups, not pages
//!
//! Within a *surviving* row group, phase B could in principle also skip the
//! **I/O** for pages of non-filter columns whose row ranges hold no survivor
//! (their decompression and decode are already skipped via the
//! [`FilterMask`] / `SkippedData` path). That needs each page's byte range
//! *before* reading the chunk, which only the Parquet `OffsetIndex` provides —
//! and the files we target don't write one (`offset_index_offset` absent from
//! every column chunk), so page boundaries can only be learned by parsing the
//! page headers, i.e. by reading the very bytes we'd want to skip. The
//! fetcher's cache-based I/O would support it (lookups are already split into
//! 4 KB sub-blocks, so a chunk read can be any set of sub-ranges); if the
//! writer ever emits offset indexes, plumbing the surviving pages' ranges into
//! [`RowGroupRequest`] is the natural next step.
//!
//! [`FilterMask`]: crate::parquet::types::filter_mask::FilterMask

use crate::parquet::reading::decoding::{RowGroupDecoder, ScanEqualityPredicate, project_schema};
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::page::DecompressedPage;
use crate::parquet::types::projection::Projection;
use crate::parquet::types::requests::RowGroupRequest;
use crate::parquet::types::table::ParquetTable;
use ahash::HashSet;
use arrow_array::Array;
use arrow_schema::SchemaRef;
use dispatch::memory::SlabAllocator;
use dispatch::{Sender, Unary, UnaryFactory};
use planner::catalog::ScanFilterEval;
use std::sync::Arc;

/// Per-worker factory for [`StagingDecoder`]s. One per worker, like
/// [`DecoderFactory`](crate::parquet::reading::decoding::DecoderFactory) —
/// the shared `evaluator` builds each worker's private filter evaluator.
pub struct StagingDecoderFactory {
    /// Maximum rows per internally-assembled filter-column batch.
    pub batch_size: usize,
    /// Shared table metadata (schema + row groups).
    pub table: Arc<ParquetTable>,
    /// The filter columns (table indices) — what phase A fetched and what the
    /// per-group decoders here decode.
    pub filter_projection: Projection,
    /// The full scan projection (table indices) — what the emitted phase-B
    /// requests fetch.
    pub full_projection: Projection,
    /// Pushed-down equality predicates for dictionary pruning. Their columns
    /// are always filter columns, so pruning fires in phase A — a pruned row
    /// group skips not just decode but the phase-B fetch entirely.
    pub eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    /// Builds one [`ScanFilterEval`] per worker, compiled against a batch of
    /// just the filter columns in `filter_projection` order.
    pub evaluator: Arc<dyn Fn() -> ScanFilterEval + Send + Sync>,
}

impl UnaryFactory<DecompressedPage, RowGroupRequest> for StagingDecoderFactory {
    type Unary = StagingDecoder;

    fn build_unary(self) -> StagingDecoder {
        let schema = Arc::new(project_schema(self.table.schema(), &self.filter_projection));
        StagingDecoder {
            batch_size: self.batch_size,
            schema,
            filter_projection: self.filter_projection,
            full_projection: self.full_projection,
            eq_predicates: self.eq_predicates,
            eval: (self.evaluator)(),
            allocator: SlabAllocator::new(true),
            groups: Vec::new(),
            closed: HashSet::default(),
        }
    }
}

/// Filter-evaluation state for one in-flight row group.
struct StagedGroup {
    /// Decodes the group's filter columns (a plain [`RowGroupDecoder`] over
    /// the filter projection — no metadata columns, no filtered indices).
    decoder: RowGroupDecoder,
    /// Phase-A metadata; the phase-B request is a fresh view of it.
    metadata: QueryRowGroupMetadata,
    /// Row indices (within the group) that survived the filter so far.
    survivors: Vec<u32>,
    /// Rows evaluated so far — the offset of the next batch's first row.
    rows_seen: u32,
}

/// Terminal stage of phase A: decodes filter-column pages per row group,
/// evaluates the pushed row filter, and emits one phase-B [`RowGroupRequest`]
/// per surviving row group (none for a group with zero survivors).
///
/// Structured like [`Decoder`](crate::parquet::reading::decoding::Decoder)
/// (one [`RowGroupDecoder`] per in-flight group, pages inserted as they
/// arrive), but instead of sending decoded batches downstream it consumes
/// them immediately: each batch is fed to the filter evaluator and dropped,
/// so phase A holds no decoded data beyond the batch under evaluation and the
/// accumulated survivor indices.
pub struct StagingDecoder {
    batch_size: usize,
    /// Projected filter-column schema (what the evaluator sees).
    schema: SchemaRef,
    filter_projection: Projection,
    full_projection: Projection,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    /// This worker's filter evaluator over filter-column batches.
    eval: ScanFilterEval,
    /// Slab allocator for the (short-lived) decoded filter-column arrays.
    allocator: SlabAllocator,
    /// One entry per in-flight row group.
    groups: Vec<StagedGroup>,
    /// Row groups already finalized (evaluated or pruned) — late pages for
    /// these are silently dropped, mirroring `Decoder::closed_row_groups`.
    closed: HashSet<usize>,
}

impl StagingDecoder {
    /// Index into `groups` for this row group, creating the entry (decoder
    /// over the filter projection) on first sight.
    fn get_or_create_group(
        &mut self,
        metadata: &QueryRowGroupMetadata,
    ) -> dispatch::UnaryResult<usize> {
        if let Some(pos) = self
            .groups
            .iter()
            .position(|g| g.decoder.row_group_idx() == metadata.index())
        {
            return Ok(pos);
        }
        self.groups.push(StagedGroup {
            decoder: RowGroupDecoder::new(
                metadata.clone(),
                self.schema.clone(),
                &self.filter_projection,
                self.batch_size,
                false,
                &self.eq_predicates,
            )
            .map_err(crate::parquet::op_err)?,
            metadata: metadata.clone(),
            survivors: Vec::new(),
            rows_seen: 0,
        });
        Ok(self.groups.len() - 1)
    }

    /// Drain every batch the group at `pos` can currently decode through the
    /// filter evaluator; if that completes (or prunes) the group, finalize it:
    /// remove it, mark it closed, and emit its phase-B request (nothing when
    /// no row survived).
    fn evaluate_and_finalize<S: Sender<RowGroupRequest>>(
        &mut self,
        pos: usize,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        {
            let group = &mut self.groups[pos];
            // A pushed-down equality constant absent from a dictionary prunes
            // the whole group: no row can match, so no phase-B fetch either.
            if group.decoder.pruned() {
                self.closed.insert(group.decoder.row_group_idx());
                self.groups.remove(pos);
                return Ok(());
            }
            while let Some(batch) = group
                .decoder
                .try_read(&mut self.allocator)
                .map_err(crate::parquet::op_err)?
            {
                let keep = (self.eval)(&batch);
                let offset = group.rows_seen;
                group.survivors.extend(
                    (0..batch.num_rows())
                        // A null filter result drops the row, matching the
                        // upstream Filter operator's semantics.
                        .filter(|&i| keep.is_valid(i) && keep.value(i))
                        .map(|i| offset + i as u32),
                );
                group.rows_seen += batch.num_rows() as u32;
            }
            if !group.decoder.exhausted() {
                return Ok(());
            }
        }
        let group = self.groups.remove(pos);
        self.closed.insert(group.decoder.row_group_idx());
        // Zero survivors (or pruned mid-decode, which implies none — the
        // pruning predicate is one of the filter's conditions): the remaining
        // columns of this row group are never fetched — the point of staging.
        if group.decoder.pruned() || group.survivors.is_empty() {
            return Ok(());
        }
        // Everything survived: a plain, unfiltered phase-B read (no masks to
        // build or apply). Otherwise carry the survivor indices so downstream
        // decodes only those rows and skips pages holding none of them.
        let indices = (group.survivors.len() as i64) < group.metadata.num_rows();
        let metadata = group
            .metadata
            .with_filtered_indices(indices.then_some(group.survivors));
        sender.send(RowGroupRequest::from(metadata, &self.full_projection))?;
        Ok(())
    }
}

impl Unary<DecompressedPage, RowGroupRequest> for StagingDecoder {
    fn consume<S: Sender<RowGroupRequest>>(
        &mut self,
        page: DecompressedPage,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        if self.closed.contains(&page.query_row_group_metadata.index()) {
            return Ok(());
        }
        let pos = self.get_or_create_group(&page.query_row_group_metadata)?;
        self.groups[pos]
            .decoder
            .insert_page(page, &mut self.allocator);
        self.evaluate_and_finalize(pos, sender)
    }

    /// Done once every in-flight group has been finalized. Every batch a group
    /// can yield is drained on page arrival (`consume`), so by the time the
    /// upstream stages finish, only fully-evaluated (hence removed) groups
    /// remain — mirroring `Decoder::finish`.
    fn finish<S: Sender<RowGroupRequest>>(
        &mut self,
        sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        for pos in (0..self.groups.len()).rev() {
            self.evaluate_and_finalize(pos, sender)?;
        }
        Ok(self.groups.is_empty())
    }
}
