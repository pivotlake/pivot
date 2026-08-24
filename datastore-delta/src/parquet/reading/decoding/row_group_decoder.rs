//! Per-row-group decoder that accumulates decompressed pages and produces
//! Arrow [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) owns one [`RowGroupDecoder`] per
//! in-flight row group. Pages are inserted as they arrive from the
//! decompressor. When every projected column has enough buffered rows,
//! [`try_read`](RowGroupDecoder::try_read) decodes the next batch.

use crate::parquet::filter_cache::FilterResultCache;
use crate::parquet::reading::decoding::ScanEqualityPredicate;
use crate::parquet::reading::decoding::column_decoder::{
    ColumnDecoder, Result, create_leaf_decoder,
};
use crate::parquet::reading::decoding::leaf_decoders::LeafDecoder;
use crate::parquet::reading::record_batch_metadata::with_row_group_metadata;
use crate::parquet::types::leaves::{leaf_fields, plan_leaves, resolve_output_reads};
use crate::parquet::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::parquet::types::page::DecompressedPage;
use crate::parquet::types::projection::Projection;
use arrow_array::{ArrayRef, BooleanArray, Datum, RecordBatch, Scalar};
use arrow_schema::{Fields, Schema, SchemaRef};
use dispatch::memory::SlabAllocator;
use std::cmp::min;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// An output column whose pushed-down equality constant can prune the row
/// group, and the decoded leaf that constant was installed on.
#[derive(Clone, Copy)]
struct PrunableColumn {
    /// The column's position in the emitted batch.
    output_idx: usize,
    /// The leaf whose dictionary decides the pruning.
    leaf: usize,
}

/// A pushed equality predicate whose outcome this decoder can prove for the
/// whole row group: its output column emits the compared leaf unchanged and
/// every row is decoded, so "no decoded row equals the constant" means the row
/// group holds no matching row at all.
struct TrackedPredicate {
    predicate: ScanEqualityPredicate,
    /// The column's position in the emitted batch.
    output_idx: usize,
    /// Whether any decoded row equaled the constant so far.
    matched: bool,
}

/// Decodes pages for a single row group into [`RecordBatch`]es.
///
/// Pages are inserted out-of-order via [`insert_page`](Self::insert_page).
/// Each call to [`try_read`](Self::try_read) checks whether all column
/// decoders have at least `batch_size` rows available and, if so, produces one
/// batch. The decoder tracks how many rows have been emitted (`row_offset`)
/// and is [`exhausted`](Self::exhausted) once all rows have been read.
pub struct RowGroupDecoder {
    /// Global row-group index (used for routing and metadata tagging).
    row_group_idx: usize,
    /// One decoder per distinct leaf column chunk, in the order the fetcher
    /// requested them, so a page's `column_idx` indexes straight into this.
    leaf_decoders: Vec<Box<dyn LeafDecoder>>,
    /// One view per output column, in projection order. Several can fold the
    /// same decoded leaf.
    column_decoders: Vec<ColumnDecoder>,
    /// Output schema (projected). A pushed-down extract column carries the
    /// type the extract emits, not the variant it was read from.
    schema: SchemaRef,
    /// Max rows per batch.
    batch_size: usize,
    /// Total rows to emit (filtered count, or full row-group count).
    total: usize,
    /// Rows emitted so far.
    row_offset: usize,
    /// Whether to append row-group-id / row-index metadata columns.
    add_row_group_metadata: bool,
    /// The output columns carrying a pushed-down equality constant whose chunk
    /// is sound to prune by (all data pages dictionary encoded), each with the
    /// leaf holding that constant. When any such leaf's dictionary excludes it,
    /// the whole row group is pruned.
    prunable: Vec<PrunableColumn>,
    /// The row group's shared pruned flag is the same `Arc` every page of this
    /// row group carries. It is set here when a prunable column's dictionary
    /// excludes its constant: the row group cannot contain a matching row, so
    /// it emits nothing and is treated as exhausted. The decompressor reads it
    /// so it can skip the row group's remaining, not-yet-decompressed pages
    /// instead of decompressing them only for this decoder to discard.
    pruned: Arc<AtomicBool>,
    /// Whether emitted batches may drop rows that fail a pushed-down
    /// equality constant. The materializer path must not: it addresses rows
    /// by their position inside the row group, so every row has to stay in
    /// place.
    filter_batches: bool,
    /// The pushed equality predicates whose outcome this full-row-group decode
    /// proves (empty when only a subset of rows is read); recorded into
    /// `filter_cache` so a later scan can skip the row group.
    tracked_predicates: Vec<TrackedPredicate>,
    /// Whether any single decoded row satisfied *every* tracked predicate at
    /// once. Only evaluated with two or more tracked predicates: the recorded
    /// conjunction lets a later scan skip the row group even when each
    /// predicate matches some row on its own.
    conjunction_matched: bool,
    /// The row group's shared filter-outcome memory, off its metadata.
    filter_cache: Arc<FilterResultCache>,
}

impl RowGroupDecoder {
    pub fn new(
        row_group_metadata: QueryRowGroupMetadata,
        projection: &Projection,
        batch_size: usize,
        add_row_group_metadata: bool,
        eq_predicates: &[ScanEqualityPredicate],
    ) -> Result<Self> {
        let pruned = row_group_metadata.pruned_flag();
        // Expand projected columns into this file's leaves. Variant layouts can
        // differ between files, so each row group resolves them independently,
        // and the fetcher resolved them the same way.
        let fields = row_group_metadata.get_metadata().schema.fields();
        let leaves = leaf_fields(fields);
        let reads = resolve_output_reads(fields, &row_group_metadata, projection);
        let plan = plan_leaves(&reads);
        let filter_batches =
            !add_row_group_metadata && row_group_metadata.filtered_indices().is_none();

        let column_chunks = row_group_metadata.columns();
        let mut leaf_decoders = plan
            .file_leaves
            .iter()
            .map(|&leaf| create_leaf_decoder(leaves[leaf].data_type(), &column_chunks[leaf]))
            .collect::<Result<Vec<_>>>()?;

        let mut column_decoders = Vec::with_capacity(projection.column_indices.len());
        for ((output_idx, &column), positions) in projection
            .column_indices
            .iter()
            .enumerate()
            .zip(plan.output_positions)
        {
            column_decoders.push(ColumnDecoder::new(
                column,
                projection.extract_at(output_idx),
                &reads[output_idx],
                positions,
                &leaves,
                fields,
            )?);
        }

        let prunable = install_eq_constants(
            &column_decoders,
            &mut leaf_decoders,
            &plan.file_leaves,
            column_chunks,
            projection,
            eq_predicates,
        );

        // A filtered read decodes only some rows, which can't prove a value
        // absent from the whole row group, so nothing is tracked.
        let tracked_predicates = if row_group_metadata.filtered_indices().is_none() {
            trackable_predicates(&column_decoders, projection, eq_predicates)
        } else {
            Vec::new()
        };
        let filter_cache = row_group_metadata.get_metadata().filter_cache.clone();

        let output_fields: Fields = column_decoders
            .iter()
            .map(|decoder| decoder.output_field().clone())
            .collect();
        Ok(Self {
            row_group_idx: row_group_metadata.index(),
            leaf_decoders,
            column_decoders,
            schema: Arc::new(Schema::new(output_fields)),
            batch_size,
            total: row_group_metadata
                .filtered_indices()
                .as_ref()
                .map(|f| f.len())
                .unwrap_or(row_group_metadata.num_rows() as usize),
            row_offset: 0,
            add_row_group_metadata,
            prunable,
            pruned,
            filter_batches,
            tracked_predicates,
            conjunction_matched: false,
            filter_cache,
        })
    }

    /// Returns the global row-group index this decoder is responsible for.
    pub fn row_group_idx(&self) -> usize {
        self.row_group_idx
    }

    /// Returns `true` when the row group has been pruned (no row can match a
    /// pushed-down equality predicate) or all its rows have been emitted.
    pub fn exhausted(&self) -> bool {
        self.pruned() || self.total - self.row_offset == 0
    }

    /// Returns `true` if this row group was pruned by dictionary pushdown and
    /// should be dropped without emitting any rows.
    pub fn pruned(&self) -> bool {
        self.pruned.load(Ordering::Relaxed)
    }

    /// Routes a decompressed page to the appropriate column decoder, then
    /// re-evaluates dictionary pruning (a just-loaded dictionary page may
    /// exclude a pushed-down constant, allowing the whole row group to be
    /// dropped before its data pages are decoded).
    pub fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator) {
        self.leaf_decoders[page.column_idx].insert_page(page, allocator);
        if !self.pruned()
            && self
                .prunable
                .iter()
                .any(|p| self.leaf_decoders[p.leaf].dict_excludes_eq_constant())
        {
            // Publish to the shared flag (seen by every page of this row group):
            // the decoder discards the rest, and the decompressor can skip the
            // row group's remaining, not-yet-decompressed pages.
            self.pruned.store(true, Ordering::Relaxed);
            // An excluding dictionary covers every row of the chunk, so it
            // proves the predicate matches no row even though nothing decodes.
            for tracked in &self.tracked_predicates {
                let excluded = self.prunable.iter().any(|p| {
                    p.output_idx == tracked.output_idx
                        && self.leaf_decoders[p.leaf].dict_excludes_eq_constant()
                });
                if excluded {
                    self.filter_cache.record(&[&tracked.predicate], false);
                }
            }
        }
    }

    /// Attempts to produce the next [`RecordBatch`].
    ///
    /// Returns `Ok(Some(batch))` if every column decoder has enough buffered
    /// rows, `Ok(None)` if more pages are needed, or an error if decoding
    /// fails.
    pub fn try_read(&mut self, allocator: &mut SlabAllocator) -> Result<Option<RecordBatch>> {
        if self.pruned() {
            return Ok(None);
        }
        let size = min(self.batch_size, self.total - self.row_offset);
        let available = min(
            self.leaf_decoders
                .iter()
                .map(|leaf| leaf.available())
                .min()
                .unwrap(),
            size,
        );

        if size > 0 && available > 0 {
            // Every distinct leaf is decoded once; the output columns then fold
            // their own views of the result, so two extracts over one variant
            // column share the decode instead of repeating it.
            let decoded = self
                .leaf_decoders
                .iter_mut()
                .map(|leaf| leaf.read(allocator, available).map_err(Into::into))
                .collect::<Result<Vec<_>>>()?;
            let columns = self
                .column_decoders
                .iter()
                .map(|decoder| decoder.read(&decoded))
                .collect::<Result<Vec<_>>>()?;
            self.evaluate_tracked_predicates(&columns);
            let record_batch = RecordBatch::try_new(self.schema.clone(), columns)?;
            // Give each column carrying a pushed-down equality constant a
            // chance to drop rows that provably fail it, before the batch
            // travels any further. Dictionary encoded string columns filter it
            // with a cheap view comparison; the rest leave it untouched. Never
            // done on the materializer path, which needs every row to stay in
            // place.
            let record_batch = if self.filter_batches {
                self.prunable.iter().fold(record_batch, |batch, p| {
                    self.leaf_decoders[p.leaf].fast_filter_record_batch(batch, p.output_idx)
                })
            } else {
                record_batch
            };
            let batch = if self.add_row_group_metadata {
                with_row_group_metadata(record_batch, self.row_group_idx, self.row_offset)
            } else {
                record_batch
            };
            self.row_offset += available;
            if self.row_offset == self.total {
                self.record_tracked_outcomes();
            }
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    /// Check each not-yet-matched tracked predicate against this batch's
    /// decoded columns, and (with several tracked predicates) whether any row
    /// satisfies all of them at once. A predicate's first match is recorded to
    /// the row group's cache immediately, so even a scan abandoned early (a
    /// LIMIT) remembers what it proved; "matched no row" waits for the full
    /// decode in [`record_tracked_outcomes`](Self::record_tracked_outcomes).
    fn evaluate_tracked_predicates(&mut self, columns: &[ArrayRef]) {
        let need_conjunction = self.tracked_predicates.len() > 1 && !self.conjunction_matched;
        if !need_conjunction && self.tracked_predicates.iter().all(|t| t.matched) {
            return;
        }
        let mut all_satisfied: Option<BooleanArray> = None;
        for tracked in &mut self.tracked_predicates {
            if tracked.matched && !need_conjunction {
                continue;
            }
            let matches =
                rows_equal_to_constant(&columns[tracked.output_idx], &tracked.predicate.value);
            if !tracked.matched && matches.true_count() > 0 {
                tracked.matched = true;
                self.filter_cache.record(&[&tracked.predicate], true);
            }
            if need_conjunction {
                all_satisfied = Some(match all_satisfied {
                    None => matches,
                    Some(so_far) => arrow_arith::boolean::and(&so_far, &matches)
                        .expect("equal-length match masks over one batch"),
                });
            }
        }
        if all_satisfied.is_some_and(|rows| rows.true_count() > 0) {
            self.conjunction_matched = true;
        }
    }

    /// The whole row group decoded: every tracked predicate's outcome is now
    /// proven, so remember the misses (matches were recorded as they were
    /// found). The conjunction adds information only when every predicate
    /// matched on its own; any individually-false predicate is the stronger
    /// fact, pruning every query that pushes it.
    fn record_tracked_outcomes(&self) {
        for tracked in &self.tracked_predicates {
            if !tracked.matched {
                self.filter_cache.record(&[&tracked.predicate], false);
            }
        }
        if self.tracked_predicates.len() > 1
            && !self.conjunction_matched
            && self.tracked_predicates.iter().all(|t| t.matched)
        {
            let predicates: Vec<&ScanEqualityPredicate> = self
                .tracked_predicates
                .iter()
                .map(|t| &t.predicate)
                .collect();
            self.filter_cache.record(&predicates, false);
        }
    }
}

/// The rows of `column` equal to `constant`, as a boolean mask (a NULL row is
/// never equal). The caller verified both sides share a type, and every leaf
/// type the decoder emits is comparable, so the kernel cannot fail.
fn rows_equal_to_constant(column: &ArrayRef, constant: &Scalar<ArrayRef>) -> BooleanArray {
    arrow_ord::cmp::eq(&column.as_ref() as &dyn Datum, constant as &dyn Datum)
        .expect("comparison of same-typed leaf column and constant")
}

/// The pushed equality predicates whose outcome this decode can prove: each
/// must read its compared leaf unchanged into the emitted batch (same shape
/// rule as [`install_eq_constants`]) and its constant must share the emitted
/// column's type so the comparison kernel applies. A NULL constant is skipped;
/// it equals nothing and the cache would not remember it anyway.
fn trackable_predicates(
    column_decoders: &[ColumnDecoder],
    projection: &Projection,
    eq_predicates: &[ScanEqualityPredicate],
) -> Vec<TrackedPredicate> {
    let mut tracked = Vec::new();
    for (output_idx, &column) in projection.column_indices.iter().enumerate() {
        if column_decoders[output_idx].untransformed_leaf().is_none() {
            continue;
        }
        let path = projection
            .extract_at(output_idx)
            .map(|extract| extract.path.as_slice())
            .unwrap_or_default();
        let Some(predicate) = eq_predicates
            .iter()
            .find(|p| p.column_idx == column && p.path == path)
        else {
            continue;
        };
        let (constant, _) = predicate.value.get();
        if constant.len() != 1 || constant.is_null(0) {
            continue;
        }
        if column_decoders[output_idx].output_field().data_type() != constant.data_type() {
            continue;
        }
        tracked.push(TrackedPredicate {
            predicate: predicate.clone(),
            output_idx,
            matched: false,
        });
    }
    tracked
}

/// Installs each pushed-down equality constant on the leaf that answers it, and
/// returns the columns the row group can then be pruned by.
///
/// A constant only goes in when its output column emits exactly one decoded
/// leaf unchanged, because that is the only shape both uses of the constant can
/// read: the row group is pruned from the leaf's own dictionary, and the batch
/// filter compares the emitted array against the dictionary's view of the
/// constant. A column that casts its leaf or reconstructs a variant emits
/// something else entirely.
///
/// It also goes in only when every data page of that chunk is dictionary
/// encoded. The decoder uses the constant to skip building a dictionary that
/// excludes it, which is sound only when an excluded dictionary prunes the whole
/// row group. A PLAIN fallback page could hold the constant even if the
/// dictionary does not, so such a chunk is still scanned and its dictionary must
/// be built to decode it.
///
/// A constant whose type does not match the leaf is dropped by the leaf decoder,
/// which forgoes the pushdown; the query's `Filter` still applies the
/// comparison.
fn install_eq_constants(
    column_decoders: &[ColumnDecoder],
    leaf_decoders: &mut [Box<dyn LeafDecoder>],
    file_leaves: &[usize],
    column_chunks: &[ColumnChunkMeta],
    projection: &Projection,
    eq_predicates: &[ScanEqualityPredicate],
) -> Vec<PrunableColumn> {
    let mut prunable = Vec::new();
    for (output_idx, &column) in projection.column_indices.iter().enumerate() {
        let Some(leaf) = column_decoders[output_idx].untransformed_leaf() else {
            continue;
        };
        // A whole-column read answers a comparison on the column itself; a
        // pushed extract answers one that names the very path it reads.
        let path = projection
            .extract_at(output_idx)
            .map(|extract| extract.path.as_slice())
            .unwrap_or_default();
        let Some(predicate) = eq_predicates
            .iter()
            .find(|p| p.column_idx == column && p.path == path)
        else {
            continue;
        };
        if !column_chunks[file_leaves[leaf]].data_pages_all_dictionary {
            continue;
        }
        leaf_decoders[leaf].set_eq_constant(&predicate.value);
        prunable.push(PrunableColumn { output_idx, leaf });
    }
    prunable
}
