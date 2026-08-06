//! Per-row-group decoder that accumulates decompressed pages and produces
//! Arrow [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) owns one [`RowGroupDecoder`] per
//! in-flight row group. Pages are inserted as they arrive from the
//! decompressor. When every projected column has enough buffered rows,
//! [`try_read`](RowGroupDecoder::try_read) decodes the next batch.

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
use arrow_array::RecordBatch;
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
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }
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
