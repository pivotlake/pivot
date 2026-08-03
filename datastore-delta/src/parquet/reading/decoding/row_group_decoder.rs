//! Per-row-group decoder that accumulates decompressed pages and produces
//! Arrow [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) owns one [`RowGroupDecoder`] per
//! in-flight row group. Pages are inserted as they arrive from the
//! decompressor. When every projected column has enough buffered rows,
//! [`try_read`](RowGroupDecoder::try_read) decodes the next batch.

use crate::parquet::reading::decoding::ScanEqualityPredicate;
use crate::parquet::reading::decoding::column_decoder::{ColumnDecoder, Result};
use crate::parquet::reading::record_batch_metadata::with_row_group_metadata;
use crate::parquet::types::leaves::leaf_fields;
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::page::DecompressedPage;
use crate::parquet::types::projection::Projection;
use arrow_array::RecordBatch;
use arrow_schema::{Fields, Schema, SchemaRef};
use dispatch::memory::SlabAllocator;
use std::cmp::min;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Where a fetched leaf's pages are decoded.
///
/// A page carries the leaf's position in the row group's fetch order, which is
/// what [`projected_leaves`](crate::parquet::types::leaves::projected_leaves)
/// produces. Both sides expand the projection the same way, so that position
/// names one leaf of one column decoder.
#[derive(Clone, Copy)]
struct LeafRoute {
    /// The output column whose decoder owns this leaf.
    column: usize,
    /// The leaf's position within that column's own leaves.
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
    /// One decoder per output column, in projection order.
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
    /// One entry per fetched leaf, in fetch order, naming the decoder that
    /// consumes its pages.
    leaf_routes: Vec<LeafRoute>,
    /// Output columns that carry a pushed-down equality constant whose column
    /// chunk is sound to prune by (all data pages dictionary encoded). When any
    /// such column's dictionary excludes its constant, the whole row group is
    /// pruned.
    prunable_columns: Vec<usize>,
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
        // differ between files, so each row group resolves them independently.
        let leaves = leaf_fields(row_group_metadata.get_metadata().schema.fields());
        let filter_batches =
            !add_row_group_metadata && row_group_metadata.filtered_indices().is_none();

        let mut column_decoders = Vec::with_capacity(projection.column_indices.len());
        let mut leaf_routes = Vec::new();
        let mut prunable_columns = Vec::new();
        for (output_idx, &column) in projection.column_indices.iter().enumerate() {
            // A pushed-down variant extract emits only the referenced path.
            let decoder = match projection.extract_at(output_idx) {
                Some(extract) => ColumnDecoder::for_extract(
                    column,
                    extract,
                    &leaves,
                    &row_group_metadata,
                    eq_predicates,
                )?,
                None => {
                    ColumnDecoder::for_column(column, &leaves, &row_group_metadata, eq_predicates)?
                }
            };
            if decoder.is_prunable() {
                prunable_columns.push(output_idx);
            }
            leaf_routes.extend((0..decoder.leaf_count()).map(|leaf| LeafRoute {
                column: output_idx,
                leaf,
            }));
            column_decoders.push(decoder);
        }

        let output_fields: Fields = column_decoders
            .iter()
            .map(|decoder| decoder.output_field().clone())
            .collect();
        Ok(Self {
            row_group_idx: row_group_metadata.index(),
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
            leaf_routes,
            prunable_columns,
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
        let route = self.leaf_routes[page.column_idx];
        self.column_decoders[route.column].insert_page(route.leaf, page, allocator);
        if !self.pruned()
            && self
                .prunable_columns
                .iter()
                .any(|&column| self.column_decoders[column].dict_excludes_eq_constant())
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
            self.column_decoders
                .iter()
                .map(|decoder| decoder.available())
                .min()
                .unwrap(),
            size,
        );

        if size > 0 && available > 0 {
            let columns = self
                .column_decoders
                .iter_mut()
                .map(|decoder| decoder.read(allocator, available))
                .collect::<Result<Vec<_>>>()?;
            let record_batch = RecordBatch::try_new(self.schema.clone(), columns)?;
            // Give each column a chance to drop rows that provably fail its
            // pushed-down equality constant, before the batch travels any
            // further. Most columns leave the batch untouched; dictionary
            // encoded string columns filter it with a cheap view comparison.
            // Never done on the materializer path, which needs every row to
            // stay in place.
            let record_batch = if self.filter_batches {
                self.column_decoders
                    .iter()
                    .enumerate()
                    .fold(record_batch, |batch, (column, decoder)| {
                        decoder.fast_filter_record_batch(batch, column)
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
