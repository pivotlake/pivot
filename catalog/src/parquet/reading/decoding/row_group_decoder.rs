//! Per-row-group decoder that accumulates decompressed pages and produces
//! Arrow [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) owns one [`RowGroupDecoder`] per
//! in-flight row group. Pages are inserted as they arrive from the
//! decompressor. When every projected column has enough buffered rows,
//! [`try_read`](RowGroupDecoder::try_read) decodes the next batch.

use crate::parquet::reading::decoding::ScanEqualityPredicate;
use crate::parquet::reading::decoding::column_decoders;
use crate::parquet::reading::decoding::column_decoders::{
    BytesViewDecoder, ColumnDecoder, PrimitiveColumnDecoder, decimal_decoder,
};
use crate::parquet::reading::record_batch_metadata::with_row_group_metadata;
use crate::parquet::types::leaves::{
    first_leaf, leaf_count, leaf_fields, nest_leaves_into_columns,
};
use crate::parquet::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::parquet::types::page::DecompressedPage;
use crate::parquet::types::projection::Projection;
use arrow_array::RecordBatch;
use arrow_array::types::{
    BinaryViewType, Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int16Type,
    Int32Type, Int64Type, StringViewType, TimestampSecondType, UInt16Type,
};
use arrow_schema::{ArrowError, DataType, Fields, Schema, SchemaRef, TimeUnit};
use dispatch::memory::SlabAllocator;
use std::cmp::min;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Unsupported column type: {0:?}")]
    UnsupportedColumnType(DataType),
    #[error("{0}")]
    ColumnDecoder(#[from] column_decoders::Error),
    #[error("{0}")]
    Arrow(#[from] ArrowError),
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// Get a column decoder for a given data type. The chunk metadata
/// disambiguates a decimal's physical storage.
fn column_decoder_for_type(
    data_type: &DataType,
    chunk: &ColumnChunkMeta,
) -> Result<Box<dyn ColumnDecoder>> {
    let max_def_level = chunk.max_def_level;
    macro_rules! primitive {
        ($t:ty) => {
            Box::new(PrimitiveColumnDecoder::<$t>::new(max_def_level)) as Box<dyn ColumnDecoder>
        };
    }
    match data_type {
        DataType::UInt16 => Ok(primitive!(UInt16Type)),
        DataType::Int16 => Ok(primitive!(Int16Type)),
        DataType::Int32 => Ok(primitive!(Int32Type)),
        DataType::Int64 => Ok(primitive!(Int64Type)),
        DataType::Date32 => Ok(primitive!(Date32Type)),
        DataType::Timestamp(TimeUnit::Second, None) => Ok(primitive!(TimestampSecondType)),
        DataType::Float32 => Ok(primitive!(Float32Type)),
        DataType::Float64 => Ok(primitive!(Float64Type)),
        DataType::Decimal64(precision, scale) => {
            Ok(decimal_decoder::<Decimal64Type>(chunk, *precision, *scale)?)
        }
        DataType::Decimal128(precision, scale) => Ok(decimal_decoder::<Decimal128Type>(
            chunk, *precision, *scale,
        )?),
        // The byte-view decoder's flavour matches the leaf's declared type, so
        // the finished column is a string or binary array directly.
        DataType::Utf8View | DataType::Utf8 | DataType::LargeUtf8 => Ok(Box::new(
            BytesViewDecoder::<StringViewType>::new(max_def_level),
        )),
        DataType::BinaryView | DataType::Binary | DataType::LargeBinary => Ok(Box::new(
            BytesViewDecoder::<BinaryViewType>::new(max_def_level),
        )),
        other => Err(Error::UnsupportedColumnType(other.clone())),
    }
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
    /// One decoder per projected column, in projection order.
    column_decoders: Vec<Box<dyn ColumnDecoder>>,
    /// Output schema (projected).
    schema: SchemaRef,
    /// Max rows per batch.
    batch_size: usize,
    /// Total rows to emit (filtered count, or full row-group count).
    total: usize,
    /// Rows emitted so far.
    row_offset: usize,
    /// Whether to append row-group-id / row-index metadata columns.
    add_row_group_metadata: bool,
    /// Projected-column positions that carry a pushed-down equality constant
    /// whose column chunk is sound to prune by (all data pages dictionary
    /// encoded). When any such column's dictionary excludes its constant, the
    /// whole row group is pruned.
    prunable_columns: Vec<usize>,
    /// Set once a `prunable_columns` entry's dictionary is found to exclude
    /// its constant: the row group cannot contain a matching row, so it emits
    /// nothing and is treated as exhausted.
    /// The row group's shared pruned flag — the same `Arc` every page of this
    /// row group carries. Set here when a prunable column's dictionary excludes
    /// its constant; read by the decompressor so it can skip the row group's
    /// remaining, not-yet-decompressed pages instead of decompressing them only
    /// for this decoder to discard.
    pruned: Arc<AtomicBool>,
    /// The batch column each decoder's output lands in, one entry per
    /// decoder. Multi-leaf (variant) columns fold several decoders into the
    /// same batch column.
    decoder_output_columns: Vec<usize>,
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
        let columns = row_group_metadata.columns();
        // Expand projected columns into this file's leaves. Variant layouts can
        // differ between files, so each row group resolves them independently.
        let fields = row_group_metadata.get_metadata().schema.fields();
        let leaves = leaf_fields(fields);

        let mut column_decoders = Vec::new();
        let mut prunable_columns = Vec::new();
        let mut decoder_output_columns = Vec::new();
        let filter_batches =
            !add_row_group_metadata && row_group_metadata.filtered_indices().is_none();
        let mut output_fields = Vec::with_capacity(projection.column_indices.len());
        for (output_idx, &column) in projection.column_indices.iter().enumerate() {
            output_fields.push(fields[column].clone());
            let start = first_leaf(fields, column);
            let count = leaf_count(&fields[column]);
            // Equality pushdown applies to a scalar (single-leaf) column only.
            let predicate = (count == 1)
                .then(|| eq_predicates.iter().find(|p| p.column_idx == column))
                .flatten();
            for leaf in start..start + count {
                decoder_output_columns.push(output_idx);
                // A column chunk can only be pruned (or batch-filtered) by
                // dictionary contents when every one of its data pages is
                // dictionary encoded; a PLAIN fallback page could hold the
                // constant even if the dictionary does not.
                let prunable = predicate.is_some() && columns[leaf].data_pages_all_dictionary;
                if prunable {
                    prunable_columns.push(column_decoders.len());
                }
                let mut decoder =
                    column_decoder_for_type(leaves[leaf].data_type(), &columns[leaf])?;
                // Install the equality constant only when the column is prunable.
                // The decoder uses it to skip building a dictionary that excludes
                // the constant, which is sound only when an excluded dictionary
                // prunes the whole row group. On a non-prunable column (e.g.
                // PLAIN fallback data pages) the row group is still scanned, so
                // the dictionary must be built to decode it.
                if prunable && let Some(p) = predicate {
                    decoder.set_eq_constant(&p.value);
                }
                column_decoders.push(decoder);
            }
        }

        Ok(Self {
            row_group_idx: row_group_metadata.index(),
            column_decoders,
            schema: Arc::new(Schema::new(Fields::from(output_fields))),
            batch_size,
            total: row_group_metadata
                .filtered_indices()
                .as_ref()
                .map(|f| f.len())
                .unwrap_or(row_group_metadata.num_rows() as usize),
            row_offset: 0,
            add_row_group_metadata,
            prunable_columns,
            pruned,
            decoder_output_columns,
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
        self.column_decoders[page.column_idx].insert_page(page, allocator);
        if !self.pruned()
            && self
                .prunable_columns
                .iter()
                .any(|&pos| self.column_decoders[pos].dict_excludes_eq_constant())
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
                .map(|c| c.available())
                .min()
                .unwrap(),
            size,
        );

        if size > 0 && available > 0 {
            let leaf_arrays = self
                .column_decoders
                .iter_mut()
                .map(|c| c.read(allocator, available).map_err(Error::from))
                .collect::<Result<Vec<_>>>()?;
            // Fold the decoded leaf arrays back under their struct parents to
            // match the (possibly nested) output schema.
            let columns =
                nest_leaves_into_columns(self.schema.fields(), &mut leaf_arrays.into_iter());
            let record_batch = RecordBatch::try_new(self.schema.clone(), columns)?;
            // Give each decoder a chance to drop rows that provably fail its
            // pushed-down equality constant, before the batch travels any
            // further. Most decoders leave the batch untouched; dictionary
            // encoded string columns filter it with a cheap view comparison.
            // Never done on the materializer path, which needs every row to
            // stay in place.
            let record_batch = if self.filter_batches {
                self.column_decoders
                    .iter()
                    .zip(&self.decoder_output_columns)
                    .fold(record_batch, |batch, (decoder, &column)| {
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
