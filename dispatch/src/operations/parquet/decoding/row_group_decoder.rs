//! Per-row-group decoder that accumulates decompressed pages and produces
//! Arrow [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) owns one [`RowGroupDecoder`] per
//! in-flight row group. Pages are inserted as they arrive from the
//! decompressor. When every projected column has enough buffered rows,
//! [`try_read`](RowGroupDecoder::try_read) decodes the next batch.

use crate::memory::SlabAllocator;
use crate::operations::parquet::decoding::column_decoders;
use crate::operations::parquet::decoding::column_decoders::{
    BytesViewDecoder, ColumnDecoder, PrimitiveColumnDecoder,
};
use crate::operations::parquet::types::metadata::QueryRowGroupMetadata;
use crate::operations::parquet::types::page::DecompressedPage;
use crate::operations::parquet::types::projection::Projection;
use crate::record_batch_metadata::with_row_group_metadata;
use arrow_array::RecordBatch;
use arrow_array::types::{Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, UInt16Type};
use arrow_schema::{ArrowError, DataType, SchemaRef};
use std::cmp::min;
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

/// Get a column decoder for a given data type.
fn column_decoder_for_type(
    data_type: &DataType,
    max_def_level: i16,
) -> Result<Box<dyn ColumnDecoder>> {
    match data_type {
        DataType::UInt16 => Ok(Box::new(PrimitiveColumnDecoder::<UInt16Type>::new(
            max_def_level,
        ))),
        DataType::Int16 => Ok(Box::new(PrimitiveColumnDecoder::<Int16Type>::new(
            max_def_level,
        ))),
        DataType::Int32 => Ok(Box::new(PrimitiveColumnDecoder::<Int32Type>::new(
            max_def_level,
        ))),
        DataType::Int64 => Ok(Box::new(PrimitiveColumnDecoder::<Int64Type>::new(
            max_def_level,
        ))),
        DataType::Float32 => Ok(Box::new(PrimitiveColumnDecoder::<Float32Type>::new(
            max_def_level,
        ))),
        DataType::Float64 => Ok(Box::new(PrimitiveColumnDecoder::<Float64Type>::new(
            max_def_level,
        ))),
        DataType::Utf8View
        | DataType::BinaryView
        | DataType::Utf8
        | DataType::Binary
        | DataType::LargeUtf8
        | DataType::LargeBinary => Ok(Box::new(BytesViewDecoder::new(max_def_level))),
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
}

impl RowGroupDecoder {
    pub fn new(
        row_group_metadata: QueryRowGroupMetadata,
        schema: SchemaRef,
        projection: &Projection,
        batch_size: usize,
        add_row_group_metadata: bool,
    ) -> Result<Self> {
        let columns = row_group_metadata.columns();
        let fields = schema.fields();
        let column_decoders = projection
            .column_indices
            .iter()
            .enumerate()
            .map(|(schema_idx, &col_idx)| {
                column_decoder_for_type(
                    fields[schema_idx].data_type(),
                    columns[col_idx].max_def_level,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            row_group_idx: row_group_metadata.index(),
            column_decoders,
            schema,
            batch_size,
            total: row_group_metadata
                .filtered_indices()
                .as_ref()
                .map(|f| f.len())
                .unwrap_or(row_group_metadata.num_rows() as usize),
            row_offset: 0,
            add_row_group_metadata,
        })
    }

    /// Returns the global row-group index this decoder is responsible for.
    pub fn row_group_idx(&self) -> usize {
        self.row_group_idx
    }

    /// Returns `true` when all rows in this row group have been emitted.
    pub fn exhausted(&self) -> bool {
        self.total - self.row_offset == 0
    }

    /// Routes a decompressed page to the appropriate column decoder.
    pub fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator) {
        self.column_decoders[page.column_idx].insert_page(page, allocator)
    }

    /// Attempts to produce the next [`RecordBatch`].
    ///
    /// Returns `Ok(Some(batch))` if every column decoder has enough buffered
    /// rows, `Ok(None)` if more pages are needed, or an error if decoding
    /// fails.
    pub fn try_read(&mut self, allocator: &mut SlabAllocator) -> Result<Option<RecordBatch>> {
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
            let columns = self
                .column_decoders
                .iter_mut()
                .map(|c| c.read(allocator, available).map_err(Error::from))
                .collect::<Result<Vec<_>>>()?;
            let record_batch = RecordBatch::try_new(self.schema.clone(), columns)?;
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
