//! Per-row-group decoder that accumulates decompressed pages and produces
//! Arrow [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) owns one [`RowGroupDecoder`] per
//! in-flight row group. Pages are inserted as they arrive from the
//! decompressor. When every projected column has enough buffered rows,
//! [`try_read`](RowGroupDecoder::try_read) decodes the next batch.

use crate::memory::SlabAllocator;
use crate::operations::unary::parquet::decoding::ScanEqualityPredicate;
use crate::operations::unary::parquet::decoding::column_decoders;
use crate::operations::unary::parquet::decoding::column_decoders::{
    BytesViewDecoder, ColumnDecoder, PrimitiveColumnDecoder,
};
use crate::operations::unary::parquet::types::metadata::QueryRowGroupMetadata;
use crate::operations::unary::parquet::types::page::DecompressedPage;
use crate::operations::unary::parquet::types::projection::Projection;
use crate::record_batch_metadata::with_row_group_metadata;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, UInt16Type,
};
use arrow_array::{ArrayRef, RecordBatch, Scalar};
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

/// Extracts the single native value from `scalar` if its array is a
/// `PrimitiveArray<T>`; otherwise `None` (logical/physical type mismatch, which
/// disables dictionary pruning for this column — always sound, the upstream
/// `Filter` still runs).
fn native_scalar<T: ArrowPrimitiveType>(scalar: &Scalar<ArrayRef>) -> Option<T::Native> {
    let (arr, _) = arrow_array::Datum::get(scalar);
    let primitive = arr.as_primitive_opt::<T>()?;
    (primitive.len() == 1).then(|| primitive.value(0))
}

/// Get a column decoder for a given data type. When `eq_const` is present and
/// downcasts to the column's native type, the constant is installed for
/// dictionary pruning.
fn column_decoder_for_type(
    data_type: &DataType,
    max_def_level: i16,
    eq_const: Option<&Scalar<ArrayRef>>,
) -> Result<Box<dyn ColumnDecoder>> {
    macro_rules! primitive {
        ($t:ty) => {{
            let mut decoder = PrimitiveColumnDecoder::<$t>::new(max_def_level);
            if let Some(value) = eq_const.and_then(native_scalar::<$t>) {
                decoder.set_eq_constant(value);
            }
            Box::new(decoder) as Box<dyn ColumnDecoder>
        }};
    }
    match data_type {
        DataType::UInt16 => Ok(primitive!(UInt16Type)),
        DataType::Int16 => Ok(primitive!(Int16Type)),
        DataType::Int32 => Ok(primitive!(Int32Type)),
        DataType::Int64 => Ok(primitive!(Int64Type)),
        DataType::Float32 => Ok(primitive!(Float32Type)),
        DataType::Float64 => Ok(primitive!(Float64Type)),
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
    /// Projected-column positions that carry a pushed-down equality constant
    /// whose column chunk is sound to prune by (all data pages dictionary
    /// encoded). When any such column's dictionary excludes its constant, the
    /// whole row group is pruned.
    prunable_columns: Vec<usize>,
    /// Set once a `prunable_columns` entry's dictionary is found to exclude
    /// its constant: the row group cannot contain a matching row, so it emits
    /// nothing and is treated as exhausted.
    pruned: bool,
}

impl RowGroupDecoder {
    pub fn new(
        row_group_metadata: QueryRowGroupMetadata,
        schema: SchemaRef,
        projection: &Projection,
        batch_size: usize,
        add_row_group_metadata: bool,
        eq_predicates: &[ScanEqualityPredicate],
    ) -> Result<Self> {
        let columns = row_group_metadata.columns();
        let fields = schema.fields();
        let mut prunable_columns = Vec::new();
        let column_decoders = projection
            .column_indices
            .iter()
            .enumerate()
            .map(|(schema_idx, &col_idx)| {
                let predicate = eq_predicates.iter().find(|p| p.column_idx == col_idx);
                // Only a column chunk whose data pages are all dictionary
                // encoded can be soundly pruned by dictionary contents.
                if predicate.is_some() && columns[col_idx].data_pages_all_dictionary {
                    prunable_columns.push(schema_idx);
                }
                column_decoder_for_type(
                    fields[schema_idx].data_type(),
                    columns[col_idx].max_def_level,
                    predicate.map(|p| &p.value),
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
            prunable_columns,
            pruned: false,
        })
    }

    /// Returns the global row-group index this decoder is responsible for.
    pub fn row_group_idx(&self) -> usize {
        self.row_group_idx
    }

    /// Returns `true` when the row group has been pruned (no row can match a
    /// pushed-down equality predicate) or all its rows have been emitted.
    pub fn exhausted(&self) -> bool {
        self.pruned || self.total - self.row_offset == 0
    }

    /// Returns `true` if this row group was pruned by dictionary pushdown and
    /// should be dropped without emitting any rows.
    pub fn pruned(&self) -> bool {
        self.pruned
    }

    /// Routes a decompressed page to the appropriate column decoder, then
    /// re-evaluates dictionary pruning (a just-loaded dictionary page may
    /// exclude a pushed-down constant, allowing the whole row group to be
    /// dropped before its data pages are decoded).
    pub fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator) {
        self.column_decoders[page.column_idx].insert_page(page, allocator);
        if !self.pruned
            && self
                .prunable_columns
                .iter()
                .any(|&pos| self.column_decoders[pos].dict_excludes_constant() == Some(true))
        {
            self.pruned = true;
        }
    }

    /// Attempts to produce the next [`RecordBatch`].
    ///
    /// Returns `Ok(Some(batch))` if every column decoder has enough buffered
    /// rows, `Ok(None)` if more pages are needed, or an error if decoding
    /// fails.
    pub fn try_read(&mut self, allocator: &mut SlabAllocator) -> Result<Option<RecordBatch>> {
        if self.pruned {
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
