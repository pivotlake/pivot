//! Traits and types for decoding individual Parquet columns into Arrow arrays.
//!
//! The core abstraction is [`ColumnDecoder`], an object-safe trait used by
//! [`RowGroupDecoder`](super::row_group_decoder::RowGroupDecoder) to decode
//! each projected column independently. The concrete, generic implementation
//! lives in [`TypedColumnDecoder`] (in the [`typed`] submodule), parameterised
//! by three helper traits that together describe how to decode a particular
//! Parquet type:
//!
//! - [`ArrayBuilder`] — accumulates decoded values and produces an Arrow array.
//! - [`DecodePlain`] — reads plain-encoded values from raw page bytes.
//! - [`Dict`] — builds and queries a dictionary for RLE-dictionary pages.
//!
//! Concrete column decoders are type aliases over `TypedColumnDecoder`:
//! - [`primitive::PrimitiveColumnDecoder`] for fixed-width numeric types.
//! - [`bytes_view::BytesViewDecoder`] for variable-length string / binary types.

mod bytes_view;
pub use bytes_view::BytesViewDecoder;

mod levels;

mod primitive;
pub use primitive::PrimitiveColumnDecoder;

mod rle;

mod typed;
pub use typed::TypedColumnDecoder;

use crate::memory::{ReaderPosition, SlabAllocator};
use crate::operations::unary::parquet::types::page::DecompressedPage;
use crate::operations::unary::parquet::types::thrift::general::Encoding;
use arrow_array::ArrayRef;
use bytes::Bytes;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("No pages are ready")]
    NoPagesReady,
    #[error("Dict page empty")]
    DictPageEmpty,
    #[error("Unsupported encoding: {0}")]
    UnsupportedEncoding(Encoding),
    #[error("Nullable columns are not yet supported (encountered nulls in data page)")]
    NullableColumnsNotSupported,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Object-safe interface for decoding a single column from decompressed pages
/// into Arrow arrays.
///
/// Used as `Box<dyn ColumnDecoder>` inside
/// [`RowGroupDecoder`](super::row_group_decoder::RowGroupDecoder) so that
/// columns of different types can be stored in a single `Vec`.
pub trait ColumnDecoder {
    /// Returns `true` if at least `size` rows can be decoded from the pages
    /// buffered so far.
    fn available(&self) -> usize;

    /// Stores a decompressed page (data, dictionary, or skipped) for later
    /// decoding.
    fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator);

    /// Decodes the next `size` rows into an Arrow array.
    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef>;
}

// `ArrayBuilder` (and the primitive builder) now live in `crate::arrays` so the
// GROUP BY output can share them; re-exported here for the parquet decoders.
pub use crate::arrays::ArrayBuilder;

/// Reads plain-encoded values from raw page bytes into an [`ArrayBuilder`].
pub trait DecodePlain {
    type Builder: ArrayBuilder;

    /// Creates a decoder starting at `position` within `data`.
    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self;

    /// Decodes `size` values into `builder`.
    fn read(&mut self, builder: &mut Self::Builder, size: usize);

    /// Advances past `size` values without decoding them.
    fn skip(&mut self, size: usize);
}

/// Builds and queries a dictionary for RLE-dictionary-encoded columns.
pub trait Dict {
    type Builder: ArrayBuilder;
    type Item;

    /// Builds the dictionary from raw page bytes containing `size` entries.
    fn new(data: Vec<Bytes>, size: usize, allocator: &mut SlabAllocator) -> Self;

    /// Looks up the value at `idx` in the dictionary.
    fn entry(&self, idx: usize) -> Self::Item;

    /// Registers dictionary buffers onto the builder (e.g. for StringView
    /// block tracking). No-op by default.
    fn register_onto(&self, _builder: &mut Self::Builder) {}
}
