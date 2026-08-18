//! Traits and types for decoding individual Parquet leaf columns into Arrow
//! arrays.
//!
//! The core abstraction is [`LeafDecoder`], an object-safe trait used by
//! [`ColumnDecoder`](super::column_decoder::ColumnDecoder) to decode each of a
//! projected column's leaf column chunks independently. The concrete, generic
//! implementation lives in [`TypedLeafDecoder`] (in the [`typed`] submodule),
//! parameterised by three helper traits that together describe how to decode a
//! particular Parquet type:
//!
//! - [`ArrayBuilder`] — accumulates decoded values and produces an Arrow array.
//! - [`DecodePlain`] — reads plain-encoded values from raw page bytes.
//! - [`Dict`] - queries a dictionary for RLE-dictionary pages, built via
//!   [`DictFromBytes`] or [`DictFromVecBytes`] depending on the dictionary
//!   page's buffer shape.
//!
//! Concrete leaf decoders are type aliases over `TypedLeafDecoder`:
//! - [`primitive::PrimitiveLeafDecoder`] for fixed-width numeric types.
//! - [`bytes_view::BytesViewDecoder`] for variable-length string / binary types.

pub(crate) mod bytes_view;
pub use bytes_view::BytesViewDecoder;

mod decimal;
pub use decimal::{DecimalStorage, decimal_decoder};

pub(crate) mod delta_binary_packed;
pub use delta_binary_packed::{DecimalDeltaDecoder, DeltaDecoder, FromDelta};

mod levels;

mod primitive;
pub use primitive::{PrimitiveLeafDecoder, TimestampMicrosecondLeafDecoder};

mod rle;

mod typed;
pub use typed::TypedLeafDecoder;

use crate::parquet::types::page::DecompressedPage;
use crate::parquet::types::thrift::general::Encoding;
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use bytes::Bytes;
use dispatch::memory::{ReaderPosition, SlabAllocator};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("No pages are ready")]
    NoPagesReady,
    #[error("Dict page empty")]
    DictPageEmpty,
    #[error("Unsupported encoding: {0}")]
    UnsupportedEncoding(Encoding),
    /// A decimal column whose chunk uses a physical storage the decoders do
    /// not read. The column's arrow type itself is supported; footer parsing
    /// only admits the storages the decoders handle, so this can only be
    /// reached by metadata that is inconsistent with the schema.
    #[error(
        "decimal column {data_type} stored as parquet physical type \
         {physical_type} with type_length {type_length:?} is not supported"
    )]
    UnsupportedDecimalStorage {
        data_type: arrow_schema::DataType,
        physical_type: i32,
        type_length: Option<i32>,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Object-safe interface for decoding a single Parquet leaf column from
/// decompressed pages into Arrow arrays.
///
/// Used as `Box<dyn LeafDecoder>` inside
/// [`ColumnDecoder`](super::column_decoder::ColumnDecoder) so that leaves of
/// different types can be stored in a single `Vec`.
pub trait LeafDecoder {
    /// Returns `true` if at least `size` rows can be decoded from the pages
    /// buffered so far.
    fn available(&self) -> usize;

    /// Stores a decompressed page (data, dictionary, or skipped) for later
    /// decoding.
    fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator);

    /// Decodes the next `size` rows into an Arrow array.
    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef>;

    /// Installs a pushed-down equality constant. A constant whose type does
    /// not match the column is ignored (the query's `Filter` still applies
    /// the condition). Once the dictionary is built the constant decides
    /// row-group pruning and scan-side batch filtering.
    fn set_eq_constant(&mut self, value: &Scalar<ArrayRef>);

    /// Whether the loaded dictionary is known to exclude the pushed-down
    /// equality constant, meaning no row of this column can match and the row
    /// group can be pruned. `false` until a dictionary page proves otherwise
    /// (no constant pushed, dictionary not loaded yet, or constant present).
    fn dict_excludes_eq_constant(&self) -> bool {
        false
    }

    /// Drops rows that cannot pass this column's pushed-down equality
    /// constant from a decoded batch, where `column` is this column's
    /// position in the batch. This only ever removes rows the query would
    /// discard anyway, so the default - returning the batch untouched - is
    /// always valid, and so is any partial filtering. Byte-view columns use
    /// it to pre-filter batches with a cheap view comparison instead of
    /// leaving all the string comparisons to the query's `Filter` (see
    /// [`Dict::filter_record_batch_by_const`]).
    fn fast_filter_record_batch(&self, batch: RecordBatch, _column: usize) -> RecordBatch {
        batch
    }
}

// `ArrayBuilder` (and the primitive builder) now live in `dispatch::arrays` so the
// GROUP BY output can share them; re-exported here for the parquet decoders.
pub use dispatch::arrays::ArrayBuilder;

/// Reads plain-encoded values from raw page bytes into an [`ArrayBuilder`].
pub trait DecodePlain {
    type Builder: ArrayBuilder;

    /// This column type's delta decoder: `DELTA_BINARY_PACKED` for the
    /// integer and decimal columns, `DELTA_LENGTH_BYTE_ARRAY` for the byte-view
    /// ones. Construction answers `None` when the page's own type rules the
    /// encoding out (a float column, or a decimal stored as fixed-length
    /// bytes), which reports it as unsupported rather than decoding nonsense.
    type Delta: DecodeDelta<Builder = Self::Builder>;

    /// Creates a decoder starting at `position` within `data`.
    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self;

    /// Decodes `size` values into `builder`.
    fn read(&mut self, builder: &mut Self::Builder, size: usize);

    /// Advances past `size` values without decoding them.
    fn skip(&mut self, size: usize);
}

/// Reads `DELTA_BINARY_PACKED` values from raw page bytes into an
/// [`ArrayBuilder`], the delta-encoded counterpart of [`DecodePlain`].
pub trait DecodeDelta: Sized {
    type Builder: ArrayBuilder;

    /// The encoding this decoder reads. A column type has exactly one delta
    /// form, so naming it here is what lets the page dispatch pair them up and
    /// reject a page whose encoding does not match the column's type.
    const ENCODING: Encoding;

    /// Creates a decoder over a page starting at `position`, reading its
    /// header. `None` when the encoding does not apply to this column type.
    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Option<Self>;

    /// Decodes `size` values into `builder`.
    fn read(&mut self, builder: &mut Self::Builder, size: usize);

    /// Advances past `size` values. The values are still decoded, because each
    /// delta is relative to the one before it.
    fn skip(&mut self, size: usize);
}

/// Queries a dictionary for RLE-dictionary-encoded columns.
///
/// Construction lives in [`DictFromBytes`] / [`DictFromVecBytes`], keyed by
/// whether the dictionary page's bytes sit in one contiguous buffer or are
/// scattered across several.
pub trait Dict {
    type Builder: ArrayBuilder;
    type Item;
    /// How a pushed-down equality constant is represented for this
    /// dictionary flavour. Primitive dictionaries take the native value;
    /// byte-view dictionaries take the raw bytes, because the value's Arrow
    /// view can only be resolved against a dictionary that has been built.
    type EqConstant;

    /// Converts a pushed-down constant into this dictionary flavour's own
    /// representation, or `None` when the scalar's type does not match the
    /// column (which simply forgoes the pushdown; the query's `Filter` still
    /// applies the condition).
    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<Self::EqConstant>;

    /// Whether `needle` might appear among the first `size` raw entries of
    /// `data`, checked without building the dictionary. Only a definite `no`
    /// matters - it prunes the row group and the dictionary is never
    /// materialized - so answering `true` is always sound. The default cannot
    /// rule anything out; implementations override it to enable the pushdown.
    fn maybe_contains(_data: &[Bytes], _size: usize, _needle: &Self::EqConstant) -> bool {
        true
    }

    /// Number of entries in the dictionary.
    fn len(&self) -> usize;

    /// Looks up the value at `idx` in the dictionary.
    fn entry(&self, idx: usize) -> Self::Item;

    /// Looks up the value at `idx` without validating it.
    ///
    /// # Safety
    ///
    /// `idx` must be `< self.len()`. The default just calls
    /// [`entry`](Self::entry); an implementation whose storage bounds-checks
    /// overrides this so the decode hot loop, whose caller proves every key
    /// in range up front, skips the per-element check.
    unsafe fn entry_unchecked(&self, idx: usize) -> Self::Item {
        self.entry(idx)
    }

    /// Registers dictionary buffers onto the builder (e.g. for StringView
    /// block tracking). No-op by default.
    fn register_onto(&self, _builder: &mut Self::Builder) {}

    /// Drops rows of `batch` whose value in `column` cannot equal `needle`,
    /// or returns the batch untouched when the dictionary cannot decide that
    /// cheaply (the default). Only rows that provably fail the equality may
    /// be dropped; keeping extra rows is always sound because the query's
    /// `Filter` re-applies every condition.
    fn filter_record_batch_by_const(
        &self,
        batch: RecordBatch,
        _column: usize,
        _needle: &Self::EqConstant,
    ) -> RecordBatch {
        batch
    }
}

/// A [`Dict`] buildable from a dictionary page held in one contiguous buffer.
///
/// A contiguous page is at most one 2MB ring slot, so implementations can
/// pick a representation that relies on that bound, e.g. entries in a single
/// [`SlabBuffer`](dispatch::memory::SlabBuffer) indexed by plain pointer
/// arithmetic with no per-element slab lookup.
pub trait DictFromBytes: Dict {
    /// Builds the dictionary from one buffer of raw page bytes containing
    /// `size` entries.
    fn new_from_bytes(data: Bytes, size: usize, allocator: &mut SlabAllocator) -> Self;
}

/// A [`Dict`] buildable from a dictionary page scattered across buffers.
pub trait DictFromVecBytes: Dict {
    /// Builds the dictionary from scattered raw page bytes containing `size`
    /// entries in total.
    fn new_from_vec_bytes(data: Vec<Bytes>, size: usize, allocator: &mut SlabAllocator) -> Self;
}
