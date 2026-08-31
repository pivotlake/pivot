//! The error type for the Parquet write pipeline.
//!
//! It converts into dispatch's `UnaryError` (see the [`From`] impl below), so a
//! helper's `?` lifts straight onto a stage's `unary` error channel with the
//! typed cause preserved rather than stringified.

use crate::thrift::parquet_thrift::ParquetError;
use arrow_schema::{ArrowError, DataType};

#[derive(Debug, thiserror::Error)]
pub(crate) enum WriteError {
    /// An Arrow kernel or schema operation failed (sort, take, concat, min/max,
    /// `index_of`, building a typed scalar map, or encoding a row key).
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    /// Serializing a page header or the file footer.
    #[error("encoding Parquet thrift: {0}")]
    Thrift(#[from] ParquetError),
    /// Snappy-compressing a page body.
    #[error("snappy-compressing a page: {0}")]
    Snappy(#[from] snap::Error),
    /// Zstd-compressing a page body (the zstd crate reports through io::Error).
    #[error("zstd-compressing a page: {0}")]
    Zstd(#[from] std::io::Error),
    /// A column whose Arrow type the encoder doesn't handle.
    #[error("unsupported column type for Parquet encoding: {0:?}")]
    UnsupportedType(DataType),
    /// A catalog parquet type-mapping error, e.g. an arrow type with no Parquet
    /// physical type (from [`crate::arrow_to_parquet_physical`]).
    #[error(transparent)]
    Catalog(#[from] crate::ParquetTableError),
    /// A row group whose leaf yielded no pages — an internal invariant break.
    #[error("leaf {path} of a row group produced no pages")]
    MissingPages { path: String },
    /// A leaf whose values still hold nulls once its absent rows were dropped,
    /// which means the array is nullable where its field is declared required.
    #[error("a required leaf has {nulls} null(s)")]
    NullsInRequiredColumn { nulls: usize },
    /// An array did not have the Arrow type its column's schema declared.
    #[error("array downcast to {expected} failed")]
    Downcast { expected: &'static str },
}

pub(crate) type WriteResult<T> = Result<T, WriteError>;

/// Carry a `WriteError` out of a stage on dispatch's `unary` error channel as the
/// typed cause, so a helper's `?` lifts straight into a `UnaryResult` (mirrors
/// `catalog`'s reader). Uses the generic `Operator` variant — the one for
/// operators living outside dispatch.
impl From<WriteError> for dispatch::UnaryError {
    fn from(e: WriteError) -> Self {
        dispatch::UnaryError::Operator(Box::new(e))
    }
}
